use steel_registry::blocks::block_state_ext::BlockStateExt;
use steel_utils::{BlockPos, BlockStateId, ChunkPos, SectionPos};

use super::{
    BlockLightVector, CachedLightBlock, LIGHT_BLOCKED, LightAxisDirection, LightCacheLayout,
    LightDirectionSet, LightLayer, LightLayerEdit, LightQueueFlags, LightSectionEmptinessChange,
    LightSectionReadCache, LightWorkset, MAX_LIGHT_LEVEL, PackedLightPropagationQueues,
    PackedLightQueueEntry, get_light_block_into, get_light_opacity, light_occlusion_shape,
    voxel_traversal::{
        dominant_direction, for_each_adjacent_block_offset, is_diagonal_block_offset,
        line_voxels_between, next_voxel_past, next_voxel_toward, ray_passes_through,
    },
};

/// Error returned when a block-light propagation context is built from mismatched caches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockLightPropagationContextError {
    /// Block-light propagation requires a block light edit cache.
    WrongLayer {
        /// Layer supplied by the edit cache.
        layer: LightLayer,
    },
    /// Section and light caches were built from different cache layouts.
    LayoutMismatch {
        /// Layout used by the section cache.
        section_layout: Box<LightCacheLayout>,
        /// Layout used by the light cache.
        light_layout: Box<LightCacheLayout>,
    },
    /// The workset does not contain its center chunk.
    MissingCenterChunk {
        /// Missing center chunk position.
        chunk_pos: ChunkPos,
    },
}

impl BlockLightPropagationContextError {
    fn layout_mismatch(section_layout: LightCacheLayout, light_layout: LightCacheLayout) -> Self {
        Self::LayoutMismatch {
            section_layout: Box::new(section_layout),
            light_layout: Box::new(light_layout),
        }
    }
}

/// Sections whose visible block-light data changed during a scoped update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockLightUpdateResult {
    /// Light sections that should be reported to the world/chunk update layer.
    pub updated_sections: Vec<SectionPos>,
}

/// Whether chunk block-light generation must validate edge consistency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockLightChunkEdgeChecks {
    /// Seed sources and validate this chunk's horizontal edges against neighbors.
    Required,
    /// Trust existing neighboring light and pull initialized edge levels inward.
    Skipped,
}

/// Runs ScalableLux-style block-light propagation for changed blocks in a scoped workset.
///
/// This is the block-light equivalent of `ScalableLux` `propagateBlockChanges`
/// plus publishing edited sections. It assumes the caller already created a
/// cache window around the affected chunk and will deliver returned section
/// updates to the world/chunk notification layer.
pub fn propagate_block_light_changes(
    workset: &LightWorkset,
    positions: impl IntoIterator<Item = BlockPos>,
) -> Result<BlockLightUpdateResult, BlockLightPropagationContextError> {
    propagate_block_light_changes_with_empty_sections(workset, positions, [])
}

/// Runs block-light propagation after applying real section emptiness transitions.
pub fn propagate_block_light_changes_with_empty_sections(
    workset: &LightWorkset,
    positions: impl IntoIterator<Item = BlockPos>,
    empty_sections: impl IntoIterator<Item = LightSectionEmptinessChange>,
) -> Result<BlockLightUpdateResult, BlockLightPropagationContextError> {
    let empty_sections = empty_sections.into_iter().collect::<Vec<_>>();

    workset.with_chunk_read_cache(|chunk_cache| {
        let layout = chunk_cache.layout();
        // ScalableLux drops queued dynamic changes once the center chunk leaves the light cache.
        let Some(center_slot) = layout.cached_chunk(layout.center_chunk()) else {
            return Ok(BlockLightUpdateResult {
                updated_sections: Vec::new(),
            });
        };
        if chunk_cache.chunk(center_slot).is_none() {
            return Ok(BlockLightUpdateResult {
                updated_sections: Vec::new(),
            });
        }

        chunk_cache.with_section_read_cache(|section_cache| {
            chunk_cache.with_light_edit(LightLayer::Block, |mut light_edit| {
                let mut queues = PackedLightPropagationQueues::new();

                {
                    apply_block_empty_section_changes(
                        section_cache,
                        &mut light_edit,
                        &empty_sections,
                    );
                    let mut context = BlockLightPropagationContext::new(
                        section_cache,
                        &mut light_edit,
                        &mut queues,
                    )?;
                    for position in positions {
                        context.check_block(position);
                    }
                    context.perform_light_decrease();
                    context.check_chunk_edges(layout.center_chunk());
                }

                let mut updated_sections = Vec::new();
                light_edit.commit(None, |section_pos| updated_sections.push(section_pos));
                Ok(BlockLightUpdateResult { updated_sections })
            })
        })
    })
}

fn apply_block_empty_section_changes(
    sections: &LightSectionReadCache<'_>,
    light: &mut LightLayerEdit<'_>,
    empty_sections: &[LightSectionEmptinessChange],
) -> usize {
    let mut changed_chunks = Vec::new();
    for change in empty_sections {
        light.set_section_empty(change.section_pos, change.empty);
        let chunk_pos = ChunkPos::new(change.section_pos.x(), change.section_pos.z());
        if !changed_chunks.contains(&chunk_pos) {
            changed_chunks.push(chunk_pos);
        }
    }

    let mut initialized = 0;
    for chunk_pos in changed_chunks {
        initialized += sync_block_empty_light_sections(sections, light, chunk_pos);
    }
    initialized
}

/// Seeds and propagates block light for the center chunk of a scoped workset.
///
/// This matches `ScalableLux` `BlockStarLightEngine.lightChunk`: source blocks in
/// the center chunk are seeded, then the caller chooses between validating edge
/// consistency or pulling already-initialized neighbor levels inward.
pub fn propagate_block_light_chunk(
    workset: &LightWorkset,
    edge_checks: BlockLightChunkEdgeChecks,
) -> Result<BlockLightUpdateResult, BlockLightPropagationContextError> {
    workset.with_chunk_read_cache(|chunk_cache| {
        let layout = chunk_cache.layout();
        let Some(center_slot) = layout.cached_chunk(layout.center_chunk()) else {
            return Err(BlockLightPropagationContextError::MissingCenterChunk {
                chunk_pos: layout.center_chunk(),
            });
        };
        let Some(center_chunk) = chunk_cache.chunk(center_slot) else {
            return Err(BlockLightPropagationContextError::MissingCenterChunk {
                chunk_pos: layout.center_chunk(),
            });
        };
        let sources = center_chunk.block_light_sources();

        chunk_cache.with_section_read_cache(|section_cache| {
            chunk_cache.with_light_edit(LightLayer::Block, |mut light_edit| {
                let mut queues = PackedLightPropagationQueues::new();

                {
                    light_edit.reset_chunk_sections_to_missing(layout.center_chunk());
                    sync_block_empty_light_sections(
                        section_cache,
                        &mut light_edit,
                        layout.center_chunk(),
                    );
                    let mut context = BlockLightPropagationContext::new(
                        section_cache,
                        &mut light_edit,
                        &mut queues,
                    )?;
                    context.seed_block_light_sources(sources);
                    match edge_checks {
                        BlockLightChunkEdgeChecks::Required => {
                            context.perform_light_increase();
                            context.check_chunk_edges(layout.center_chunk());
                        }
                        BlockLightChunkEdgeChecks::Skipped => {
                            context.propagate_neighbor_levels(layout.center_chunk());
                            context.perform_light_increase();
                        }
                    }
                }

                let mut updated_sections = Vec::new();
                light_edit.commit(None, |section_pos| updated_sections.push(section_pos));
                Ok(BlockLightUpdateResult { updated_sections })
            })
        })
    })
}

/// Force-synchronizes block-light sections for an already-lit loaded chunk.
///
/// This matches the block layer of `ScalableLux` `forceLoadInChunk`: existing
/// light data is kept, empty-section state is synchronized, and dirty visible
/// sections are published before the later edge-check pass.
pub fn force_load_block_light_chunk(
    workset: &LightWorkset,
) -> Result<BlockLightUpdateResult, BlockLightPropagationContextError> {
    workset.with_chunk_read_cache(|chunk_cache| {
        let layout = ensure_center_chunk(chunk_cache)?;

        chunk_cache.with_section_read_cache(|section_cache| {
            chunk_cache.with_light_edit(LightLayer::Block, |mut light_edit| {
                sync_block_empty_light_sections(
                    section_cache,
                    &mut light_edit,
                    layout.center_chunk(),
                );

                let mut updated_sections = Vec::new();
                light_edit.commit(None, |section_pos| updated_sections.push(section_pos));
                Ok(BlockLightUpdateResult { updated_sections })
            })
        })
    })
}

/// Validates already-loaded block-light chunk edges without resetting sections.
///
/// This matches `ScalableLux` `checkBlockEdges`: the force-load pass has already
/// synchronized empty-section state, so this pass only checks horizontal
/// consistency against loaded neighbors and publishes its own dirty sections.
pub fn check_block_light_chunk_edges(
    workset: &LightWorkset,
) -> Result<BlockLightUpdateResult, BlockLightPropagationContextError> {
    workset.with_chunk_read_cache(|chunk_cache| {
        let layout = ensure_center_chunk(chunk_cache)?;

        chunk_cache.with_section_read_cache(|section_cache| {
            chunk_cache.with_light_edit(LightLayer::Block, |mut light_edit| {
                let mut queues = PackedLightPropagationQueues::new();

                {
                    let mut context = BlockLightPropagationContext::new(
                        section_cache,
                        &mut light_edit,
                        &mut queues,
                    )?;
                    context.check_chunk_edges(layout.center_chunk());
                }

                let mut updated_sections = Vec::new();
                light_edit.commit(None, |section_pos| updated_sections.push(section_pos));
                Ok(BlockLightUpdateResult { updated_sections })
            })
        })
    })
}

/// Loads already-persisted block light and validates chunk edges without resetting sections.
///
/// This is the complete block-layer `lit == true` path: force-load
/// empty-section state first, then run the edge-check pass.
pub fn load_block_light_chunk(
    workset: &LightWorkset,
) -> Result<BlockLightUpdateResult, BlockLightPropagationContextError> {
    let mut updated_sections = force_load_block_light_chunk(workset)?.updated_sections;
    updated_sections.extend(check_block_light_chunk_edges(workset)?.updated_sections);
    Ok(BlockLightUpdateResult { updated_sections })
}

fn ensure_center_chunk(
    chunk_cache: &super::LightChunkReadCache<'_>,
) -> Result<LightCacheLayout, BlockLightPropagationContextError> {
    let layout = chunk_cache.layout();
    let Some(center_slot) = layout.cached_chunk(layout.center_chunk()) else {
        return Err(BlockLightPropagationContextError::MissingCenterChunk {
            chunk_pos: layout.center_chunk(),
        });
    };
    if chunk_cache.chunk(center_slot).is_none() {
        return Err(BlockLightPropagationContextError::MissingCenterChunk {
            chunk_pos: layout.center_chunk(),
        });
    }

    Ok(layout)
}

fn sync_block_empty_light_sections(
    sections: &LightSectionReadCache<'_>,
    light: &mut LightLayerEdit<'_>,
    chunk_pos: ChunkPos,
) -> usize {
    let layout = sections.layout();
    let mut initialized = 0;

    for section_y in
        (layout.range().min_chunk_section_y()..layout.range().max_chunk_section_y_exclusive()).rev()
    {
        let section_pos = SectionPos::new(chunk_pos.0.x, section_y, chunk_pos.0.y);
        if !section_is_non_empty(sections, light, section_pos) {
            continue;
        }

        for offset_z in -1..=1 {
            for offset_x in -1..=1 {
                for offset_y in (-1..=1).rev() {
                    let target = SectionPos::new(
                        chunk_pos.0.x + offset_x,
                        section_y + offset_y,
                        chunk_pos.0.y + offset_z,
                    );
                    if light.set_section_non_missing(target) {
                        initialized += 1;
                    }
                }
            }
        }
    }

    for offset_z in -1..=1 {
        for offset_x in -1..=1 {
            let target_chunk = ChunkPos::new(chunk_pos.0.x + offset_x, chunk_pos.0.y + offset_z);

            for section_y in
                (layout.range().min_section_y()..layout.range().max_section_y_exclusive()).rev()
            {
                let section_pos = SectionPos::new(target_chunk.0.x, section_y, target_chunk.0.y);
                match section_neighborhood_all_empty_if_known(sections, target_chunk, section_y) {
                    Some(true) => {
                        light.set_section_internal(section_pos);
                    }
                    Some(false) => {
                        if light.set_section_non_missing(section_pos) {
                            initialized += 1;
                        }
                    }
                    None => {
                        if !section_neighborhood_all_empty(sections, light, target_chunk, section_y)
                            && light.set_section_non_missing(section_pos)
                        {
                            initialized += 1;
                        }
                    }
                }
            }
        }
    }

    initialized
}

fn section_neighborhood_all_empty(
    sections: &LightSectionReadCache<'_>,
    light: &LightLayerEdit<'_>,
    chunk_pos: ChunkPos,
    section_y: i32,
) -> bool {
    for offset_y in -1..=1 {
        let neighbor_y = section_y + offset_y;
        if neighbor_y < sections.layout().range().min_chunk_section_y()
            || neighbor_y >= sections.layout().range().max_chunk_section_y_exclusive()
        {
            continue;
        }

        for offset_z in -1..=1 {
            for offset_x in -1..=1 {
                let section_pos = SectionPos::new(
                    chunk_pos.0.x + offset_x,
                    neighbor_y,
                    chunk_pos.0.y + offset_z,
                );
                if section_is_non_empty(sections, light, section_pos) {
                    return false;
                }
            }
        }
    }

    true
}

fn section_neighborhood_all_empty_if_known(
    sections: &LightSectionReadCache<'_>,
    chunk_pos: ChunkPos,
    section_y: i32,
) -> Option<bool> {
    for offset_y in -1..=1 {
        let neighbor_y = section_y + offset_y;
        if neighbor_y < sections.layout().range().min_chunk_section_y()
            || neighbor_y >= sections.layout().range().max_chunk_section_y_exclusive()
        {
            continue;
        }

        for offset_z in -1..=1 {
            for offset_x in -1..=1 {
                let section_pos = SectionPos::new(
                    chunk_pos.0.x + offset_x,
                    neighbor_y,
                    chunk_pos.0.y + offset_z,
                );
                let empty = sections.section_empty(section_pos)?;
                if !empty {
                    return Some(false);
                }
            }
        }
    }

    Some(true)
}

fn section_is_non_empty(
    sections: &LightSectionReadCache<'_>,
    light: &LightLayerEdit<'_>,
    section_pos: SectionPos,
) -> bool {
    if let Some(empty) = sections.section_empty(section_pos) {
        return !empty;
    }

    if let Some(empty) = light.section_empty(section_pos) {
        return !empty;
    }

    sections.has_non_empty_section(section_pos)
}

/// ScalableLux-style block-light propagation over scoped Steel light caches.
///
/// This keeps the queue algorithm close to `ScalableLux` while avoiding long-lived
/// references into chunks: the caller owns the scoped section and light caches,
/// and this context only borrows them for one propagation pass.
pub struct BlockLightPropagationContext<'a, 'sections, 'light> {
    layout: LightCacheLayout,
    sections: &'a LightSectionReadCache<'sections>,
    light: &'a mut LightLayerEdit<'light>,
    queues: &'a mut PackedLightPropagationQueues,
}

impl<'a, 'sections, 'light> BlockLightPropagationContext<'a, 'sections, 'light> {
    /// Creates a block-light propagation context from matching scoped caches.
    pub fn new(
        sections: &'a LightSectionReadCache<'sections>,
        light: &'a mut LightLayerEdit<'light>,
        queues: &'a mut PackedLightPropagationQueues,
    ) -> Result<Self, BlockLightPropagationContextError> {
        if light.layer() != LightLayer::Block {
            return Err(BlockLightPropagationContextError::WrongLayer {
                layer: light.layer(),
            });
        }

        if sections.layout() != light.layout() {
            return Err(BlockLightPropagationContextError::layout_mismatch(
                sections.layout(),
                light.layout(),
            ));
        }

        Ok(Self {
            layout: light.layout(),
            sections,
            light,
            queues,
        })
    }

    /// Handles one block-light source/opacity change, matching `ScalableLux` `checkBlock`.
    ///
    /// Returns false when the changed block is outside this cache window.
    pub fn check_block(&mut self, block_pos: BlockPos) -> bool {
        let Some(cached_block) = self.layout.cached_block(block_pos) else {
            return false;
        };

        let current_level = self.light.get(cached_block);
        let block_state = self.sections.get_block_state(cached_block);
        let emitted_level = block_state.get_light_emission() & MAX_LIGHT_LEVEL;

        self.light.set(cached_block, emitted_level);
        if emitted_level != 0 {
            self.light.set_block_light_vector(cached_block, BlockLightVector::ZERO);
            self.enqueue_increase(
                block_pos,
                emitted_level,
                LightDirectionSet::all(),
                Self::shape_flags(block_state),
            );
        } else {
            self.light.set_block_light_vector(cached_block, BlockLightVector::ZERO);
        }

        self.enqueue_decrease(
            block_pos,
            current_level,
            LightDirectionSet::all(),
            LightQueueFlags::EMPTY,
        );
        true
    }

    /// Seeds block-light sources in `ScalableLux` local-index order.
    pub fn seed_block_light_sources(&mut self, positions: impl IntoIterator<Item = BlockPos>) {
        for position in positions {
            self.seed_block_light_source(position);
        }
    }

    /// Pulls horizontal neighbor levels into this chunk's increase queue.
    pub fn propagate_neighbor_levels(&mut self, chunk_pos: ChunkPos) {
        for section_y in (self.layout.range().min_section_y()
            ..self.layout.range().max_section_y_exclusive())
            .rev()
        {
            let section_pos = SectionPos::new(chunk_pos.0.x, section_y, chunk_pos.0.y);
            if !self.light.has_non_missing_section(section_pos) {
                continue;
            }

            for direction in LightAxisDirection::HORIZONTAL {
                self.propagate_neighbor_level_section(chunk_pos, section_y, direction);
            }
        }
    }

    /// Validates this chunk's horizontal edges against cached neighbor edges.
    ///
    /// This mirrors `ScalableLux` `checkChunkEdges`: edge values whose calculated
    /// level differs from the stored value are delayed, converted into regular
    /// block checks, then resolved through the decrease queue.
    pub fn check_chunk_edges(&mut self, chunk_pos: ChunkPos) {
        for section_y in (self.layout.range().min_section_y()
            ..self.layout.range().max_section_y_exclusive())
            .rev()
        {
            self.check_chunk_edge(chunk_pos, section_y);
        }

        self.perform_light_decrease();
    }

    /// Calculates the block-light value that should exist at `block_pos` from direct
    /// line-of-sight to emitters only.
    ///
    /// Returns `None` when the position is outside this cache window.
    #[must_use]
    pub fn calculate_light_value(&self, block_pos: BlockPos, _expect: u8) -> Option<u8> {
        self.direct_block_light_level(block_pos)
    }

    /// Block-light level from self-emission and direct rays to emitting blocks only.
    #[must_use]
    fn direct_block_light_level(&self, block_pos: BlockPos) -> Option<u8> {
        self.direct_block_light_from(block_pos).map(|(level, _)| level)
    }

    /// Returns the strongest direct block-light level at `block_pos` and its emitter.
    #[must_use]
    fn direct_block_light_from(&self, block_pos: BlockPos) -> Option<(u8, BlockPos)> {
        let cached_block = self.layout.cached_block(block_pos)?;
        let block_state = self.sections.get_block_state(cached_block);
        let emission = block_state.get_light_emission() & MAX_LIGHT_LEVEL;
        let mut best_level = emission;
        let mut best_emitter = block_pos;

        for_each_adjacent_block_offset(|dx, dy, dz| {
            let neighbor_pos = block_pos.offset(dx, dy, dz);
            let Some(neighbor_block) = self.layout.cached_block(neighbor_pos) else {
                return;
            };
            let neighbor_state = self.sections.get_block_state(neighbor_block);
            let neighbor_emission = neighbor_state.get_light_emission() & MAX_LIGHT_LEVEL;
            if neighbor_emission != 0 {
                if let Some(direct_level) = self.propagated_level_along_ray(neighbor_pos, block_pos)
                    && direct_level > best_level
                {
                    best_level = direct_level;
                    best_emitter = neighbor_pos;
                }
                return;
            }

            let neighbor_level = self.light.get(neighbor_block);
            if neighbor_level <= best_level {
                return;
            }
            let Some((neighbor_emitter, continued_level)) =
                self.neighbor_light_continues_to_block(neighbor_pos, block_pos)
            else {
                return;
            };
            if continued_level > best_level {
                best_level = continued_level;
                best_emitter = neighbor_emitter;
            }
        });

        let vector = self.light.get_block_light_vector(cached_block);
        if !vector.is_zero() {
            let emitter_pos = vector.source_position(block_pos);
            if let Some(direct_level) = self.propagated_level_along_ray(emitter_pos, block_pos)
                && direct_level > best_level
            {
                best_level = direct_level;
                best_emitter = emitter_pos;
            }
        }

        Some((best_level, best_emitter))
    }

    /// Clears blocks that should not be lit, then runs the increase pass.
    ///
    /// Each decrease entry walks one downstream ray step and clears stored light when
    /// it exceeds the direct level from the tracked emitter. Neighbors off the ray are
    /// checked the same way.
    pub fn perform_light_decrease(&mut self) {
        while let Some(entry) = self.queues.dequeue_decrease() {
            let Some(source_block) = self.cached_block_from_entry(entry) else {
                continue;
            };
            let source_pos = source_block.block_pos;
            let emitter_pos = self.emitter_position(source_block);

            if let Some(downstream_pos) = next_voxel_past(source_pos, emitter_pos) {
                self.try_clear_unlit_downstream(
                    source_block,
                    downstream_pos,
                    emitter_pos,
                    entry,
                );
            }

            for axis_direction in LightAxisDirection::ALL {
                self.try_clear_unlit_neighbor(source_block, emitter_pos, axis_direction);
            }
            for_each_adjacent_block_offset(|dx, dy, dz| {
                if !is_diagonal_block_offset(dx, dy, dz) {
                    return;
                }
                let neighbor_pos = source_pos.offset(dx, dy, dz);
                self.try_clear_unlit_adjacent(source_block, emitter_pos, neighbor_pos);
            });
        }

        self.perform_light_increase();
    }

    /// Applies direct block light to blocks that should be lit.
    ///
    /// Each increase entry spreads from emitters along cardinal rays and steps the DDA
    /// line toward the stored emitter for all other lit cells.
    pub fn perform_light_increase(&mut self) {
        while let Some(entry) = self.queues.dequeue_increase() {
            let Some(source_block) = self.cached_block_from_entry(entry) else {
                continue;
            };
            let source_pos = source_block.block_pos;

            if entry.should_recheck_level() {
                if self.light.get(source_block) != entry.level() {
                    continue;
                }
                self.try_apply_lit_at(source_pos, LightQueueFlags::EMPTY);
            } else if entry.should_write_level() {
                self.light.set(source_block, entry.level());
                if entry.level() != 0 {
                    let emission = self
                        .sections
                        .get_block_state(source_block)
                        .get_light_emission()
                        & MAX_LIGHT_LEVEL;
                    if emission != 0 {
                        self.light.set_block_light_vector(source_block, BlockLightVector::ZERO);
                    } else if let Some((_, emitter_pos)) =
                        self.direct_block_light_from(source_pos)
                        && emitter_pos != source_pos
                    {
                        self.light.set_block_light_vector(
                            source_block,
                            BlockLightVector::from_positions(source_pos, emitter_pos),
                        );
                    }
                }
            } else {
                let base_flags = if entry.has_sided_transparent_blocks() {
                    LightQueueFlags::EMPTY.with(LightQueueFlags::HAS_SIDED_TRANSPARENT_BLOCKS)
                } else {
                    LightQueueFlags::EMPTY
                };
                self.try_apply_lit_at(source_pos, base_flags);
            }

            let emitter_pos = self.emitter_position(source_block);

            if source_pos == emitter_pos {
                self.try_apply_lit_emitter_fan(source_block, emitter_pos, entry);
            }

            if entry.level() > 1 {
                if let Some(downstream_pos) = next_voxel_past(source_pos, emitter_pos) {
                    self.try_apply_lit_ray(
                        source_block,
                        downstream_pos,
                        emitter_pos,
                        entry,
                    );
                }
            }

            for_each_adjacent_block_offset(|dx, dy, dz| {
                self.try_enqueue_increase_from_brighter_adjacent(source_pos, dx, dy, dz);
            });
        }
    }

    fn emitter_position(&self, cached_block: CachedLightBlock) -> BlockPos {
        self.light
            .get_block_light_vector(cached_block)
            .source_position(cached_block.block_pos)
    }

    fn try_clear_unlit_downstream(
        &mut self,
        source_block: CachedLightBlock,
        downstream_pos: BlockPos,
        emitter_pos: BlockPos,
        entry: PackedLightQueueEntry,
    ) {
        let Some(downstream_block) = self.layout.cached_block(downstream_pos) else {
            return;
        };
        if !self.light.has_non_missing(downstream_block) {
            return;
        }

        let downstream_vector = self.light.get_block_light_vector(downstream_block);
        if downstream_vector.source_position(downstream_pos) != emitter_pos {
            return;
        }
        if !ray_passes_through(downstream_pos, emitter_pos, source_block.block_pos) {
            return;
        }

        let current_level = self.light.get(downstream_block);
        if current_level == 0 {
            return;
        }

        let direct_level = self
            .propagated_level_along_ray(emitter_pos, downstream_pos)
            .unwrap_or(0);
        if current_level <= direct_level {
            return;
        }

        let flags = if entry.has_sided_transparent_blocks() {
            LightQueueFlags::EMPTY.with(LightQueueFlags::HAS_SIDED_TRANSPARENT_BLOCKS)
        } else {
            LightQueueFlags::EMPTY
        };

        self.light.set(downstream_block, 0);
        self.light
            .set_block_light_vector(downstream_block, BlockLightVector::ZERO);
        self.enqueue_decrease(
            downstream_pos,
            current_level,
            LightDirectionSet::all(),
            flags,
        );
    }

    fn try_clear_unlit_neighbor(
        &mut self,
        source_block: CachedLightBlock,
        emitter_pos: BlockPos,
        axis_direction: LightAxisDirection,
    ) {
        let neighbor_pos = Self::offset(source_block.block_pos, axis_direction);
        if Some(neighbor_pos) == next_voxel_past(source_block.block_pos, emitter_pos) {
            return;
        }

        let Some(neighbor_block) = self.layout.cached_block(neighbor_pos) else {
            return;
        };
        if !self.light.has_non_missing(neighbor_block) {
            return;
        }

        let current_level = self.light.get(neighbor_block);
        if current_level == 0 {
            return;
        }

        let Some((direct_level, _)) = self.direct_block_light_from(neighbor_pos) else {
            return;
        };
        if current_level <= direct_level {
            return;
        }

        self.light.set(neighbor_block, 0);
        self.light
            .set_block_light_vector(neighbor_block, BlockLightVector::ZERO);
        self.enqueue_decrease(
            neighbor_pos,
            current_level,
            LightDirectionSet::all(),
            LightQueueFlags::EMPTY,
        );
    }

    fn try_clear_unlit_adjacent(
        &mut self,
        source_block: CachedLightBlock,
        emitter_pos: BlockPos,
        neighbor_pos: BlockPos,
    ) {
        if Some(neighbor_pos) == next_voxel_past(source_block.block_pos, emitter_pos) {
            return;
        }

        let Some(neighbor_block) = self.layout.cached_block(neighbor_pos) else {
            return;
        };
        if !self.light.has_non_missing(neighbor_block) {
            return;
        }

        let current_level = self.light.get(neighbor_block);
        if current_level == 0 {
            return;
        }

        let Some((direct_level, _)) = self.direct_block_light_from(neighbor_pos) else {
            return;
        };
        if current_level <= direct_level {
            return;
        }

        self.light.set(neighbor_block, 0);
        self.light
            .set_block_light_vector(neighbor_block, BlockLightVector::ZERO);
        self.enqueue_decrease(
            neighbor_pos,
            current_level,
            LightDirectionSet::all(),
            LightQueueFlags::EMPTY,
        );
    }

    fn try_enqueue_increase_from_brighter_adjacent(
        &mut self,
        block_pos: BlockPos,
        dx: i32,
        dy: i32,
        dz: i32,
    ) {
        let neighbor_pos = block_pos.offset(dx, dy, dz);
        let Some(neighbor_block) = self.layout.cached_block(neighbor_pos) else {
            return;
        };
        if !self.light.has_non_missing(neighbor_block) {
            return;
        }

        let Some(cached_block) = self.layout.cached_block(block_pos) else {
            return;
        };
        if !self.light.has_non_missing(cached_block) {
            return;
        }

        let neighbor_level = self.light.get(neighbor_block);
        let block_level = self.light.get(cached_block);
        if neighbor_level <= block_level {
            return;
        }

        let Some((_, continued_level)) =
            self.neighbor_light_continues_to_block(neighbor_pos, block_pos)
        else {
            return;
        };
        if continued_level <= block_level {
            return;
        }

        let block_state = self.sections.get_block_state(cached_block);
        let mut flags = LightQueueFlags::EMPTY;
        if !light_occlusion_shape(block_state).is_empty() {
            flags = flags.with(LightQueueFlags::HAS_SIDED_TRANSPARENT_BLOCKS);
        }

        self.enqueue_increase(
            block_pos,
            continued_level,
            LightDirectionSet::all(),
            flags,
        );
    }

    /// Returns the emitter and one-step continued level when `neighbor_pos` is lit and
    /// `block_pos` is the adjacent downstream cell on the same light line.
    fn neighbor_light_continues_to_block(
        &self,
        neighbor_pos: BlockPos,
        block_pos: BlockPos,
    ) -> Option<(BlockPos, u8)> {
        let neighbor_block = self.layout.cached_block(neighbor_pos)?;
        let neighbor_state = self.sections.get_block_state(neighbor_block);
        let neighbor_level = self.light.get(neighbor_block);
        if neighbor_level == 0 {
            return None;
        }

        let neighbor_emission = neighbor_state.get_light_emission() & MAX_LIGHT_LEVEL;
        let neighbor_vector = self.light.get_block_light_vector(neighbor_block);
        if neighbor_emission == 0 && neighbor_vector.is_zero() {
            return None;
        }

        let neighbor_emitter = self.emitter_position(neighbor_block);
        if !Self::is_adjacent_light_continuation(neighbor_pos, block_pos, neighbor_emitter) {
            return None;
        }

        let block_state = self.sections.get_block_state(
            self.layout.cached_block(block_pos)?,
        );
        let direction_from_neighbor = dominant_direction(neighbor_pos, block_pos);
        let opacity = get_light_block_into(
            neighbor_state,
            block_state,
            direction_from_neighbor,
            get_light_opacity(block_state),
        );
        if opacity == LIGHT_BLOCKED || opacity >= neighbor_level {
            return None;
        }

        Some((neighbor_emitter, neighbor_level - opacity))
    }

    /// Returns true when `block_pos` is the adjacent next cell on the light line through
    /// `neighbor_pos` from `neighbor_emitter`.
    fn is_adjacent_light_continuation(
        neighbor_pos: BlockPos,
        block_pos: BlockPos,
        neighbor_emitter: BlockPos,
    ) -> bool {
        if neighbor_emitter == neighbor_pos {
            return next_voxel_toward(neighbor_pos, block_pos) == Some(block_pos)
                || Self::is_adjacent_block_pos(neighbor_pos, block_pos);
        }

        next_voxel_past(neighbor_pos, neighbor_emitter) == Some(block_pos)
    }

    const fn is_adjacent_block_pos(from: BlockPos, to: BlockPos) -> bool {
        let dx = to.x() - from.x();
        let dy = to.y() - from.y();
        let dz = to.z() - from.z();
        dx.abs() <= 1 && dy.abs() <= 1 && dz.abs() <= 1 && (dx != 0 || dy != 0 || dz != 0)
    }

    fn try_apply_lit_at(&mut self, block_pos: BlockPos, base_flags: LightQueueFlags) {
        let Some((direct_level, emitter_pos)) = self.direct_block_light_from(block_pos) else {
            return;
        };
        let Some(cached_block) = self.layout.cached_block(block_pos) else {
            return;
        };
        let current_level = self.light.get(cached_block);
        if direct_level <= current_level {
            return;
        }

        let block_state = self.sections.get_block_state(cached_block);
        let mut flags = base_flags;
        if !light_occlusion_shape(block_state).is_empty() {
            flags = flags.with(LightQueueFlags::HAS_SIDED_TRANSPARENT_BLOCKS);
        }

        self.light.set(cached_block, direct_level);
        if direct_level != 0 && emitter_pos != block_pos {
            self.light.set_block_light_vector(
                cached_block,
                BlockLightVector::from_positions(block_pos, emitter_pos),
            );
        }
        if direct_level > 1 {
            self.enqueue_increase(
                block_pos,
                direct_level,
                LightDirectionSet::all(),
                flags,
            );
        }
    }

    fn emitter_has_open_cardinal_neighbors(&self, emitter_pos: BlockPos) -> bool {
        let Some(emitter_cached) = self.layout.cached_block(emitter_pos) else {
            return false;
        };
        let emitter_state = self.sections.get_block_state(emitter_cached);

        for axis_direction in LightAxisDirection::ALL {
            let neighbor_pos = Self::offset(emitter_pos, axis_direction);
            let Some(neighbor_cached) = self.layout.cached_block(neighbor_pos) else {
                return false;
            };
            let neighbor_state = self.sections.get_block_state(neighbor_cached);
            let opacity = get_light_block_into(
                emitter_state,
                neighbor_state,
                axis_direction.direction(),
                get_light_opacity(neighbor_state),
            );
            if opacity == LIGHT_BLOCKED {
                return false;
            }
        }

        true
    }

    fn try_apply_lit_emitter_fan(
        &mut self,
        source_block: CachedLightBlock,
        emitter_pos: BlockPos,
        entry: PackedLightQueueEntry,
    ) {
        let fan_diagonal = self.emitter_has_open_cardinal_neighbors(emitter_pos);
        let base_flags = if entry.has_sided_transparent_blocks() {
            LightQueueFlags::EMPTY.with(LightQueueFlags::HAS_SIDED_TRANSPARENT_BLOCKS)
        } else {
            LightQueueFlags::EMPTY
        };

        for_each_adjacent_block_offset(|dx, dy, dz| {
            if is_diagonal_block_offset(dx, dy, dz) && !fan_diagonal {
                return;
            }

            let neighbor_pos = source_block.block_pos.offset(dx, dy, dz);
            self.try_apply_lit_adjacent(
                source_block,
                emitter_pos,
                neighbor_pos,
                entry,
                base_flags,
            );
        });
    }

    fn try_apply_lit_adjacent(
        &mut self,
        _source_block: CachedLightBlock,
        emitter_pos: BlockPos,
        neighbor_pos: BlockPos,
        _entry: PackedLightQueueEntry,
        base_flags: LightQueueFlags,
    ) {
        let Some(neighbor_block) = self.layout.cached_block(neighbor_pos) else {
            return;
        };
        if !self.light.has_non_missing(neighbor_block) {
            return;
        }
        let Some(direct_level) = self.propagated_level_along_ray(emitter_pos, neighbor_pos) else {
            return;
        };
        if direct_level == 0 {
            return;
        }

        let current_level = self.light.get(neighbor_block);
        if direct_level <= current_level {
            return;
        }

        let neighbor_state = self.sections.get_block_state(neighbor_block);
        let mut flags = base_flags;
        if !light_occlusion_shape(neighbor_state).is_empty() {
            flags = flags.with(LightQueueFlags::HAS_SIDED_TRANSPARENT_BLOCKS);
        }

        self.light.set(neighbor_block, direct_level);
        self.light.set_block_light_vector(
            neighbor_block,
            BlockLightVector::from_positions(neighbor_block.block_pos, emitter_pos),
        );
        if direct_level > 1 {
            self.enqueue_increase(
                neighbor_pos,
                direct_level,
                LightDirectionSet::all(),
                flags,
            );
        }
    }

    fn try_apply_lit_ray(
        &mut self,
        source_block: CachedLightBlock,
        downstream_pos: BlockPos,
        emitter_pos: BlockPos,
        entry: PackedLightQueueEntry,
    ) {
        if !ray_passes_through(downstream_pos, emitter_pos, source_block.block_pos) {
            return;
        }

        let Some(downstream_block) = self.layout.cached_block(downstream_pos) else {
            return;
        };
        if !self.light.has_non_missing(downstream_block) {
            return;
        }
        let Some(direct_level) = self.propagated_level_along_ray(emitter_pos, downstream_pos) else {
            return;
        };
        if direct_level == 0 {
            return;
        }

        let current_level = self.light.get(downstream_block);
        if direct_level <= current_level {
            return;
        }

        let flags = if entry.has_sided_transparent_blocks() {
            LightQueueFlags::EMPTY.with(LightQueueFlags::HAS_SIDED_TRANSPARENT_BLOCKS)
        } else {
            LightQueueFlags::EMPTY
        };

        self.light.set(downstream_block, direct_level);
        self.light.set_block_light_vector(
            downstream_block,
            BlockLightVector::from_positions(downstream_pos, emitter_pos),
        );
        if direct_level > 1 {
            self.enqueue_increase(
                downstream_pos,
                direct_level,
                LightDirectionSet::all(),
                flags,
            );
        }
    }

    fn propagated_level_along_ray(
        &self,
        emitter_pos: BlockPos,
        target_pos: BlockPos,
    ) -> Option<u8> {
        let Some(emitter_cached) = self.layout.cached_block(emitter_pos) else {
            return None;
        };
        let emitter_state = self.sections.get_block_state(emitter_cached);
        let emission = emitter_state.get_light_emission() & MAX_LIGHT_LEVEL;
        let mut level = if emission != 0 {
            emission
        } else {
            self.light.get(emitter_cached)
        };

        if emitter_pos == target_pos {
            return Some(level);
        }

        let dx = target_pos.x() - emitter_pos.x();
        let dy = target_pos.y() - emitter_pos.y();
        let dz = target_pos.z() - emitter_pos.z();
        if dx.abs() <= 1 && dy.abs() <= 1 && dz.abs() <= 1 {
            let Some(target_cached) = self.layout.cached_block(target_pos) else {
                return None;
            };
            let target_state = self.sections.get_block_state(target_cached);
            let opacity = get_light_block_into(
                emitter_state,
                target_state,
                dominant_direction(emitter_pos, target_pos),
                get_light_opacity(target_state),
            );
            if opacity == LIGHT_BLOCKED {
                return Some(0);
            }
            return Some(level.saturating_sub(opacity));
        }

        let mut from = emitter_pos;
        for step in line_voxels_between(emitter_pos, target_pos) {
            if step == emitter_pos {
                continue;
            }

            let Some(step_cached) = self.layout.cached_block(step) else {
                return None;
            };
            let Some(from_cached) = self.layout.cached_block(from) else {
                return None;
            };
            let from_state = self.sections.get_block_state(from_cached);
            let step_state = self.sections.get_block_state(step_cached);
            let direction = dominant_direction(from, step);
            let opacity = get_light_block_into(
                from_state,
                step_state,
                direction,
                get_light_opacity(step_state),
            );
            if opacity == LIGHT_BLOCKED {
                return Some(0);
            }
            level = level.saturating_sub(opacity);
            from = step;
            if level == 0 {
                return Some(0);
            }
        }

        Some(level)
    }

    fn cached_block_from_entry(&self, entry: PackedLightQueueEntry) -> Option<CachedLightBlock> {
        self.layout.cached_block_from_packed(entry.block_pos())
    }

    fn enqueue_decrease(
        &mut self,
        block_pos: BlockPos,
        level: u8,
        directions: LightDirectionSet,
        flags: LightQueueFlags,
    ) {
        let Some(packed_pos) = self.layout.encode_block_pos(block_pos) else {
            return;
        };
        self.queues
            .enqueue_decrease(PackedLightQueueEntry::from_parts(
                packed_pos, level, directions, flags,
            ));
    }

    fn check_chunk_edge(&mut self, chunk_pos: ChunkPos, section_y: i32) {
        let current_section_pos = SectionPos::new(chunk_pos.0.x, section_y, chunk_pos.0.y);
        if !self.light.has_cached_section(current_section_pos) {
            return;
        }

        for direction in LightAxisDirection::HORIZONTAL {
            let (neighbor_offset_x, _, neighbor_offset_z) = direction.offset();
            let neighbor_chunk_pos = ChunkPos::new(
                chunk_pos.0.x + neighbor_offset_x,
                chunk_pos.0.y + neighbor_offset_z,
            );
            let neighbor_section_pos =
                SectionPos::new(neighbor_chunk_pos.0.x, section_y, neighbor_chunk_pos.0.y);
            if !self.light.has_cached_section(neighbor_section_pos) {
                continue;
            }
            if !self.light.has_light_data_section(current_section_pos)
                && !self.light.has_light_data_section(neighbor_section_pos)
            {
                continue;
            }

            self.check_chunk_edge_direction(chunk_pos, neighbor_chunk_pos, section_y, direction);
        }
    }

    fn check_chunk_edge_direction(
        &mut self,
        chunk_pos: ChunkPos,
        neighbor_chunk_pos: ChunkPos,
        section_y: i32,
        direction: LightAxisDirection,
    ) {
        let (neighbor_offset_x, _, neighbor_offset_z) = direction.offset();
        let (increment_x, increment_z, start_x, start_z) =
            Self::current_edge_scan(chunk_pos, direction);
        let mut center_delayed_checks = [0usize; 16 * 16];
        let mut neighbor_delayed_checks = [0usize; 16 * 16];
        let mut center_delayed_check_count = 0;
        let mut neighbor_delayed_check_count = 0;

        let min_y = section_y << 4;
        let max_y = min_y | 15;
        for y in min_y..=max_y {
            let mut x = start_x;
            let mut z = start_z;
            for _ in 0..16 {
                let current_pos = BlockPos::new(x, y, z);
                let neighbor_pos = BlockPos::new(x + neighbor_offset_x, y, z + neighbor_offset_z);
                let Some(current_block) = self.layout.cached_block(current_pos) else {
                    x += increment_x;
                    z += increment_z;
                    continue;
                };
                let Some(neighbor_block) = self.layout.cached_block(neighbor_pos) else {
                    x += increment_x;
                    z += increment_z;
                    continue;
                };

                let current_level = self.light.get(current_block);
                if self
                    .direct_block_light_level(current_pos)
                    .is_some_and(|calculated| calculated != current_level)
                {
                    center_delayed_checks[center_delayed_check_count] = current_block.local_index;
                    center_delayed_check_count += 1;
                }

                let neighbor_level = self.light.get(neighbor_block);
                if self
                    .direct_block_light_level(neighbor_pos)
                    .is_some_and(|calculated| calculated != neighbor_level)
                {
                    neighbor_delayed_checks[neighbor_delayed_check_count] =
                        neighbor_block.local_index;
                    neighbor_delayed_check_count += 1;
                }

                x += increment_x;
                z += increment_z;
            }
        }

        let current_chunk_offset_x = chunk_pos.0.x << 4;
        let current_chunk_offset_z = chunk_pos.0.y << 4;
        let neighbor_chunk_offset_x = neighbor_chunk_pos.0.x << 4;
        let neighbor_chunk_offset_z = neighbor_chunk_pos.0.y << 4;
        let chunk_offset_y = section_y << 4;
        let delayed_check_count = center_delayed_check_count.max(neighbor_delayed_check_count);
        for delayed_check_index in 0..delayed_check_count {
            if delayed_check_index < center_delayed_check_count {
                let local_index = center_delayed_checks[delayed_check_index];
                self.check_block(Self::block_pos_from_local_index(
                    current_chunk_offset_x,
                    chunk_offset_y,
                    current_chunk_offset_z,
                    local_index,
                ));
            }
            if delayed_check_index < neighbor_delayed_check_count {
                let local_index = neighbor_delayed_checks[delayed_check_index];
                self.check_block(Self::block_pos_from_local_index(
                    neighbor_chunk_offset_x,
                    chunk_offset_y,
                    neighbor_chunk_offset_z,
                    local_index,
                ));
            }
        }
    }

    fn seed_block_light_source(&mut self, block_pos: BlockPos) -> bool {
        let Some(cached_block) = self.layout.cached_block(block_pos) else {
            return false;
        };

        let block_state = self.sections.get_block_state(cached_block);
        let emitted_level = block_state.get_light_emission() & MAX_LIGHT_LEVEL;
        if emitted_level <= self.light.get(cached_block) {
            return false;
        }

        self.enqueue_increase(
            block_pos,
            emitted_level,
            LightDirectionSet::all(),
            Self::shape_flags(block_state),
        );
        self.light.set(cached_block, emitted_level);
        self.light.set_block_light_vector(cached_block, BlockLightVector::ZERO);
        true
    }

    fn propagate_neighbor_level_section(
        &mut self,
        chunk_pos: ChunkPos,
        section_y: i32,
        direction: LightAxisDirection,
    ) {
        let (neighbor_offset_x, _, neighbor_offset_z) = direction.offset();
        let neighbor_section_pos = SectionPos::new(
            chunk_pos.0.x + neighbor_offset_x,
            section_y,
            chunk_pos.0.y + neighbor_offset_z,
        );
        if !self.light.has_light_data_section(neighbor_section_pos) {
            return;
        }

        let (increment_x, increment_z, start_x, start_z) =
            Self::neighbor_edge_scan(chunk_pos, direction);
        let directions = LightDirectionSet::only(direction.opposite());
        let flags = LightQueueFlags::EMPTY.with(LightQueueFlags::HAS_SIDED_TRANSPARENT_BLOCKS);

        let min_y = section_y << 4;
        let max_y = min_y | 15;
        for y in min_y..=max_y {
            let mut x = start_x;
            let mut z = start_z;
            for _ in 0..16 {
                let source_pos = BlockPos::new(x, y, z);
                let Some(source_block) = self.layout.cached_block(source_pos) else {
                    x += increment_x;
                    z += increment_z;
                    continue;
                };
                let level = self.light.get(source_block);
                if level > 1 {
                    self.enqueue_increase(source_pos, level, directions, flags);
                }
                x += increment_x;
                z += increment_z;
            }
        }
    }

    const fn current_edge_scan(
        chunk_pos: ChunkPos,
        direction: LightAxisDirection,
    ) -> (i32, i32, i32, i32) {
        let (offset_x, _, offset_z) = direction.offset();
        if offset_x != 0 {
            let start_x = if offset_x < 0 {
                chunk_pos.0.x << 4
            } else {
                (chunk_pos.0.x << 4) | 15
            };
            return (0, 1, start_x, chunk_pos.0.y << 4);
        }

        let start_z = if offset_z < 0 {
            chunk_pos.0.y << 4
        } else {
            (chunk_pos.0.y << 4) | 15
        };
        (1, 0, chunk_pos.0.x << 4, start_z)
    }

    const fn neighbor_edge_scan(
        chunk_pos: ChunkPos,
        direction: LightAxisDirection,
    ) -> (i32, i32, i32, i32) {
        let (offset_x, _, offset_z) = direction.offset();
        if offset_x != 0 {
            let start_x = if offset_x < 0 {
                (chunk_pos.0.x << 4) - 1
            } else {
                (chunk_pos.0.x << 4) + 16
            };
            return (0, 1, start_x, chunk_pos.0.y << 4);
        }

        let start_z = if offset_z < 0 {
            (chunk_pos.0.y << 4) - 1
        } else {
            (chunk_pos.0.y << 4) + 16
        };
        (1, 0, chunk_pos.0.x << 4, start_z)
    }

    const fn block_pos_from_local_index(
        chunk_offset_x: i32,
        chunk_offset_y: i32,
        chunk_offset_z: i32,
        local_index: usize,
    ) -> BlockPos {
        BlockPos::new(
            chunk_offset_x | (local_index & 15) as i32,
            chunk_offset_y | (local_index >> 8) as i32,
            chunk_offset_z | ((local_index >> 4) & 15) as i32,
        )
    }

    fn enqueue_increase(
        &mut self,
        block_pos: BlockPos,
        level: u8,
        directions: LightDirectionSet,
        flags: LightQueueFlags,
    ) {
        let Some(packed_pos) = self.layout.encode_block_pos(block_pos) else {
            return;
        };
        self.queues
            .enqueue_increase(PackedLightQueueEntry::from_parts(
                packed_pos, level, directions, flags,
            ));
    }

    fn shape_flags(block_state: BlockStateId) -> LightQueueFlags {
        if light_occlusion_shape(block_state).is_empty() {
            LightQueueFlags::EMPTY
        } else {
            LightQueueFlags::EMPTY.with(LightQueueFlags::HAS_SIDED_TRANSPARENT_BLOCKS)
        }
    }

    const fn offset(block_pos: BlockPos, direction: LightAxisDirection) -> BlockPos {
        let (dx, dy, dz) = direction.offset();
        block_pos.offset(dx, dy, dz)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Weak};

    use steel_registry::{
        blocks::properties::{BlockStateProperties, SlabType},
        test_support::init_test_registry,
        vanilla_blocks,
    };
    use steel_utils::{ChunkPos, types::UpdateFlags};

    use super::*;
    use crate::behavior::init_behaviors;
    use crate::chunk::{
        chunk_access::{ChunkAccess, ChunkStatus},
        chunk_holder::ChunkHolder,
        chunk_ticket_manager::ChunkTicketLevel,
        light::{LightCacheSetupRadius, LightSection, LightSectionData, LightSectionRange},
        proto_chunk::ProtoChunk,
        section::{ChunkSection, Sections},
    };

    fn init_tests() {
        init_test_registry();
        init_behaviors();
    }

    fn range() -> LightSectionRange {
        let Ok(range) = LightSectionRange::from_world_height(0, 16) else {
            panic!("test height should create a valid light range");
        };
        range
    }

    fn holder_with_section(pos: ChunkPos, section: ChunkSection) -> Arc<ChunkHolder> {
        let sections = Sections::from_owned(vec![section].into_boxed_slice());
        let proto = ProtoChunk::new(sections, pos, 0, 16, Weak::new());
        let holder = Arc::new(ChunkHolder::new(
            pos,
            ChunkTicketLevel::FULL_CHUNK,
            Some(ChunkTicketLevel::FULL_CHUNK),
            0,
            16,
        ));
        holder.insert_chunk(ChunkAccess::Proto(proto), ChunkStatus::Light);
        holder
    }

    fn initialize_holder_light(holder: &ChunkHolder) {
        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        chunk.initialize_light_sources();
    }

    fn set_block_section_non_missing(holder: &ChunkHolder, section_y: i32) {
        set_block_light_section(
            holder,
            section_y,
            LightSection::visible(LightSectionData::homogeneous(0)),
        );
    }

    fn set_visible_block_light(
        holder: &ChunkHolder,
        section_y: i32,
        x: usize,
        y: usize,
        z: usize,
        level: u8,
    ) {
        let mut data = LightSectionData::homogeneous(0);
        data.set(x, y, z, level);
        set_block_light_section(holder, section_y, LightSection::visible(data));
    }

    fn set_block_light_section(holder: &ChunkHolder, section_y: i32, section: LightSection) {
        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let mut light = chunk.light_mut();
        let Some(target) = light.block.section_mut(section_y) else {
            panic!("test section should be inside light range");
        };
        *target = section;
    }

    fn block_light_at(holder: &ChunkHolder, pos: BlockPos) -> u8 {
        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        chunk.light().get_light_value(LightLayer::Block, pos)
    }

    fn block_light_vector_at(holder: &ChunkHolder, pos: BlockPos) -> BlockLightVector {
        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        chunk.light().get_block_light_vector(pos)
    }

    #[test]
    fn context_requires_block_layer() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let holder = holder_with_section(center, ChunkSection::new_empty());
        set_block_section_non_missing(&holder, 0);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        workset.with_chunk_read_cache(|chunk_cache| {
            chunk_cache.with_section_read_cache(|section_cache| {
                chunk_cache.with_light_edit(LightLayer::Sky, |mut light_edit| {
                    let mut queues = PackedLightPropagationQueues::new();
                    let result = BlockLightPropagationContext::new(
                        section_cache,
                        &mut light_edit,
                        &mut queues,
                    );

                    assert_eq!(
                        result.err(),
                        Some(BlockLightPropagationContextError::WrongLayer {
                            layer: LightLayer::Sky,
                        })
                    );
                });
            });
        });
    }

    #[test]
    fn block_light_runner_publishes_visible_updates() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let source_pos = BlockPos::new(1, 1, 1);
        let mut section = ChunkSection::new_empty();
        section.set_block_state(1, 1, 1, vanilla_blocks::LIGHT.default_state());
        let holder = holder_with_section(center, section);
        set_block_section_non_missing(&holder, 0);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        let Ok(result) = propagate_block_light_changes(&workset, [source_pos]) else {
            panic!("matching block caches should run block light updates");
        };

        assert!(result.updated_sections.contains(&SectionPos::new(0, 0, 0)));
        assert_eq!(block_light_at(&holder, source_pos), 15);
        assert_eq!(block_light_at(&holder, BlockPos::new(2, 1, 1)), 14);
    }

    #[test]
    fn block_light_emitter_fans_diagonal_in_open_air() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let source_pos = BlockPos::new(1, 1, 1);
        let diagonal_pos = BlockPos::new(2, 2, 1);
        let mut section = ChunkSection::new_empty();
        section.set_block_state(1, 1, 1, vanilla_blocks::LIGHT.default_state());
        let holder = holder_with_section(center, section);
        set_block_section_non_missing(&holder, 0);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        let Ok(()) = propagate_block_light_changes(&workset, [source_pos]).map(|_| ()) else {
            panic!("matching block caches should run block light updates");
        };

        assert_eq!(block_light_at(&holder, diagonal_pos), 14);
    }

    #[test]
    fn block_light_removal_clears_downstream_air() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let source_pos = BlockPos::new(1, 1, 1);
        let air_pos = BlockPos::new(2, 1, 1);
        let mut section = ChunkSection::new_empty();
        section.set_block_state(1, 1, 1, vanilla_blocks::LIGHT.default_state());
        let holder = holder_with_section(center, section);
        set_block_section_non_missing(&holder, 0);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        let Ok(()) = propagate_block_light_changes(&workset, [source_pos]).map(|_| ()) else {
            panic!("matching block caches should run block light updates");
        };
        assert_eq!(block_light_at(&holder, air_pos), 14);

        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("chunk should be available");
        };
        assert_eq!(
            chunk.set_block_state(
                source_pos,
                vanilla_blocks::AIR.default_state(),
                UpdateFlags::UPDATE_NONE,
            ),
            Some(vanilla_blocks::LIGHT.default_state())
        );
        drop(chunk);

        let Ok(()) = propagate_block_light_changes(&workset, [source_pos]).map(|_| ()) else {
            panic!("matching block caches should run block light updates");
        };

        assert_eq!(block_light_at(&holder, source_pos), 0);
        assert_eq!(block_light_at(&holder, air_pos), 0);
        assert_eq!(block_light_vector_at(&holder, air_pos), BlockLightVector::ZERO);
    }

    #[test]
    fn block_light_propagation_records_source_vectors_in_air() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let source_pos = BlockPos::new(1, 1, 1);
        let air_pos = BlockPos::new(2, 1, 1);
        let mut section = ChunkSection::new_empty();
        section.set_block_state(1, 1, 1, vanilla_blocks::LIGHT.default_state());
        let holder = holder_with_section(center, section);
        set_block_section_non_missing(&holder, 0);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        let Ok(()) = propagate_block_light_changes(&workset, [source_pos]).map(|_| ()) else {
            panic!("matching block caches should run block light updates");
        };

        assert_eq!(block_light_at(&holder, air_pos), 14);
        assert_eq!(
            block_light_vector_at(&holder, air_pos),
            BlockLightVector {
                dx: -1,
                dy: 0,
                dz: 0,
            }
        );
        assert_eq!(
            block_light_vector_at(&holder, source_pos),
            BlockLightVector::ZERO
        );
    }

    #[test]
    fn block_light_vectors_chain_through_indirect_air_to_emitter() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let source_pos = BlockPos::new(1, 1, 1);
        let intermediate_pos = BlockPos::new(2, 1, 1);
        let distant_air_pos = BlockPos::new(3, 1, 1);
        let mut section = ChunkSection::new_empty();
        section.set_block_state(1, 1, 1, vanilla_blocks::LIGHT.default_state());
        let holder = holder_with_section(center, section);
        set_block_section_non_missing(&holder, 0);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        let Ok(()) = propagate_block_light_changes(&workset, [source_pos]).map(|_| ()) else {
            panic!("matching block caches should run block light updates");
        };

        assert_eq!(block_light_at(&holder, distant_air_pos), 13);
        assert_eq!(
            block_light_vector_at(&holder, intermediate_pos),
            BlockLightVector {
                dx: -1,
                dy: 0,
                dz: 0,
            }
        );
        let distant_vector = block_light_vector_at(&holder, distant_air_pos);
        assert_eq!(
            distant_vector,
            BlockLightVector {
                dx: -2,
                dy: 0,
                dz: 0,
            }
        );
        assert_eq!(distant_vector.source_position(distant_air_pos), source_pos);
    }

    #[test]
    fn block_light_changes_apply_empty_section_transitions() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let removed_pos = BlockPos::new(1, 1, 1);
        let mut holders = Vec::new();
        let mut center_holder = None;
        for z in -2..=2 {
            for x in -2..=2 {
                let pos = ChunkPos::new(x, z);
                let mut section = ChunkSection::new_empty();
                if pos == center {
                    section.set_block_state(1, 1, 1, vanilla_blocks::STONE.default_state());
                }
                let holder = holder_with_section(pos, section);
                initialize_holder_light(&holder);
                if pos == center {
                    center_holder = Some(Arc::clone(&holder));
                }
                holders.push((pos, holder));
            }
        }
        let Some(center_holder) = center_holder else {
            panic!("center holder should be created");
        };
        set_visible_block_light(&center_holder, 0, 1, 1, 1, 9);

        let Some(chunk) = center_holder.try_chunk(ChunkStatus::Empty) else {
            panic!("center chunk should be available");
        };
        assert_eq!(
            chunk.set_block_state(
                removed_pos,
                vanilla_blocks::AIR.default_state(),
                UpdateFlags::UPDATE_NONE,
            ),
            Some(vanilla_blocks::STONE.default_state())
        );
        drop(chunk);

        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Full,
            true,
            |pos| {
                holders
                    .iter()
                    .find(|(holder_pos, _)| *holder_pos == pos)
                    .map(|(_, holder)| Arc::clone(holder))
            },
            |_| true,
        ) else {
            panic!("relaxed setup should accept cached test chunks");
        };

        let Ok(result) = propagate_block_light_changes_with_empty_sections(
            &workset,
            [removed_pos],
            [LightSectionEmptinessChange {
                section_pos: SectionPos::new(0, 0, 0),
                empty: true,
            }],
        ) else {
            panic!("matching block caches should run block light updates");
        };

        assert!(result.updated_sections.contains(&SectionPos::new(0, 0, 0)));
        let Some(chunk) = center_holder.try_chunk(ChunkStatus::Empty) else {
            panic!("center chunk should be available");
        };
        let light = chunk.light();
        assert_eq!(light.block.section_empty(0), Some(true));
        assert_eq!(light.get_light_value(LightLayer::Block, removed_pos), 0);
        assert!(matches!(
            light.block.section(0),
            Some(LightSection::Missing | LightSection::Internal(_))
        ));
    }

    #[test]
    fn block_light_chunk_seeds_center_sources() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let source_pos = BlockPos::new(1, 1, 1);
        let mut section = ChunkSection::new_empty();
        section.set_block_state(1, 1, 1, vanilla_blocks::LIGHT.default_state());
        let holder = holder_with_section(center, section);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        let Ok(result) = propagate_block_light_chunk(&workset, BlockLightChunkEdgeChecks::Skipped)
        else {
            panic!("matching block caches should run block chunk lighting");
        };

        assert!(result.updated_sections.contains(&SectionPos::new(0, 0, 0)));
        assert_eq!(block_light_at(&holder, source_pos), 15);
        assert_eq!(block_light_at(&holder, BlockPos::new(2, 1, 1)), 14);
    }

    #[test]
    fn block_light_chunk_pulls_neighbor_edge_levels() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let east_chunk = ChunkPos::new(1, 0);
        let center_holder = holder_with_section(center, ChunkSection::new_empty());
        let mut east_section = ChunkSection::new_empty();
        east_section.set_block_state(0, 1, 1, vanilla_blocks::LIGHT.default_state());
        let east_holder = holder_with_section(east_chunk, east_section);
        set_visible_block_light(&east_holder, 0, 0, 1, 1, 15);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| {
                if pos == center {
                    Some(Arc::clone(&center_holder))
                } else if pos == east_chunk {
                    Some(Arc::clone(&east_holder))
                } else {
                    None
                }
            },
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        let Ok(result) = propagate_block_light_chunk(&workset, BlockLightChunkEdgeChecks::Skipped)
        else {
            panic!("matching block caches should run block chunk lighting");
        };

        assert!(result.updated_sections.contains(&SectionPos::new(0, 0, 0)));
        assert_eq!(block_light_at(&center_holder, BlockPos::new(15, 1, 1)), 14);
        assert_eq!(block_light_at(&center_holder, BlockPos::new(14, 1, 1)), 13);
    }

    #[test]
    fn block_light_chunk_requires_center_chunk() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |_| None,
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing chunks");
        };

        assert_eq!(
            propagate_block_light_chunk(&workset, BlockLightChunkEdgeChecks::Skipped).err(),
            Some(BlockLightPropagationContextError::MissingCenterChunk { chunk_pos: center })
        );
    }

    #[test]
    fn block_light_changes_skip_missing_center_chunk() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Full,
            true,
            |_| None,
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing chunks");
        };

        let Ok(result) = propagate_block_light_changes_with_empty_sections(
            &workset,
            [BlockPos::new(1, 1, 1)],
            [LightSectionEmptinessChange {
                section_pos: SectionPos::new(0, 0, 0),
                empty: true,
            }],
        ) else {
            panic!("dynamic block changes should skip a missing center chunk");
        };

        assert!(result.updated_sections.is_empty());
    }

    #[test]
    fn block_light_calculation_respects_occluding_faces() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let mut section = ChunkSection::new_empty();
        let bottom_slab = vanilla_blocks::STONE_SLAB
            .default_state()
            .set_value(&BlockStateProperties::SLAB_TYPE, SlabType::Bottom);
        section.set_block_state(1, 1, 1, bottom_slab);
        let holder = holder_with_section(center, section);
        set_block_section_non_missing(&holder, 0);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        workset.with_chunk_read_cache(|chunk_cache| {
            chunk_cache.with_section_read_cache(|section_cache| {
                chunk_cache.with_light_edit(LightLayer::Block, |mut light_edit| {
                    let mut queues = PackedLightPropagationQueues::new();
                    let Ok(context) = BlockLightPropagationContext::new(
                        section_cache,
                        &mut light_edit,
                        &mut queues,
                    ) else {
                        panic!("matching block caches should build a propagation context");
                    };
                    let Some(below) = layout.cached_block(BlockPos::new(1, 0, 1)) else {
                        panic!("below neighbor should be cached");
                    };
                    assert!(context.light.set(below, 15));

                    assert_eq!(
                        context.calculate_light_value(BlockPos::new(1, 1, 1), 0),
                        Some(0)
                    );
                });
            });
        });
    }
}
