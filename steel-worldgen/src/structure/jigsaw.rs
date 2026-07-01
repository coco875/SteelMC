//! Jigsaw assembly. Ports vanilla's `JigsawPlacement` BFS: connects pieces via
//! jigsaw blocks given a start pool + config. Produces typed piece state;
//! block placement runs in a later worldgen stage.

use std::cmp::Reverse;

use glam::IVec3;
use rustc_hash::FxHashMap;
use steel_registry::structure::{
    JigsawConfig, LiquidSettingsData, PoolAlias, StartHeight, StructureData,
};
use steel_registry::template_pool::{
    JigsawBlock, JigsawOrientation, JointType, PoolElement, Projection, TemplateData,
    TemplatePoolData,
};
use steel_utils::random::legacy_random::LegacyRandom;
use steel_utils::random::{PositionalRandom, Random};
use steel_utils::{BoundingBox, Identifier, Rotation};

use crate::structure::{
    GenerationStub, Structure, StructureGenerationContext, StructurePiece, StructurePiecePayload,
};

/// A placed piece produced by jigsaw assembly.
#[derive(Debug, Clone)]
pub struct PlacedPiece {
    /// Selected pool element.
    pub element: PoolElement,
    /// Template location (Single/LegacySingle).
    pub template_location: Option<Identifier>,
    /// World-space origin.
    pub position: IVec3,
    /// Rotation.
    pub rotation: Rotation,
    /// Template-sized BB (used for beardifier + world save).
    pub bounding_box: BoundingBox,
    /// Assembly-time BB, possibly expanded vertically by the expansion hack.
    /// Used only during assembly — not persisted.
    pub assembly_bb: BoundingBox,
    /// Ground-level delta for Beardifier.
    pub ground_level_delta: i32,
    /// Rigid or terrain-matching.
    pub projection: Projection,
    /// BFS tree depth.
    pub depth: i32,
    /// Junctions to neighbors.
    pub junctions: Vec<JigsawJunction>,
}

/// Typed state needed to place or compare a vanilla jigsaw piece.
#[derive(Debug, Clone)]
pub struct JigsawPieceData {
    /// Selected pool element.
    pub pool_element: PoolElement,
    /// World-space template origin.
    pub position: IVec3,
    /// Template rotation.
    pub rotation: Rotation,
    /// Liquid handling mode for block placement.
    pub liquid_settings: LiquidSettingsData,
}

/// Junction between two jigsaw pieces (terrain adaptation).
#[derive(Debug, Clone)]
pub struct JigsawJunction {
    /// World-space source position.
    pub source_pos: IVec3,
    /// Y delta between source and target.
    pub delta_y: i32,
    /// Destination projection.
    pub dest_projection: Projection,
}

/// Resolves pool aliases for a specific structure instance.
pub fn resolve_aliases(
    aliases: &[PoolAlias],
    rng: &mut impl Random,
) -> FxHashMap<Identifier, Identifier> {
    let mut map = FxHashMap::default();
    for alias in aliases {
        match alias {
            PoolAlias::Direct { alias, target } => {
                map.insert(alias.clone(), target.clone());
            }
            PoolAlias::Random { alias, targets } => {
                let total: i32 = targets.iter().map(|(_, w)| *w).sum();
                if total > 0 {
                    let mut pick = rng.next_i32_bounded(total);
                    for (target, weight) in targets {
                        pick -= weight;
                        if pick < 0 {
                            map.insert(alias.clone(), target.clone());
                            break;
                        }
                    }
                }
            }
            PoolAlias::RandomGroup { groups } => {
                let total: i32 = groups.iter().map(|(_, w)| *w).sum();
                if total > 0 {
                    let mut pick = rng.next_i32_bounded(total);
                    for (bindings, weight) in groups {
                        pick -= weight;
                        if pick < 0 {
                            for (alias, target) in bindings {
                                map.insert(alias.clone(), target.clone());
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
    map
}

fn sample_start_height(config: &JigsawConfig, rng: &mut impl Random) -> i32 {
    match &config.start_height {
        StartHeight::Constant(y) => *y,
        StartHeight::Uniform { min, max } => rng.next_i32_between(*min, *max),
    }
}

/// Java integer midpoint used by vanilla jigsaw placement: `(min + max) / 2`.
const fn java_center(min: i32, max: i32) -> i32 {
    min.wrapping_add(max) / 2
}

static SYNTHETIC_BOTTOM_JIGSAW: Identifier = Identifier::new_static("minecraft", "bottom");
static SYNTHETIC_EMPTY_POOL: Identifier = Identifier::new_static("minecraft", "empty");

type PoolTemplateCache<'a> = FxHashMap<Identifier, Vec<&'a PoolElement>>;

const fn rotation_index(rotation: Rotation) -> usize {
    match rotation {
        Rotation::None => 0,
        Rotation::Clockwise90 => 1,
        Rotation::Clockwise180 => 2,
        Rotation::CounterClockwise90 => 3,
    }
}

/// Vanilla-matching shuffle (reverse Fisher-Yates).
fn vanilla_shuffle<T>(list: &mut [T], rng: &mut LegacyRandom) {
    for i in (1..list.len()).rev() {
        let j = rng.next_i32_bounded((i + 1) as i32) as usize;
        list.swap(i, j);
    }
}

/// Vanilla `Util.shuffle` then stable priority grouping using precomputed priorities.
fn shuffled_jigsaw_indices(template: &TemplateData, rng: &mut LegacyRandom) -> Vec<usize> {
    if template.jigsaws.is_empty() {
        return Vec::new();
    }

    let mut indices: Vec<usize> = (0..template.jigsaws.len()).collect();
    vanilla_shuffle(&mut indices, rng);
    if template.selection_priorities_desc.len() > 1 {
        let shuffled = indices;
        let mut ordered = Vec::with_capacity(shuffled.len());
        for &priority in &template.selection_priorities_desc {
            ordered.extend(
                shuffled
                    .iter()
                    .copied()
                    .filter(|&idx| template.jigsaws[idx].selection_priority == priority),
            );
        }
        indices = ordered;
    }
    indices
}

fn template_rotated_jigsaws(template: &TemplateData, rotation: Rotation) -> &[JigsawBlock] {
    &template.rotated_jigsaws[rotation_index(rotation)]
}

fn jigsaw_view(block: &JigsawBlock) -> TransformedJigsaw<'_> {
    TransformedJigsaw {
        pos: IVec3::from(block.pos),
        orientation: block.orientation,
        name: &block.name,
        target: &block.target,
        pool: &block.pool,
        joint: block.joint,
        placement_priority: block.placement_priority,
    }
}

/// Gets the template location from a pool element.
///
/// For `List` elements, delegates to the first sub-element matching vanilla's
/// `ListPoolElement` which uses `elements.get(0)` for jigsaws and BB.
fn element_location(element: &PoolElement) -> Option<&Identifier> {
    match element {
        PoolElement::Single { location, .. } | PoolElement::LegacySingle { location, .. } => {
            Some(location)
        }
        PoolElement::List { elements, .. } => elements.first().and_then(element_location),
        _ => None,
    }
}

/// Gets shuffled jigsaws for a pool element at a given position and rotation.
///
/// Returns the jigsaws with their positions transformed by rotation, sorted
/// by `selection_priority` (descending), then shuffled within equal priorities.
fn get_shuffled_jigsaws<'a>(
    element: &PoolElement,
    templates: &'a FxHashMap<Identifier, TemplateData>,
    rotation: Rotation,
    rng: &mut LegacyRandom,
) -> Vec<TransformedJigsaw<'a>> {
    let Some(location) = element_location(element) else {
        // Feature/Empty elements: synthetic jigsaw at origin facing down
        return vec![TransformedJigsaw {
            pos: IVec3::ZERO,
            orientation: JigsawOrientation::DownSouth,
            name: &SYNTHETIC_BOTTOM_JIGSAW,
            target: &SYNTHETIC_EMPTY_POOL,
            pool: &SYNTHETIC_EMPTY_POOL,
            joint: JointType::Rollable,
            placement_priority: 0,
        }];
    };

    let Some(template) = templates.get(location) else {
        return vec![];
    };

    let rotated = template_rotated_jigsaws(template, rotation);
    shuffled_jigsaw_indices(template, rng)
        .into_iter()
        .map(|idx| jigsaw_view(&rotated[idx]))
        .collect()
}

/// A jigsaw block with its position transformed by rotation.
#[derive(Clone, Copy)]
struct TransformedJigsaw<'a> {
    pos: IVec3,
    orientation: JigsawOrientation,
    name: &'a Identifier,
    target: &'a Identifier,
    pool: &'a Identifier,
    joint: JointType,
    placement_priority: i32,
}

/// Vanilla's `StructureTemplatePool.getMaxSize` — max Y span across all templates.
fn pool_max_y_size(
    pool: &TemplatePoolData,
    templates: &FxHashMap<Identifier, TemplateData>,
) -> i32 {
    pool.elements
        .iter()
        .filter_map(|(element, _)| {
            let (PoolElement::Single { location: loc, .. }
            | PoolElement::LegacySingle { location: loc, .. }) = element
            else {
                return None;
            };
            templates.get(loc).map(|t| t.size[1])
        })
        .max()
        .unwrap_or(0)
}

/// Checks if two jigsaws can connect.
///
/// Vanilla's `JigsawBlock.canAttach`: opposite facing, name match, joint compatibility.
fn can_attach(source: &TransformedJigsaw, target: &TransformedJigsaw) -> bool {
    let source_front = source.orientation.front_direction();
    let target_front = target.orientation.front_direction();

    if source_front != target_front.opposite() {
        return false;
    }
    if source.target != target.name {
        return false;
    }
    if source.joint == JointType::Aligned {
        let source_top = source.orientation.top_direction();
        let target_top = target.orientation.top_direction();
        if source_top != target_top {
            return false;
        }
    }
    true
}

/// Gets the bounding box for a pool element at a position with rotation.
///
/// Feature elements return a 1×1×1 BB at the given position, matching
/// vanilla's `FeaturePoolElement.getBoundingBox`.
/// List elements return the encapsulating BB of all sub-elements, matching
/// vanilla's `ListPoolElement.getBoundingBox`.
fn element_bounding_box(
    element: &PoolElement,
    templates: &FxHashMap<Identifier, TemplateData>,
    pos: IVec3,
    rotation: Rotation,
) -> Option<BoundingBox> {
    match element {
        PoolElement::Feature { .. } => Some(BoundingBox::new(pos, pos)),
        PoolElement::List { elements, .. } => {
            let mut result: Option<BoundingBox> = None;
            for sub in elements {
                if let Some(sub_bb) = element_bounding_box(sub, templates, pos, rotation) {
                    result = Some(match result {
                        Some(prev) => BoundingBox::new(
                            IVec3::new(
                                prev.min_x().min(sub_bb.min_x()),
                                prev.min_y().min(sub_bb.min_y()),
                                prev.min_z().min(sub_bb.min_z()),
                            ),
                            IVec3::new(
                                prev.max_x().max(sub_bb.max_x()),
                                prev.max_y().max(sub_bb.max_y()),
                                prev.max_z().max(sub_bb.max_z()),
                            ),
                        ),
                        None => sub_bb,
                    });
                }
            }
            result
        }
        _ => {
            let location = element_location(element)?;
            let template = templates.get(location)?;
            let size = IVec3::from(template.size);
            Some(rotation.get_bounding_box(pos, size))
        }
    }
}

fn expand_pool_weights(pool: &TemplatePoolData) -> Vec<&PoolElement> {
    let mut expanded = Vec::with_capacity(pool.elements.iter().map(|(_, w)| *w as usize).sum());
    for (element, weight) in &pool.elements {
        for _ in 0..*weight {
            expanded.push(element);
        }
    }
    expanded
}

/// Appends vanilla's `StructureTemplatePool.getShuffledTemplates` to `out`.
fn append_shuffled_templates_cached<'a>(
    pool: &'a TemplatePoolData,
    cache: &mut PoolTemplateCache<'a>,
    rng: &mut LegacyRandom,
    out: &mut Vec<&'a PoolElement>,
) {
    let expanded = cache
        .entry(pool.key.clone())
        .or_insert_with(|| expand_pool_weights(pool));
    let start = out.len();
    out.extend(expanded.iter().copied());
    vanilla_shuffle(&mut out[start..], rng);
}

/// Vanilla's `StructureTemplatePool.getRandomTemplate`.
fn get_random_template<'a>(pool: &'a TemplatePoolData, rng: &mut LegacyRandom) -> &'a PoolElement {
    let expanded = expand_pool_weights(pool);
    if expanded.is_empty() {
        static EMPTY: PoolElement = PoolElement::Empty;
        return &EMPTY;
    }
    let idx = rng.next_i32_bounded(expanded.len() as i32) as usize;
    expanded[idx]
}

/// Hierarchical free-space tracker. Vanilla uses `MutableObject<VoxelShape>`
/// with subtraction; for integer-aligned BBs, `constraint + occupied` is
/// equivalent. Internal children share the source's internal free space;
/// external children share the parent's context.
struct FreeSpace {
    constraint: BoundingBox,
    occupied: Vec<BoundingBox>,
}

impl FreeSpace {
    fn collides(&self, candidate: &BoundingBox) -> bool {
        if candidate.min_x() < self.constraint.min_x()
            || candidate.max_x() > self.constraint.max_x()
            || candidate.min_y() < self.constraint.min_y()
            || candidate.max_y() > self.constraint.max_y()
            || candidate.min_z() < self.constraint.min_z()
            || candidate.max_z() > self.constraint.max_z()
        {
            return true;
        }
        self.occupied.iter().any(|p| candidate.intersects(*p))
    }
}

/// Result of a successful jigsaw assembly.
pub struct AssemblyResult {
    /// The placed pieces.
    pub pieces: Vec<PlacedPiece>,
    /// The biome check position (centerX, centerY, centerZ from the `GenerationStub`).
    pub biome_check_pos: IVec3,
}

struct StartedAssembly {
    pieces: Vec<PlacedPiece>,
    biome_check_pos: IVec3,
}

/// Vanilla's `JigsawPlacement.addPieces` before the lazy `GenerationStub` child builder.
#[expect(
    clippy::too_many_arguments,
    reason = "matches vanilla's addPieces call surface"
)]
fn start_assembly(
    config: &JigsawConfig,
    rng: &mut LegacyRandom,
    chunk_x: i32,
    chunk_z: i32,
    pools: &FxHashMap<Identifier, TemplatePoolData>,
    templates: &FxHashMap<Identifier, TemplateData>,
    alias_map: &FxHashMap<Identifier, Identifier>,
    get_height: &mut dyn FnMut(i32, i32) -> i32,
    min_y: i32,
    max_y: i32,
) -> Option<StartedAssembly> {
    let start_y = sample_start_height(config, rng);
    let start_x = chunk_x * 16;
    let start_z = chunk_z * 16;
    let center_rotation = Rotation::get_random(rng);

    let start_pool_key = alias_map
        .get(&config.start_pool)
        .unwrap_or(&config.start_pool);
    let start_pool = pools.get(start_pool_key)?;
    let center_element = get_random_template(start_pool, rng);
    if center_element.is_empty() {
        return None;
    }

    let anchor_offset = if let Some(ref jigsaw_name) = config.start_jigsaw_name {
        let jigsaws = get_shuffled_jigsaws(center_element, templates, center_rotation, rng);
        let j = jigsaws.iter().find(|j| j.name == jigsaw_name)?;
        j.pos
    } else {
        IVec3::ZERO
    };

    let adjusted = IVec3::new(
        start_x - anchor_offset.x,
        start_y - anchor_offset.y,
        start_z - anchor_offset.z,
    );

    let center_bb = element_bounding_box(center_element, templates, adjusted, center_rotation)?;

    let bottom_y = if config.project_start_to_heightmap.is_some() {
        let mid_x = java_center(center_bb.min_x(), center_bb.max_x());
        let mid_z = java_center(center_bb.min_z(), center_bb.max_z());
        start_y + get_height(mid_x, mid_z)
    } else {
        adjusted.y
    };

    let ground_level_delta = center_element.projection().ground_level_delta();
    let dy = bottom_y - (center_bb.min_y() + ground_level_delta);
    let center_bb = BoundingBox::new(
        IVec3::new(center_bb.min_x(), center_bb.min_y() + dy, center_bb.min_z()),
        IVec3::new(center_bb.max_x(), center_bb.max_y() + dy, center_bb.max_z()),
    );
    let adjusted_y = adjusted.y + dy;

    let padding = &config.dimension_padding;
    if center_bb.min_y() < min_y + padding.bottom || center_bb.max_y() > max_y - 1 - padding.top {
        return None;
    }

    let pieces = vec![PlacedPiece {
        element: center_element.clone(),
        template_location: element_location(center_element).cloned(),
        position: IVec3::new(adjusted.x, adjusted_y, adjusted.z),
        rotation: center_rotation,
        bounding_box: center_bb,
        assembly_bb: center_bb,
        ground_level_delta,
        projection: center_element.projection(),
        depth: 0,
        junctions: Vec::new(),
    }];

    let center_stub_x = java_center(center_bb.min_x(), center_bb.max_x());
    let center_stub_z = java_center(center_bb.min_z(), center_bb.max_z());
    let center_stub_y = bottom_y + anchor_offset.y;
    let biome_check_pos = IVec3::new(center_stub_x, center_stub_y, center_stub_z);

    Some(StartedAssembly {
        pieces,
        biome_check_pos,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "matches vanilla's addPieces child-builder call surface"
)]
fn finish_assembly<'a>(
    mut started: StartedAssembly,
    config: &JigsawConfig,
    rng: &mut LegacyRandom,
    pools: &'a FxHashMap<Identifier, TemplatePoolData>,
    templates: &'a FxHashMap<Identifier, TemplateData>,
    alias_map: &FxHashMap<Identifier, Identifier>,
    get_height: &mut dyn FnMut(i32, i32) -> i32,
    min_y: i32,
    max_y: i32,
) -> AssemblyResult {
    let biome_check_pos = started.biome_check_pos;

    if config.max_depth <= 0 {
        return AssemblyResult {
            pieces: started.pieces,
            biome_check_pos,
        };
    }

    let Some(center_piece) = started.pieces.first() else {
        return AssemblyResult {
            pieces: started.pieces,
            biome_check_pos,
        };
    };
    let center_bb = center_piece.assembly_bb;
    let center_stub_x = biome_check_pos.x;
    let center_stub_y = biome_check_pos.y;
    let center_stub_z = biome_check_pos.z;

    let max_dist = config.max_distance_from_center;
    let constraint_bb = BoundingBox::new(
        IVec3::new(
            center_stub_x - max_dist,
            (center_stub_y - max_dist).max(min_y + config.dimension_padding.bottom),
            center_stub_z - max_dist,
        ),
        IVec3::new(
            center_stub_x + max_dist,
            (center_stub_y + max_dist).min(max_y - 1 - config.dimension_padding.top),
            center_stub_z + max_dist,
        ),
    );

    let mut free_spaces: Vec<FreeSpace> = {
        let mut space = FreeSpace::new(constraint_bb);
        space.occupied.add_box(center_bb);
        vec![space]
    };
    let mut pool_template_cache = PoolTemplateCache::default();
    let mut queue: Vec<(usize, i32, i32, usize)> = Vec::new();

    try_placing_children(
        0,
        0,
        0,
        config,
        pools,
        templates,
        alias_map,
        &mut pool_template_cache,
        &mut started.pieces,
        &mut free_spaces,
        &mut queue,
        rng,
        get_height,
    );

    while !queue.is_empty() {
        queue.sort_by_key(|entry| Reverse(entry.2));
        let (piece_idx, depth, _priority, context_idx) = queue.remove(0);
        try_placing_children(
            piece_idx,
            depth,
            context_idx,
            config,
            pools,
            templates,
            alias_map,
            &mut pool_template_cache,
            &mut started.pieces,
            &mut free_spaces,
            &mut queue,
            rng,
            get_height,
        );
    }

    AssemblyResult {
        pieces: started.pieces,
        biome_check_pos,
    }
}

/// Vanilla's `JigsawPlacement.addPieces`. Returns `None` on failure (empty start
/// pool, dimension padding violation, etc.).
#[expect(
    clippy::too_many_arguments,
    reason = "matches vanilla's addPieces call surface"
)]
#[expect(
    clippy::implicit_hasher,
    reason = "FxHashMap avoids SipHash overhead on Identifier lookups"
)]
pub fn assemble(
    config: &JigsawConfig,
    rng: &mut LegacyRandom,
    chunk_x: i32,
    chunk_z: i32,
    pools: &FxHashMap<Identifier, TemplatePoolData>,
    templates: &FxHashMap<Identifier, TemplateData>,
    alias_map: &FxHashMap<Identifier, Identifier>,
    get_height: &mut dyn FnMut(i32, i32) -> i32,
    min_y: i32,
    max_y: i32,
) -> Option<AssemblyResult> {
    let started = start_assembly(
        config, rng, chunk_x, chunk_z, pools, templates, alias_map, get_height, min_y, max_y,
    )?;
    Some(finish_assembly(
        started, config, rng, pools, templates, alias_map, get_height, min_y, max_y,
    ))
}

/// Registered under `minecraft:jigsaw` for pool-based structures such as villages,
/// bastions, ancient cities, and trail ruins.
pub struct JigsawStructure;

impl Structure for JigsawStructure {
    fn find_generation_point(
        &self,
        ctx: &mut dyn StructureGenerationContext,
        structure: &StructureData,
        _rng: &mut LegacyRandom,
    ) -> Option<GenerationStub> {
        let config = structure.config.as_jigsaw()?;

        let mut alias_position_rng = LegacyRandom::from_seed(0);
        alias_position_rng.set_large_feature_seed(ctx.seed(), ctx.chunk_x(), ctx.chunk_z());
        let start_y = sample_start_height(config, &mut alias_position_rng);
        let mut alias_source = LegacyRandom::from_seed(ctx.seed() as u64);
        let mut alias_rng =
            alias_source
                .next_positional()
                .at(ctx.chunk_min_x(), start_y, ctx.chunk_min_z());
        let alias_map = resolve_aliases(&config.pool_aliases, &mut alias_rng);

        let mut assembly_rng = LegacyRandom::from_seed(0);
        assembly_rng.set_large_feature_seed(ctx.seed(), ctx.chunk_x(), ctx.chunk_z());

        let started = {
            let mut get_height = |x: i32, z: i32| ctx.terrain_surface_height(x, z, false);
            start_assembly(
                config,
                &mut assembly_rng,
                ctx.chunk_x(),
                ctx.chunk_z(),
                ctx.template_pools(),
                ctx.templates(),
                &alias_map,
                &mut get_height,
                ctx.min_y(),
                ctx.max_y(),
            )?
        };

        if started.pieces.is_empty() {
            return None;
        }

        let biome = ctx.biome_at(
            started.biome_check_pos.x,
            started.biome_check_pos.y,
            started.biome_check_pos.z,
        );
        if !structure.allowed_biomes.contains(&biome.key) {
            return None;
        }

        let assembly = {
            let mut get_height = |x: i32, z: i32| ctx.terrain_surface_height(x, z, false);
            finish_assembly(
                started,
                config,
                &mut assembly_rng,
                ctx.template_pools(),
                ctx.templates(),
                &alias_map,
                &mut get_height,
                ctx.min_y(),
                ctx.max_y(),
            )
        };

        let pieces = assembly
            .pieces
            .into_iter()
            .map(|piece| StructurePiece {
                piece_type: Identifier::new_static("minecraft", "jigsaw"),
                bounding_box: piece.assembly_bb,
                gen_depth: 0,
                orientation: None,
                payload: StructurePiecePayload::Jigsaw(JigsawPieceData {
                    pool_element: piece.element,
                    position: piece.position,
                    rotation: piece.rotation,
                    liquid_settings: config.liquid_settings,
                }),
                ground_level_delta: piece.ground_level_delta,
                junctions: piece.junctions,
                projection: Some(piece.projection),
            })
            .collect();

        Some(GenerationStub {
            position: (
                assembly.biome_check_pos.x,
                assembly.biome_check_pos.y,
                assembly.biome_check_pos.z,
            ),
            pieces,
        })
    }
}

/// Vanilla's `tryPlacingChildren`. `context_idx` is this piece's collision context
/// in `free_spaces` — external children get the parent's context, internal
/// children get the parent's internal free space.
#[expect(
    clippy::too_many_arguments,
    reason = "matches vanilla's tryPlacingChildren signature"
)]
#[expect(
    clippy::too_many_lines,
    reason = "inlined to mirror vanilla's source-jigsaw/child-pool loop"
)]
fn try_placing_children<'a>(
    source_idx: usize,
    depth: i32,
    context_idx: usize,
    config: &JigsawConfig,
    pools: &'a FxHashMap<Identifier, TemplatePoolData>,
    templates: &'a FxHashMap<Identifier, TemplateData>,
    alias_map: &FxHashMap<Identifier, Identifier>,
    pool_template_cache: &mut PoolTemplateCache<'a>,
    pieces: &mut Vec<PlacedPiece>,
    free_spaces: &mut Vec<FreeSpace>,
    queue: &mut Vec<(usize, i32, i32, usize)>,
    rng: &mut LegacyRandom,
    get_height: &mut dyn FnMut(i32, i32) -> i32,
) {
    let (source_jigsaws, source_bb, source_projection, source_ground_level_delta) = {
        let source_piece = &pieces[source_idx];
        let mut jigsaws = get_shuffled_jigsaws(
            &source_piece.element,
            templates,
            source_piece.rotation,
            rng,
        );
        if jigsaws.is_empty() {
            return;
        }
        let origin = source_piece.position;
        for jigsaw in &mut jigsaws {
            jigsaw.pos += origin;
        }
        (
            jigsaws,
            source_piece.assembly_bb,
            source_piece.projection,
            source_piece.ground_level_delta,
        )
    };
    let source_box_y = source_bb.min_y();
    let source_rigid = source_projection == Projection::Rigid;

    let mut internal_ctx_idx: Option<usize> = None;
    let mut candidates: Vec<&PoolElement> = Vec::new();

    'source_jigsaw: for source_jigsaw in &source_jigsaws {
        candidates.clear();
        let front = source_jigsaw.orientation.front_direction();
        let foff = front.offset_vec();
        let target_jigsaw_world = source_jigsaw.pos + foff;

        let source_jigsaw_local_y = source_jigsaw.pos.y - source_box_y;

        let pool_key = alias_map
            .get(source_jigsaw.pool)
            .unwrap_or(source_jigsaw.pool);
        let raw_pool = pools.get(pool_key);
        let target_pool = raw_pool.filter(|p| !p.elements.is_empty());
        let fallback_pool = raw_pool
            .and_then(|p| pools.get(&p.fallback))
            .filter(|p| !p.elements.is_empty());

        let attach_inside = source_bb.contains_xyz(
            target_jigsaw_world.x,
            target_jigsaw_world.y,
            target_jigsaw_world.z,
        );

        if depth != config.max_depth
            && let Some(pool) = target_pool
        {
            append_shuffled_templates_cached(pool, pool_template_cache, rng, &mut candidates);
        }
        if let Some(fallback) = fallback_pool {
            append_shuffled_templates_cached(fallback, pool_template_cache, rng, &mut candidates);
        }

        let placement_priority = source_jigsaw.placement_priority;
        let mut source_jigsaw_base_height: Option<i32> = None;

        for &candidate_element in &candidates {
            if candidate_element.is_empty() {
                break;
            }

            let candidate_projection = candidate_element.projection();
            let candidate_rigid = candidate_projection == Projection::Rigid;

            let rotations = Rotation::get_shuffled(rng);
            for candidate_rotation in rotations {
                let expand_to = if config.use_expansion_hack {
                    let hack_data =
                        element_location(candidate_element).and_then(|loc| templates.get(loc));
                    if let Some(template_data) = hack_data {
                        let hack_box = candidate_rotation
                            .get_bounding_box(IVec3::ZERO, IVec3::from(template_data.size));
                        if hack_box.max_y() - hack_box.min_y() < 16 {
                            template_data
                                .rotated_jigsaws[rotation_index(candidate_rotation)]
                                .iter()
                                .map(|j| {
                                    let pos = IVec3::from(j.pos);
                                    let front = j.orientation.front_direction();
                                    let front_pos = pos + front.offset_vec();
                                    if !hack_box.contains_xyz(front_pos.x, front_pos.y, front_pos.z)
                                    {
                                        return 0;
                                    }
                                    let child_pool_key = alias_map.get(&j.pool).unwrap_or(&j.pool);
                                    let child_pool_size = pools
                                        .get(child_pool_key)
                                        .map_or(0, |p| pool_max_y_size(p, templates));
                                    let child_fallback_size = pools
                                        .get(child_pool_key)
                                        .and_then(|p| pools.get(&p.fallback))
                                        .map_or(0, |p| pool_max_y_size(p, templates));
                                    child_pool_size.max(child_fallback_size)
                                })
                                .max()
                                .unwrap_or(0)
                        } else {
                            0
                        }
                    } else {
                        0
                    }
                } else {
                    0
                };

                let mut candidate_bb_at_origin: Option<BoundingBox> = None;

                let mut try_target_jigsaw = |target_jigsaw: TransformedJigsaw<'_>| -> bool {
                    if !can_attach(source_jigsaw, &target_jigsaw) {
                        return false;
                    }

                    let target_jigsaw_local = target_jigsaw.pos;

                    let raw_target = IVec3::new(
                        target_jigsaw_world.x - target_jigsaw_local.x,
                        0,
                        target_jigsaw_world.z - target_jigsaw_local.z,
                    );

                    let raw_bb = if let Some(bb) = candidate_bb_at_origin {
                        bb.translate(IVec3::new(raw_target.x, 0, raw_target.z))
                    } else {
                        let Some(bb) = element_bounding_box(
                            candidate_element,
                            templates,
                            IVec3::ZERO,
                            candidate_rotation,
                        ) else {
                            return false;
                        };
                        candidate_bb_at_origin = Some(bb);
                        bb.translate(IVec3::new(raw_target.x, 0, raw_target.z))
                    };

                    let target_jigsaw_local_y = target_jigsaw_local.y;
                    let delta_y = source_jigsaw_local_y - target_jigsaw_local_y + foff.y;

                    let target_box_y = if source_rigid && candidate_rigid {
                        source_box_y + delta_y
                    } else {
                        let base_height = *source_jigsaw_base_height.get_or_insert_with(|| {
                            get_height(source_jigsaw.pos.x, source_jigsaw.pos.z)
                        });
                        base_height - target_jigsaw_local_y
                    };

                    let y_offset = target_box_y - raw_bb.min_y();
                    let candidate_bb = BoundingBox::new(
                        IVec3::new(raw_bb.min_x(), raw_bb.min_y() + y_offset, raw_bb.min_z()),
                        IVec3::new(raw_bb.max_x(), raw_bb.max_y() + y_offset, raw_bb.max_z()),
                    );
                    let target_position =
                        IVec3::new(raw_target.x, raw_bb.min_y() + y_offset, raw_target.z);

                    let expanded_bb = if expand_to > 0 {
                        let new_size =
                            (expand_to + 1).max(candidate_bb.max_y() - candidate_bb.min_y());
                        BoundingBox::new(
                            IVec3::new(
                                candidate_bb.min_x(),
                                candidate_bb.min_y(),
                                candidate_bb.min_z(),
                            ),
                            IVec3::new(
                                candidate_bb.max_x(),
                                candidate_bb.min_y() + new_size,
                                candidate_bb.max_z(),
                            ),
                        )
                    } else {
                        candidate_bb
                    };

                    let effective_ctx = if attach_inside {
                        *internal_ctx_idx.get_or_insert_with(|| {
                            free_spaces.push(FreeSpace {
                                constraint: source_bb,
                                occupied: Vec::new(),
                            });
                            free_spaces.len() - 1
                        })
                    } else {
                        context_idx
                    };

                    if free_spaces[effective_ctx].collides(&expanded_bb) {
                        return false;
                    }

                    free_spaces[effective_ctx].occupied.push(expanded_bb);

                    let target_ground_level_delta = if candidate_rigid {
                        source_ground_level_delta - delta_y
                    } else {
                        candidate_projection.ground_level_delta()
                    };

                    let junction_y = if source_rigid {
                        source_box_y + source_jigsaw_local_y
                    } else if candidate_rigid {
                        target_box_y + target_jigsaw_local_y
                    } else {
                        let base_height = *source_jigsaw_base_height.get_or_insert_with(|| {
                            get_height(source_jigsaw.pos.x, source_jigsaw.pos.z)
                        });
                        base_height + delta_y / 2
                    };

                    pieces[source_idx].junctions.push(JigsawJunction {
                        source_pos: IVec3::new(
                            target_jigsaw_world.x,
                            junction_y - source_jigsaw_local_y + source_ground_level_delta,
                            target_jigsaw_world.z,
                        ),
                        delta_y,
                        dest_projection: candidate_projection,
                    });

                    let new_piece_idx = pieces.len();
                    let mut target_piece = PlacedPiece {
                        element: candidate_element.clone(),
                        template_location: element_location(candidate_element).cloned(),
                        position: target_position,
                        rotation: candidate_rotation,
                        bounding_box: candidate_bb,
                        assembly_bb: expanded_bb,
                        ground_level_delta: target_ground_level_delta,
                        projection: candidate_projection,
                        depth: depth + 1,
                        junctions: Vec::new(),
                    };

                    target_piece.junctions.push(JigsawJunction {
                        source_pos: IVec3::new(
                            source_jigsaw.pos.x,
                            junction_y - target_jigsaw_local_y + target_ground_level_delta,
                            source_jigsaw.pos.z,
                        ),
                        delta_y: -delta_y,
                        dest_projection: source_projection,
                    });

                    pieces.push(target_piece);

                    if depth < config.max_depth {
                        queue.push((new_piece_idx, depth + 1, placement_priority, effective_ctx));
                    }

                    true
                };

                if let Some(location) = element_location(candidate_element) {
                    let Some(template) = templates.get(location) else {
                        continue;
                    };
                    let rotated = template_rotated_jigsaws(template, candidate_rotation);
                    for target_jigsaw_idx in shuffled_jigsaw_indices(template, rng) {
                        if try_target_jigsaw(jigsaw_view(&rotated[target_jigsaw_idx])) {
                            continue 'source_jigsaw;
                        }
                    }
                } else {
                    for target_jigsaw in get_shuffled_jigsaws(
                        candidate_element,
                        templates,
                        candidate_rotation,
                        rng,
                    ) {
                        if try_target_jigsaw(target_jigsaw) {
                            continue 'source_jigsaw;
                        }
                    }
                }
            }
        }
    }
}
