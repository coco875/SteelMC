//! Template pool and structure template data for jigsaw assembly.
//!
//! Parsed at build time from vanilla datapack JSONs and structure NBT files.
//! Used by the jigsaw placement system to assemble structures from pools.

use std::io::Read;

use crate::structure_processor::StructureProcessorKind;
use steel_utils::{Direction, Identifier, Rotation};

/// Orientation of a jigsaw block, encoding both facing direction and up direction.
///
/// Vanilla's `FrontAndTop` enum — the orientation block state property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JigsawOrientation {
    DownEast,
    DownNorth,
    DownSouth,
    DownWest,
    UpEast,
    UpNorth,
    UpSouth,
    UpWest,
    WestUp,
    EastUp,
    NorthUp,
    SouthUp,
}

impl JigsawOrientation {
    /// Parses from the block state property string (e.g., `"up_north"`).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "down_east" => Self::DownEast,
            "down_north" => Self::DownNorth,
            "down_south" => Self::DownSouth,
            "down_west" => Self::DownWest,
            "up_east" => Self::UpEast,
            "up_north" => Self::UpNorth,
            "up_south" => Self::UpSouth,
            "up_west" => Self::UpWest,
            "west_up" => Self::WestUp,
            "east_up" => Self::EastUp,
            "north_up" => Self::NorthUp,
            "south_up" => Self::SouthUp,
            _ => return None,
        })
    }

    /// Returns the front-facing direction.
    #[must_use]
    pub const fn front_direction(self) -> Direction {
        match self {
            Self::DownEast | Self::DownNorth | Self::DownSouth | Self::DownWest => Direction::Down,
            Self::UpEast | Self::UpNorth | Self::UpSouth | Self::UpWest => Direction::Up,
            Self::WestUp => Direction::West,
            Self::EastUp => Direction::East,
            Self::NorthUp => Direction::North,
            Self::SouthUp => Direction::South,
        }
    }

    /// Returns the top direction.
    #[must_use]
    pub const fn top_direction(self) -> Direction {
        match self {
            Self::DownEast | Self::UpEast => Direction::East,
            Self::DownNorth | Self::UpNorth => Direction::North,
            Self::DownSouth | Self::UpSouth => Direction::South,
            Self::DownWest | Self::UpWest => Direction::West,
            Self::WestUp | Self::EastUp | Self::NorthUp | Self::SouthUp => Direction::Up,
        }
    }

    /// Returns the front-facing direction offset as (dx, dy, dz).
    #[must_use]
    pub fn front(self) -> (i32, i32, i32) {
        self.front_direction().offset()
    }

    /// Constructs an orientation from front and top directions.
    ///
    /// Returns `None` if the combination is invalid.
    #[must_use]
    pub const fn from_directions(front: Direction, top: Direction) -> Option<Self> {
        // Match vanilla's FrontAndTop lookup table
        Some(match (front, top) {
            (Direction::Down, Direction::East) => Self::DownEast,
            (Direction::Down, Direction::North) => Self::DownNorth,
            (Direction::Down, Direction::South) => Self::DownSouth,
            (Direction::Down, Direction::West) => Self::DownWest,
            (Direction::Up, Direction::East) => Self::UpEast,
            (Direction::Up, Direction::North) => Self::UpNorth,
            (Direction::Up, Direction::South) => Self::UpSouth,
            (Direction::Up, Direction::West) => Self::UpWest,
            (Direction::West, Direction::Up) => Self::WestUp,
            (Direction::East, Direction::Up) => Self::EastUp,
            (Direction::North, Direction::Up) => Self::NorthUp,
            (Direction::South, Direction::Up) => Self::SouthUp,
            _ => return None,
        })
    }

    /// Rotates this orientation by the given rotation around the Y axis.
    ///
    /// Both the front and top directions are rotated, matching vanilla's
    /// `BlockState.rotate(rotation)` for jigsaw blocks.
    #[must_use]
    pub fn rotate(self, rotation: Rotation) -> Self {
        let front = rotation.rotate(self.front_direction());
        let top = rotation.rotate(self.top_direction());
        // Rotation of valid FrontAndTop always produces a valid FrontAndTop
        Self::from_directions(front, top).expect("rotated orientation should be valid")
    }
}

/// Joint type for jigsaw connections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointType {
    /// Can rotate freely around the connection axis.
    Rollable,
    /// Must maintain alignment with the source piece.
    Aligned,
}

/// A jigsaw connector block extracted from a structure template.
#[derive(Debug, Clone)]
pub struct JigsawBlock {
    /// Position relative to template origin.
    pub pos: [i32; 3],
    /// Orientation (determines facing direction).
    pub orientation: JigsawOrientation,
    /// Name of this jigsaw connector.
    pub name: Identifier,
    /// Target connector name to attach to.
    pub target: Identifier,
    /// Pool to draw target pieces from.
    pub pool: Identifier,
    /// Joint type.
    pub joint: JointType,
    /// Block state string to replace jigsaw with after placement.
    pub final_state: String,
    /// Priority for selecting this jigsaw among siblings in a piece (higher = tried first).
    pub selection_priority: i32,
    /// Priority for BFS queue ordering when placing children (higher = processed first).
    pub placement_priority: i32,
}

/// Extracted data from a structure template NBT file.
///
/// Contains only the information needed for jigsaw assembly — not the full
/// block data (which is loaded separately for actual placement).
#[derive(Debug, Clone)]
pub struct TemplateData {
    /// Template size in blocks (x, y, z).
    pub size: [i32; 3],
    /// Jigsaw connector blocks in this template.
    pub jigsaws: Vec<JigsawBlock>,
}

/// Projection mode for pool elements.
///
/// Vanilla's `StructureTemplatePool.Projection`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projection {
    /// Fixed Y position, stacked based on jigsaw positions.
    Rigid,
    /// Adjusts vertically to match terrain surface.
    TerrainMatching,
}

impl Projection {
    /// Returns the ground level delta for this projection.
    ///
    /// Vanilla's `StructurePoolElement.getGroundLevelDelta()` returns 1 by default.
    /// This is the offset from the piece's minY to ground level.
    #[must_use]
    pub const fn ground_level_delta(self) -> i32 {
        1
    }
}

/// Vanilla's `Holder<StructureProcessorList>` on single pool elements.
///
#[derive(Debug, Clone)]
pub enum ProcessorList {
    /// Direct empty processor list: `{ "processors": [] }`.
    Empty,
    /// Direct processor list payload.
    Direct(Vec<StructureProcessorKind>),
    /// Registry-backed processor list, e.g. `minecraft:street_savanna`.
    Registry(Identifier),
}

impl PartialEq for ProcessorList {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Empty, Self::Empty) => true,
            (Self::Registry(left), Self::Registry(right)) => left == right,
            _ => false,
        }
    }
}

/// A pool element — one possible piece that can be drawn from a template pool.
#[derive(Debug, Clone)]
pub enum PoolElement {
    /// Single structure template piece.
    Single {
        /// Template location (e.g., `minecraft:village/plains/houses/small_house_1`).
        location: Identifier,
        /// Processors applied during block placement.
        processors: ProcessorList,
        /// Vertical placement mode.
        projection: Projection,
    },
    /// Legacy single piece (same as Single but uses legacy jigsaw processing).
    LegacySingle {
        /// Template location.
        location: Identifier,
        /// Processors applied during block placement.
        processors: ProcessorList,
        /// Vertical placement mode.
        projection: Projection,
    },
    /// Empty placeholder element — signals no piece should be placed.
    Empty,
    /// A placed feature (not a structure template).
    Feature {
        /// Feature identifier.
        feature: Identifier,
        /// Vertical placement mode.
        projection: Projection,
    },
    /// A list of elements placed as a group.
    List {
        /// Sub-elements.
        elements: Vec<PoolElement>,
        /// Vertical placement mode.
        projection: Projection,
    },
}

impl PoolElement {
    /// Returns the projection mode, or `Rigid` for empty elements.
    #[must_use]
    pub fn projection(&self) -> Projection {
        match self {
            Self::Single { projection, .. }
            | Self::LegacySingle { projection, .. }
            | Self::Feature { projection, .. }
            | Self::List { projection, .. } => *projection,
            Self::Empty => Projection::Rigid,
        }
    }

    /// Returns true if this is an empty placeholder element.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
}

/// A template pool — a collection of weighted pool elements.
///
/// Vanilla's `StructureTemplatePool`.
#[derive(Debug, Clone)]
pub struct TemplatePoolData {
    /// Registry key (e.g., `minecraft:village/plains/town_centers`).
    pub key: Identifier,
    /// Fallback pool used when the main pool is exhausted.
    pub fallback: Identifier,
    /// Weighted elements. Each entry is (element, weight).
    pub elements: Vec<(PoolElement, i32)>,
}

fn extract_template_data_from_nbt_bytes(
    bytes: &[u8],
    key: &Identifier,
) -> Result<TemplateData, String> {
    let mut decoder = flate2::read::GzDecoder::new(bytes);
    let mut data = Vec::new();
    decoder
        .read_to_end(&mut data)
        .map_err(|err| format!("failed to decompress template {key}: {err}"))?;

    let nbt = simdnbt::borrow::read(&mut std::io::Cursor::new(&data))
        .map_err(|err| format!("failed to parse template {key}: {err}"))?;
    let simdnbt::borrow::Nbt::Some(root) = nbt else {
        return Err(format!("template {key} is empty"));
    };
    let compound = root.as_compound();

    let size_list = compound
        .list("size")
        .ok_or_else(|| format!("template {key} is missing size"))?;
    let size_ints = size_list
        .ints()
        .ok_or_else(|| format!("template {key} has non-int size list"))?;
    if size_ints.len() < 3 {
        return Err(format!("template {key} size list has fewer than 3 entries"));
    }
    let size = [size_ints[0], size_ints[1], size_ints[2]];

    let palette = if let Some(palette) = compound.list("palette").and_then(|list| list.compounds())
    {
        Some(palette)
    } else if let Some(palettes) = compound.list("palettes").and_then(|list| list.lists()) {
        match palettes.into_iter().next() {
            Some(first_palette) => Some(
                first_palette
                    .compounds()
                    .ok_or_else(|| format!("template {key} palettes[0] is not a compound list"))?,
            ),
            None => None,
        }
    } else {
        None
    };

    let Some(palette) = palette else {
        return Ok(TemplateData {
            size,
            jigsaws: Vec::new(),
        });
    };

    let palette_len = palette.len();
    let mut jigsaw_indices = Vec::new();
    for (index, entry) in palette.into_iter().enumerate() {
        let Some(name) = entry.string("Name") else {
            continue;
        };
        if name.to_str() != "minecraft:jigsaw" {
            continue;
        }
        let Some(orientation) = entry
            .compound("Properties")
            .and_then(|properties| properties.string("orientation"))
            .map(|orientation| orientation.to_str().to_string())
        else {
            return Err(format!(
                "jigsaw block state in template {key} is missing orientation"
            ));
        };
        jigsaw_indices.push((index, orientation));
    }

    let blocks = compound
        .list("blocks")
        .ok_or_else(|| format!("template {key} is missing blocks"))?
        .compounds()
        .ok_or_else(|| format!("template {key} has non-compound blocks list"))?;

    if jigsaw_indices.is_empty() {
        return Ok(TemplateData {
            size,
            jigsaws: Vec::new(),
        });
    }

    let mut jigsaws = Vec::new();
    for block in blocks {
        let state = block
            .int("state")
            .ok_or_else(|| format!("block in template {key} is missing state"))?;
        if state < 0 {
            return Err(format!(
                "block in template {key} has negative state {state}"
            ));
        }
        let state = usize::try_from(state)
            .map_err(|_| format!("block state {state} in template {key} does not fit usize"))?;
        if state >= palette_len {
            return Err(format!(
                "block state {state} in template {key} is outside palette length {palette_len}"
            ));
        }
        let Some((_, orientation)) = jigsaw_indices.iter().find(|(index, _)| *index == state)
        else {
            continue;
        };

        let pos_list = block
            .list("pos")
            .ok_or_else(|| format!("jigsaw block in template {key} is missing pos"))?
            .ints()
            .ok_or_else(|| format!("jigsaw block in template {key} has non-int pos list"))?;
        if pos_list.len() < 3 {
            return Err(format!(
                "jigsaw block in template {key} has fewer than 3 pos entries"
            ));
        }

        let nbt_data = block
            .compound("nbt")
            .ok_or_else(|| format!("jigsaw block in template {key} is missing nbt"))?;
        let get_str = |field: &str| -> Result<String, String> {
            nbt_data
                .string(field)
                .map(|value| value.to_str().to_string())
                .ok_or_else(|| format!("jigsaw block in template {key} is missing {field}"))
        };

        let orientation = JigsawOrientation::parse(orientation).ok_or_else(|| {
            format!("template {key} has unknown jigsaw orientation {orientation}")
        })?;
        let joint = match get_str("joint")?.as_str() {
            "aligned" => JointType::Aligned,
            "rollable" => JointType::Rollable,
            other => return Err(format!("template {key} has unknown jigsaw joint {other}")),
        };

        jigsaws.push(JigsawBlock {
            pos: [pos_list[0], pos_list[1], pos_list[2]],
            orientation,
            name: get_str("name")?
                .parse()
                .map_err(|err| format!("template {key} jigsaw name: {err}"))?,
            target: get_str("target")?
                .parse()
                .map_err(|err| format!("template {key} jigsaw target: {err}"))?,
            pool: get_str("pool")?
                .parse()
                .map_err(|err| format!("template {key} jigsaw pool: {err}"))?,
            joint,
            final_state: get_str("final_state")?,
            selection_priority: nbt_data.int("selection_priority").unwrap_or(0),
            placement_priority: nbt_data.int("placement_priority").unwrap_or(0),
        });
    }

    jigsaws.sort_by(|a, b| {
        a.pos[1]
            .cmp(&b.pos[1])
            .then(a.pos[0].cmp(&b.pos[0]))
            .then(a.pos[2].cmp(&b.pos[2]))
    });

    Ok(TemplateData { size, jigsaws })
}

/// Loads jigsaw template summaries from embedded compressed structure template NBT.
pub fn load_template_data_from_nbt_keys(
    keys: &[Identifier],
    nbt_bytes: fn(&Identifier) -> Option<&'static [u8]>,
) -> Vec<(Identifier, TemplateData)> {
    keys.iter()
        .map(|key| {
            let bytes = nbt_bytes(key)
                .unwrap_or_else(|| panic!("missing embedded structure template NBT for {key}"));
            let data = extract_template_data_from_nbt_bytes(bytes, key)
                .unwrap_or_else(|err| panic!("{err}"));
            (key.clone(), data)
        })
        .collect()
}
