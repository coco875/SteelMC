use glam::IVec3;
use steel_utils::{Direction, Identifier, Rotation};

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

    pub const fn top_direction(self) -> Direction {
        match self {
            Self::DownEast | Self::UpEast => Direction::East,
            Self::DownNorth | Self::UpNorth => Direction::North,
            Self::DownSouth | Self::UpSouth => Direction::South,
            Self::DownWest | Self::UpWest => Direction::West,
            Self::WestUp | Self::EastUp | Self::NorthUp | Self::SouthUp => Direction::Up,
        }
    }

    pub const fn from_directions(front: Direction, top: Direction) -> Option<Self> {
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

    pub fn rotate(self, rotation: Rotation) -> Self {
        let front = rotation.rotate(self.front_direction());
        let top = rotation.rotate(self.top_direction());
        Self::from_directions(front, top).expect("rotated orientation should be valid")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointType {
    Rollable,
    Aligned,
}

#[derive(Debug, Clone)]
pub struct JigsawBlock {
    pub pos: [i32; 3],
    pub orientation: JigsawOrientation,
    pub name: Identifier,
    pub target: Identifier,
    pub pool: Identifier,
    pub joint: JointType,
    pub final_state: Identifier,
    pub selection_priority: i32,
    pub selection_priority_bucket: u8,
    pub placement_priority: i32,
}

const ROTATIONS: [Rotation; 4] = [
    Rotation::None,
    Rotation::Clockwise90,
    Rotation::Clockwise180,
    Rotation::CounterClockwise90,
];

impl JigsawBlock {
    pub fn rotated(block: &Self, rotation: Rotation) -> Self {
        let pos = rotation.transform_pos(IVec3::from(block.pos), IVec3::ZERO);
        Self {
            pos: [pos.x, pos.y, pos.z],
            orientation: block.orientation.rotate(rotation),
            name: block.name.clone(),
            target: block.target.clone(),
            pool: block.pool.clone(),
            joint: block.joint,
            final_state: block.final_state.clone(),
            selection_priority: block.selection_priority,
            selection_priority_bucket: block.selection_priority_bucket,
            placement_priority: block.placement_priority,
        }
    }
}

pub fn rotated_jigsaw_sets(jigsaws: &[JigsawBlock]) -> [Vec<JigsawBlock>; 4] {
    std::array::from_fn(|idx| {
        jigsaws
            .iter()
            .map(|jigsaw| JigsawBlock::rotated(jigsaw, ROTATIONS[idx]))
            .collect()
    })
}

pub fn selection_priorities_desc(jigsaws: &[JigsawBlock]) -> Vec<i32> {
    let mut priorities_desc = Vec::new();
    for jigsaw in jigsaws {
        if !priorities_desc.contains(&jigsaw.selection_priority) {
            priorities_desc.push(jigsaw.selection_priority);
        }
    }
    if priorities_desc.len() > 1 {
        priorities_desc.sort_unstable_by(|a, b| b.cmp(a));
    }
    priorities_desc
}

pub fn assign_selection_priority_buckets(
    jigsaws: &mut [JigsawBlock],
    priorities_desc: &[i32],
) {
    for jigsaw in jigsaws.iter_mut() {
        jigsaw.selection_priority_bucket = priorities_desc
            .iter()
            .position(|&priority| priority == jigsaw.selection_priority)
            .unwrap_or(priorities_desc.len()) as u8;
    }
}
