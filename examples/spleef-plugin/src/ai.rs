//! Small local controller: walk toward rivals on intact snow and dig near their feet.
use crate::{api, map};
use steel_plugin_api::{BlockPos, BotInput, PlayerControls, PlayerInfo, Position, Server};
fn distance(a: Position, b: Position) -> f64 {
    (a.x - b.x).powi(2) + (a.z - b.z).powi(2)
}
fn cell(p: Position) -> (i32, i32) {
    (p.x.floor() as i32, p.z.floor() as i32)
}
pub fn step(
    server: &Server<'_>,
    world: &str,
    actor: &PlayerInfo,
    rivals: &[&PlayerInfo],
    tick: u64,
) -> Result<(), String> {
    let pos = actor.position;
    let target = rivals
        .iter()
        .filter(|p| p.id != actor.id)
        .min_by(|a, b| distance(pos, a.position).total_cmp(&distance(pos, b.position)))
        .map_or(pos, |p| p.position);
    let snow = map::snow()?;
    let (x, z) = cell(pos);
    let mut candidates = Vec::new();
    for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let (cx, cz) = (x + dx, z + dz);
        if cx.abs() >= map::RADIUS - 1 || cz.abs() >= map::RADIUS - 1 {
            continue;
        }
        if api!(server.block(
            world.into(),
            BlockPos {
                x: cx,
                y: map::FLOOR,
                z: cz
            }
        ))? == snow
        {
            candidates.push(Position {
                x: f64::from(cx) + 0.5,
                y: pos.y,
                z: f64::from(cz) + 0.5,
            });
        }
    }
    let waypoint = candidates
        .into_iter()
        .min_by(|a, b| distance(*a, target).total_cmp(&distance(*b, target)))
        .unwrap_or(pos);
    let dx = waypoint.x - pos.x;
    let dz = waypoint.z - pos.z;
    let walking = dx * dx + dz * dz > 0.08;
    let yaw = (-dx.atan2(dz).to_degrees()) as f32;
    let mut dig = None;
    if tick.is_multiple_of(5) {
        let mut best = f64::INFINITY;
        for cx in x - 3..=x + 3 {
            for cz in z - 3..=z + 3 {
                if !can_dig((cx, cz), cell(pos), cell(waypoint)) {
                    continue;
                }
                let block = BlockPos {
                    x: cx,
                    y: map::FLOOR,
                    z: cz,
                };
                let candidate = Position {
                    x: f64::from(cx) + 0.5,
                    y: pos.y,
                    z: f64::from(cz) + 0.5,
                };
                let score = distance(candidate, target);
                if distance(pos, candidate) <= 12.0
                    && score < best
                    && api!(server.block(world.into(), block))? == snow
                {
                    best = score;
                    dig = Some(block);
                }
            }
        }
    }
    api!(server.control_bot(
        actor.id,
        BotInput {
            yaw,
            pitch: 45.0,
            movement: PlayerControls {
                forward: walking,
                ..Default::default()
            },
            dig: dig.into()
        }
    ))
}
fn can_dig(candidate: (i32, i32), current: (i32, i32), next: (i32, i32)) -> bool {
    candidate != current && candidate != next
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn controller_preserves_its_support_and_next_step() {
        assert!(!can_dig((0, 0), (0, 0), (1, 0)));
        assert!(!can_dig((1, 0), (0, 0), (1, 0)));
        assert!(can_dig((2, 0), (0, 0), (1, 0)));
    }
}
