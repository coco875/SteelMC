//! Steel server commands: /steel tp <targets> <world>

use std::sync::Arc;

use steel_registry::blocks::block_state_ext::BlockStateExt;
use text_components::TextComponent;

use crate::command::arguments::block_pos::BlockPosArgument;
use crate::command::arguments::player::PlayerArgument;
use crate::command::arguments::world::WorldArgument;
use crate::command::commands::{CommandHandlerBuilder, CommandHandlerDyn, argument, literal};
use crate::command::context::CommandContext;
use crate::command::error::CommandError;
use crate::entity::SharedEntity;
use crate::player::Player;
use crate::portal::WorldChangeRequest;
use crate::world::World;
use steel_utils::BlockPos;

/// Handler for the "steel" command group.
#[must_use]
pub fn command_handler() -> impl CommandHandlerDyn {
    CommandHandlerBuilder::new(
        &["steel"],
        "Steel server commands.",
        "minecraft:command.steel",
    )
    .then(
        literal("tp").then(argument("targets", PlayerArgument::multiple()).then(
            argument("world", WorldArgument).executes(
                |(((), targets), world): (((), Vec<Arc<Player>>), Arc<World>),
                 context: &mut CommandContext|
                 -> Result<(), CommandError> {
                    let dim_name = &world.key;
                    let count = targets.len();

                    for target in &targets {
                        if target.is_domain_switching() {
                            return Err(CommandError::CommandFailed(Box::new(
                                TextComponent::plain(format!(
                                    "{} is already switching domains",
                                    target.gameprofile.name
                                )),
                            )));
                        }
                    }

                    for target in &targets {
                        let current_world = target.get_world();
                        if current_world.domain() == world.domain() {
                            context.server.queue_world_change(
                                target.clone() as SharedEntity,
                                WorldChangeRequest::WorldSpawn {
                                    target_world: world.clone(),
                                },
                            );
                        } else {
                            context
                                .server
                                .queue_domain_switch_to_world(target.clone(), world.clone())
                                .map_err(|error| {
                                    CommandError::CommandFailed(Box::new(TextComponent::plain(
                                        error,
                                    )))
                                })?;
                        }
                    }

                    let msg = if count == 1 {
                        format!(
                            "Teleporting {} to {}",
                            targets[0].gameprofile.name, dim_name
                        )
                    } else {
                        format!("Teleporting {count} players to {dim_name}")
                    };
                    context.sender.send_message(&TextComponent::from(msg));

                    Ok(())
                },
            ),
        )),
    )
    .then(
        literal("lightsource").executes(
            |(), context: &mut CommandContext| -> Result<(), CommandError> {
                let pos = BlockPos::containing(
                    context.position.x,
                    context.position.y,
                    context.position.z,
                );
                send_block_light_source_message(&context.world, pos, &context.sender);
                Ok(())
            },
        )
        .then(
            argument("pos", BlockPosArgument).executes(
                |(_, pos): ((), BlockPos), context: &mut CommandContext| -> Result<(), CommandError> {
                    send_block_light_source_message(&context.world, pos, &context.sender);
                    Ok(())
                },
            ),
        ),
    )
}

fn send_block_light_source_message(
    world: &Arc<World>,
    pos: BlockPos,
    sender: &crate::command::sender::CommandSender,
) {
    let block_light = world.block_light_at(pos);
    let vector = world.block_light_vector_at(pos);
    let source_pos = vector.source_position(pos);
    let source_state = world.get_block_state(source_pos);
    let source_name = source_state.get_block().key.to_string();

    let message = if block_light == 0 {
        format!(
            "Block light at {}: 0 (no block-light source recorded)",
            format_block_pos(pos)
        )
    } else if vector.is_zero() {
        format!(
            "Block light at {}: {} | source offset (0, 0, 0) | block {} at {}",
            format_block_pos(pos),
            block_light,
            source_name,
            format_block_pos(source_pos),
        )
    } else {
        format!(
            "Block light at {}: {} | source offset ({}, {}, {}) | block {} at {}",
            format_block_pos(pos),
            block_light,
            vector.dx,
            vector.dy,
            vector.dz,
            source_name,
            format_block_pos(source_pos),
        )
    };

    sender.send_message(&TextComponent::from(message));
}

fn format_block_pos(pos: BlockPos) -> String {
    format!("({}, {}, {})", pos.x(), pos.y(), pos.z())
}
