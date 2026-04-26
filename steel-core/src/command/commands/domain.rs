//! Handler for the "domain" command.

use crate::command::arguments::domain::DomainArgument;
use crate::command::commands::{CommandHandlerBuilder, CommandHandlerDyn, argument};
use crate::command::context::CommandContext;
use crate::command::error::CommandError;
use text_components::TextComponent;

/// Handler for switching to another configured domain.
#[must_use]
pub fn command_handler() -> impl CommandHandlerDyn {
    CommandHandlerBuilder::new(
        &["domain"],
        "Switches to another configured domain.",
        "minecraft:command.domain",
    )
    .then(argument("domain", DomainArgument).executes(
        |((), domain): ((), String), context: &mut CommandContext| -> Result<(), CommandError> {
            let player = context
                .sender
                .get_player()
                .cloned()
                .ok_or(CommandError::InvalidRequirement)?;
            let server = context.server.clone();
            let domain_for_task = domain.clone();
            let player_name = player.gameprofile.name.clone();
            tokio::spawn(async move {
                if let Err(error) = server.switch_player_domain(player, domain_for_task).await {
                    log::error!("Failed to switch {player_name} domain: {error}");
                }
            });

            context.sender.send_message(&TextComponent::plain(format!(
                "Switching to domain {domain}"
            )));
            Ok(())
        },
    ))
}
