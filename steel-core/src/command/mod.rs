//! This module contains everything needed for commands (e.g., parsing, execution, and sender handling).
pub mod arguments;
pub mod commands;
pub mod context;
pub mod error;
pub mod sender;

use std::sync::Arc;

use steel_protocol::packets::game::{
    ArgumentType, ArgumentStringTypeBehavior, CCommandSuggestions, CCommands, CommandNode,
    CommandNodeInfo, SuggestionEntry,
};
use steel_plugin_api::hook::{
    CommandInitAction, PluginCommandHandlerRef, PluginCommandHandlerDyn,
    PluginCommandNodeBuffer, PluginCommandRootChildren,
    PluginCommandSender,
};
use steel_plugin_loader::hook::get_host_registry;
use text_components::{Modifier, TextComponent, format::Color};

use crate::command::commands::CommandHandlerDyn;
use crate::command::context::CommandContext;
use crate::command::error::CommandError;
use crate::command::sender::CommandSender;
use crate::entity::Entity;
use crate::player::Player;
use crate::server::Server;

/// A struct that parses and dispatches commands to their appropriate handlers.
#[derive(Default)]
pub struct CommandDispatcher {
    /// A map of command names to their handlers.
    handlers: scc::HashMap<&'static str, Arc<dyn CommandHandlerDyn + Send + Sync>>,
}

impl CommandDispatcher {
    /// Creates a new command dispatcher with vanilla handlers.
    #[must_use]
    pub fn new() -> Self {
        let dispatcher = CommandDispatcher::new_empty();
        dispatcher.register(commands::clear::command_handler());
        dispatcher.register(commands::domain::command_handler());
        dispatcher.register(commands::enchant::command_handler());
        dispatcher.register(commands::execute::command_handler());
        dispatcher.register(commands::fly::command_handler());
        dispatcher.register(commands::gamemode::command_handler());
        dispatcher.register(commands::gamerule::command_handler());
        dispatcher.register(commands::kill::command_handler());
        dispatcher.register(commands::list::command_handler());
        dispatcher.register(commands::locate::command_handler());
        dispatcher.register(commands::give::command_handler());
        dispatcher.register(commands::seed::command_handler());
        dispatcher.register(commands::setworldspawn::command_handler());
        dispatcher.register(commands::stop::command_handler());
        dispatcher.register(commands::summon::command_handler());
        dispatcher.register(commands::tellraw::command_handler());
        dispatcher.register(commands::tick::command_handler());
        dispatcher.register(commands::time::command_handler());
        dispatcher.register(commands::tp::command_handler());
        dispatcher.register(commands::weather::command_handler());
        dispatcher.register(commands::difficulty::command_handler());
        dispatcher.register(commands::steel::command_handler());
        dispatcher.register(commands::xp::command_handler());

        // Fire command initialization hook
        let host_dispatcher: &'static HostCommandDispatcher = Box::leak(Box::new(HostCommandDispatcher {
            dispatcher: &raw const dispatcher,
        }));
        let dispatcher_dynptr = host_dispatcher.into();

        let hook_registry = get_host_registry();
        let action_args = CommandInitAction {
            dispatcher: dispatcher_dynptr,
        };
        hook_registry.do_action_typed(&action_args);

        dispatcher
    }

    /// Creates a new command dispatcher with no handlers.
    #[must_use]
    pub fn new_empty() -> Self {
        CommandDispatcher {
            handlers: scc::HashMap::new(),
        }
    }

    /// Executes a command.
    pub fn handle_command(&self, sender: CommandSender, command: String, server: &Arc<Server>) {
        let mut context = CommandContext::new(sender.clone(), server.clone());

        if let Err(error) = Self::split_command(&command)
            .and_then(|(command, args)| self.execute(command, &args, &mut context, server))
        {
            let text = match error {
                CommandError::InvalidConsumption(s) => {
                    log::error!(
                        "Error while parsing command \"{command}\": {s:?} was consumed, but couldn't be parsed"
                    );
                    TextComponent::const_plain("Internal error (See logs for details)")
                }
                CommandError::InvalidRequirement => {
                    log::error!(
                        "Error while parsing command \"{command}\": a requirement that was expected was not met."
                    );
                    TextComponent::const_plain("Internal error (See logs for details)")
                }
                CommandError::PermissionDenied => {
                    log::warn!("Permission denied for command \"{command}\"");
                    TextComponent::const_plain(
                        "I'm sorry, but you do not have permission to perform this command. Please contact the server administrator if you believe this is an error.",
                    )
                }
                CommandError::CommandFailed(text_component) => *text_component,
            };

            // TODO: Use vanilla error messages
            sender.send_message(&text.color(Color::Red));
        }
    }

    /// Executes a command.
    fn execute(
        &self,
        command: &str,
        command_args: &[&str],
        context: &mut CommandContext,
        server: &Arc<Server>,
    ) -> Result<(), CommandError> {
        let Some(handler) = self.handlers.read_sync(command, |_, v| v.clone()) else {
            return Err(CommandError::CommandFailed(Box::new(
                format!("Command {command} does not exist").into(),
            )));
        };

        // TODO: Implement permission checking logic here
        // if let CommandSender::Player(&player) = sender
        //     && !server.player_has_permission(player, &handler.permission)
        // {
        //     return Err(PermissionDenied);
        // };

        handler.execute(command_args, context, server)
    }

    /// Parses a command string into its components.
    fn split_command(command: &str) -> Result<(&str, Box<[&str]>), CommandError> {
        let command = command.trim();
        if command.is_empty() {
            return Err(CommandError::CommandFailed(Box::new(
                TextComponent::const_plain("Empty Command"),
            )));
        }

        let Some((command, command_args)) = command.split_once(' ') else {
            return Ok((command, Box::new([])));
        };

        // TODO: Implement proper command parsing (handling quotes, escapes, etc.)
        // This will likely be handled by a String argument parser that consumes quoted strings.

        Ok((command, command_args.split_whitespace().collect()))
    }

    /// Generates the `CCommands` packet, containing the usage information of every registered commands.
    pub fn get_commands(&self) -> CCommands {
        let mut nodes = Vec::with_capacity(self.handlers.len() + 1);
        nodes.push(CommandNode::new_root());

        let mut root_children = Vec::with_capacity(self.handlers.len());
        self.handlers.iter_sync(|command, handler| {
            if *command != handler.names()[0] {
                return true;
            }

            // TODO: Implement permission checking logic here

            handler.usage(&mut nodes, &mut root_children);
            true
        });
        nodes[0].set_children(root_children);

        CCommands {
            root_index: 0,
            nodes,
        }
    }

    /// Registers a command handler.
    pub fn register(&self, handler: impl CommandHandlerDyn + Send + Sync + 'static) {
        let handler = Arc::new(handler);
        for name in handler.names() {
            if let Err((name, _)) = self.handlers.insert_sync(name, handler.clone()) {
                log::warn!("Command {name} is already registered");
            }
        }
    }

    /// Unregisters a command handler.
    pub fn unregister(&self, names: &[&'static str]) {
        for name in names {
            self.handlers.remove_sync(name);
        }
    }

    /// Handles a command suggestion request from a player.
    pub fn handle_player_suggestions(
        &self,
        player: &Arc<Player>,
        id: i32,
        command: &str,
        server: Arc<Server>,
    ) {
        let (suggestions, start, length) =
            self.handle_suggestions(CommandSender::Player(Arc::clone(player)), command, server);
        player.send_packet(CCommandSuggestions::new(id, start, length, suggestions));
    }

    /// Handles a command suggestion request from a player.
    pub fn handle_suggestions(
        &self,
        sender: CommandSender,
        command: &str,
        server: Arc<Server>,
    ) -> (Vec<SuggestionEntry>, i32, i32) {
        // Remove leading slash if present
        let command = command.strip_prefix('/').unwrap_or(command);

        // Split into parts, preserving trailing space as empty string
        let mut parts: Vec<&str> = command.split(' ').collect();

        // Remove empty parts from the middle but keep trailing empty if command ends with space
        let has_trailing_space = command.ends_with(' ');
        parts.retain(|s| !s.is_empty());
        if has_trailing_space {
            parts.push("");
        }

        // If empty or typing command name, suggest command names
        if parts.is_empty() || (parts.len() == 1 && !has_trailing_space) {
            let prefix = parts.first().copied().unwrap_or("");
            let suggestions = self.get_command_suggestions(prefix);
            // Start position is 1 (after the slash)
            return (suggestions, 1, prefix.len() as i32);
        }

        // Get the command handler
        let command_name = parts[0];
        let Some(handler) = self.handlers.read_sync(command_name, |_, v| v.clone()) else {
            // Unknown command - no suggestions
            return (vec![], 0, 0);
        };

        // Calculate where args start (after "command_name ")
        let args_start_pos = command_name.len() + 1; // +1 for space

        // Get the args (everything after command name)
        let args = &parts[1..];

        // Create context for suggestion
        let mut context = CommandContext::new(sender, server);

        // Get suggestions from handler
        if let Some(result) = handler.suggest(args, args_start_pos, &mut context) {
            // Adjust start position to account for leading slash
            (result.suggestions, result.start + 1, result.length)
        } else {
            // No suggestions
            (vec![], 0, 0)
        }
    }

    /// Gets command name suggestions matching the given prefix.
    fn get_command_suggestions(&self, prefix: &str) -> Vec<SuggestionEntry> {
        let mut suggestions = Vec::new();
        let prefix_lower = prefix.to_lowercase();

        self.handlers.iter_sync(|name, handler| {
            // Only include primary command names (not aliases)
            if *name == handler.names()[0] && name.to_lowercase().starts_with(&prefix_lower) {
                suggestions.push(SuggestionEntry::new(*name));
            }
            true
        });

        suggestions.sort_by(|a, b| a.text.cmp(&b.text));
        suggestions
    }
}

struct PluginCommandHandler {
    names: &'static [&'static str],
    handler: PluginCommandHandlerRef,
}

struct HostCommandNodeBuffer<'a> {
    buffer: std::cell::RefCell<&'a mut Vec<CommandNode>>,
}

impl<'a> PluginCommandNodeBuffer for HostCommandNodeBuffer<'a> {
    extern "C" fn add_literal(&self, name: steel_plugin_api::AbiStr<'_>, is_executable: bool) -> i32 {
        let mut buffer = self.buffer.borrow_mut();
        let index = buffer.len() as i32;
        buffer.push(CommandNode::new_literal(
            if is_executable {
                CommandNodeInfo::new_executable()
            } else {
                CommandNodeInfo::new(vec![])
            },
            name.to_string(),
        ));
        index
    }

    extern "C" fn add_argument(&self, name: steel_plugin_api::AbiStr<'_>, is_executable: bool) -> i32 {
        let mut buffer = self.buffer.borrow_mut();
        let index = buffer.len() as i32;
        buffer.push(CommandNode::new_argument(
            if is_executable {
                CommandNodeInfo::new_executable()
            } else {
                CommandNodeInfo::new(vec![])
            },
            name.to_string(),
            (
                ArgumentType::String {
                    behavior: ArgumentStringTypeBehavior::GreedyPhrase,
                },
                None,
            ),
        ));
        index
    }

    extern "C" fn link_child(&self, parent_index: i32, child_index: i32) {
        let mut buffer = self.buffer.borrow_mut();
        if let Some(parent) = buffer.get_mut(parent_index as usize) {
            match parent {
                CommandNode::Root { children } => {
                    children.push(child_index);
                }
                CommandNode::Literal { children, .. } => {
                    children.push(child_index);
                }
                CommandNode::Argument { children, .. } => {
                    children.push(child_index);
                }
            }
        }
    }
}

struct HostCommandRootChildren<'a> {
    root_children: std::cell::RefCell<&'a mut Vec<i32>>,
}

impl<'a> PluginCommandRootChildren for HostCommandRootChildren<'a> {
    extern "C" fn push(&self, node_index: i32) {
        self.root_children.borrow_mut().push(node_index);
    }
}

struct HostCommandSender<'a> {
    sender: &'a CommandSender,
}

impl<'a> PluginCommandSender for HostCommandSender<'a> {
    extern "C" fn send_message(&self, message: steel_plugin_api::AbiStr<'_>) {
        self.sender.send_message(&TextComponent::from(message.as_str().to_string()));
    }

    extern "C" fn sender_type(&self) -> steel_plugin_api::types::CommandSenderType {
        match self.sender {
            CommandSender::Player(_) => steel_plugin_api::types::CommandSenderType::Player,
            CommandSender::Console => steel_plugin_api::types::CommandSenderType::Console,
            CommandSender::Rcon => steel_plugin_api::types::CommandSenderType::Rcon,
        }
    }

    extern "C" fn name(&self) -> steel_plugin_api::AbiString {
        steel_plugin_api::AbiString::from(self.sender.to_string())
    }

    extern "C" fn player_id(&self) -> i32 {
        if let CommandSender::Player(player) = self.sender {
            player.id()
        } else {
            -1
        }
    }

    extern "C" fn uuid(&self) -> steel_plugin_api::types::PluginUuid {
        if let CommandSender::Player(player) = self.sender {
            steel_plugin_api::types::PluginUuid {
                bytes: player.uuid().into_bytes(),
            }
        } else {
            steel_plugin_api::types::PluginUuid { bytes: [0; 16] }
        }
    }

    extern "C" fn kick(&self, reason: steel_plugin_api::AbiStr<'_>) -> bool {
        if let CommandSender::Player(player) = self.sender {
            player.disconnect(reason.as_str().to_string());
            true
        } else {
            false
        }
    }
}

impl CommandHandlerDyn for PluginCommandHandler {
    fn names(&self) -> &'static [&'static str] {
        self.names
    }

    fn description(&self) -> &'static str {
        self.handler.description().as_str()
    }

    fn permission(&self) -> &'static str {
        self.handler.permission().as_str()
    }

    fn execute(
        &self,
        command_args: &[&str],
        context: &mut CommandContext,
        _server: &Arc<Server>,
    ) -> Result<(), CommandError> {
        let abi_args: Vec<steel_plugin_api::AbiStr<'_>> = command_args
            .iter()
            .map(|s| steel_plugin_api::AbiStr::from(*s))
            .collect();

        let host_sender = HostCommandSender {
            sender: &context.sender,
        };

        // SAFETY: The FFI call is synchronous, so the reference to the stack-allocated structure is guaranteed to remain valid.
        let host_sender_static_ref = unsafe {
            std::mem::transmute::<&HostCommandSender<'_>, &'static HostCommandSender<'static>>(&host_sender)
        };
        let host_sender_ref = host_sender_static_ref.into();

        let args_ptr = abi_args.as_ptr().cast::<steel_plugin_api::AbiStr<'static>>();
        self.handler.execute(host_sender_ref, args_ptr, abi_args.len());
        Ok(())
    }

    fn usage(&self, buffer: &mut Vec<CommandNode>, root_children: &mut Vec<i32>) {
        let node_buffer = HostCommandNodeBuffer {
            buffer: std::cell::RefCell::new(buffer),
        };
        let node_root_children = HostCommandRootChildren {
            root_children: std::cell::RefCell::new(root_children),
        };

        // SAFETY: The FFI call is synchronous, so the reference to the stack-allocated structures is guaranteed to remain valid.
        let node_buffer_static_ref = unsafe {
            std::mem::transmute::<&HostCommandNodeBuffer<'_>, &'static HostCommandNodeBuffer<'static>>(&node_buffer)
        };
        let node_root_children_static_ref = unsafe {
            std::mem::transmute::<&HostCommandRootChildren<'_>, &'static HostCommandRootChildren<'static>>(&node_root_children)
        };

        let node_buffer_ref = node_buffer_static_ref.into();
        let node_root_children_ref = node_root_children_static_ref.into();

        self.handler.usage(node_buffer_ref, node_root_children_ref);
    }
}

/// Host-side wrapper that implements the plugin API's `PluginCommandDispatcher` trait.
pub struct HostCommandDispatcher {
    /// Opaque pointer to the underlying command dispatcher.
    pub dispatcher: *const CommandDispatcher,
}

// SAFETY: HostCommandDispatcher is only used synchronously during initialization.
unsafe impl Send for HostCommandDispatcher {}
// SAFETY: HostCommandDispatcher is only used synchronously during initialization.
unsafe impl Sync for HostCommandDispatcher {}

impl steel_plugin_api::hook::PluginCommandDispatcher for HostCommandDispatcher {
    extern "C" fn register_command(
        &self,
        handler: PluginCommandHandlerRef,
    ) {
        // SAFETY: The host guarantees that `dispatcher` points to a valid `CommandDispatcher`.
        let dispatcher = unsafe { &*self.dispatcher };

        let name = handler.name().as_str();
        let leaked_names: &'static [&'static str] = Box::leak(Box::new([name]));

        let host_handler = PluginCommandHandler {
            names: leaked_names,
            handler,
        };
        dispatcher.register(host_handler);
    }
}
