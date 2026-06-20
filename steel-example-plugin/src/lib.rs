//! Example plugin implementation for SteelMC.

use std::sync::OnceLock;
use steel_plugin_api::hook::{
    BehaviorInitAction, CommandInitAction, HookApi, InteractionResult, PluginBlockBehavior,
    PluginBlockBehaviorRef, PluginCommandHandler, PluginCommandHandlerRef,
    PluginCommandNodeBufferRef, PluginCommandNodeBufferDyn,
    PluginCommandRootChildrenRef, PluginCommandRootChildrenDyn,
    PluginCommandSenderRef, PluginCommandSenderDyn,
    PluginItemBehavior, PluginItemBehaviorRef,
    PluginWorld, PluginWorldRef, ServerStopAction,
};
use steel_plugin_api::{InitResult, PluginInitContext};
use steel_utils::Identifier;

static HOOK_API: OnceLock<HookApi> = OnceLock::new();

extern "C" fn on_server_stop(_args: &ServerStopAction) {
    log::info!("Stop received in plugin");
}

struct CustomFlintAndSteelBehavior {
    old: PluginItemBehaviorRef,
}

impl PluginItemBehavior for CustomFlintAndSteelBehavior {
    extern "C" fn get_original(&self) -> PluginItemBehaviorRef {
        self.old.clone()
    }

    extern "C" fn use_on(
        &self,
        world: PluginWorldRef,
        player_id: i32,
        hand: steel_utils::types::InteractionHand,
        x: i32,
        y: i32,
        z: i32,
        face: u8,
        hit_x: f64,
        hit_y: f64,
        hit_z: f64,
    ) -> InteractionResult {
        world.play_sound(
            steel_plugin_api::AbiStr::from("minecraft:entity.ghast.scream"),
            x as f64 + 0.5,
            y as f64 + 0.5,
            z as f64 + 0.5,
            1.0,
            1.0,
            -1,
        );
        self.get_original()
            .use_on(world, player_id, hand, x, y, z, face, hit_x, hit_y, hit_z)
    }
}

extern "C" fn fn_custom_flint_and_steel_behavior(
    old: PluginItemBehaviorRef,
) -> PluginItemBehaviorRef {
    let behavior = stabby::sync::Arc::new(CustomFlintAndSteelBehavior { old });
    PluginItemBehaviorRef(behavior.into())
}

struct CustomCycleBehavior {
    old: PluginBlockBehaviorRef,
}

impl PluginBlockBehavior for CustomCycleBehavior {
    extern "C" fn get_original(&self) -> PluginBlockBehaviorRef {
        self.old.clone()
    }
    extern "C" fn use_item_on(
        &self,
        state: steel_utils::BlockStateId,
        world: PluginWorldRef,
        player_id: i32,
        hand: steel_utils::types::InteractionHand,
        x: i32,
        y: i32,
        z: i32,
        face: u8,
        hit_x: f64,
        hit_y: f64,
        hit_z: f64,
    ) -> InteractionResult {
        let res = self.get_original().use_item_on(
            state, world, player_id, hand, x, y, z, face, hit_x, hit_y, hit_z,
        );
        if hand == steel_utils::types::InteractionHand::OffHand {
            return res;
        }
        let api = HOOK_API.get().expect("HookApi not initialized");

        // Get the current block state at the clicked position
        let current_state = world.get_block_state(x, y, z);

        // Get their default state IDs
        let iron_state = api
            .registry
            .get_block_state_id(&Identifier::new_static("minecraft", "iron_block"));
        let gold_state = api
            .registry
            .get_block_state_id(&Identifier::new_static("minecraft", "gold_block"));
        let diamond_state = api
            .registry
            .get_block_state_id(&Identifier::new_static("minecraft", "diamond_block"));
        let redstone_state = api
            .registry
            .get_block_state_id(&Identifier::new_static("minecraft", "redstone_block"));

        if iron_state == 0 || gold_state == 0 || diamond_state == 0 || redstone_state == 0 {
            return res;
        }

        // Determine if the clicked block is one of them and cycle accordingly:
        // Iron -> Gold -> Diamond -> Redstone -> Iron
        let next_state = if current_state == iron_state {
            Some(gold_state)
        } else if current_state == gold_state {
            Some(diamond_state)
        } else if current_state == diamond_state {
            Some(redstone_state)
        } else if current_state == redstone_state {
            Some(iron_state)
        } else {
            None
        };

        if let Some(next) = next_state {
            log::info!(
                "Right-clicked target block at ({}, {}, {}). Cycling state from {} to {} using hand {:?}",
                x,
                y,
                z,
                current_state,
                next,
                hand
            );
            world.set_block_state(x, y, z, next);
            InteractionResult::Success
        } else {
            res
        }
    }
}

extern "C" fn fn_custom_cycle_behavior(old: PluginBlockBehaviorRef) -> PluginBlockBehaviorRef {
    let behavior = stabby::sync::Arc::new(CustomCycleBehavior { old });
    PluginBlockBehaviorRef(behavior.into())
}

extern "C" fn on_behavior_init(args: &BehaviorInitAction) {
    args.register_block_behavior(
        &Identifier::new_static("minecraft", "iron_block"),
        fn_custom_cycle_behavior,
    );
    args.register_block_behavior(
        &Identifier::new_static("minecraft", "gold_block"),
        fn_custom_cycle_behavior,
    );
    args.register_block_behavior(
        &Identifier::new_static("minecraft", "diamond_block"),
        fn_custom_cycle_behavior,
    );
    args.register_block_behavior(
        &Identifier::new_static("minecraft", "redstone_block"),
        fn_custom_cycle_behavior,
    );
    args.register_item_behavior(
        &Identifier::new_static("minecraft", "flint_and_steel"),
        fn_custom_flint_and_steel_behavior,
    );
}

struct ExampleCommand;

impl PluginCommandHandler for ExampleCommand {
    extern "C" fn name(&self) -> steel_plugin_api::AbiStr<'static> {
        steel_plugin_api::AbiStr::from("steel_plugin_test")
    }

    extern "C" fn description(&self) -> steel_plugin_api::AbiStr<'static> {
        steel_plugin_api::AbiStr::from("A cool test command registered by a plugin")
    }

    extern "C" fn permission(&self) -> steel_plugin_api::AbiStr<'static> {
        steel_plugin_api::AbiStr::from("steel.plugin.test")
    }

    extern "C" fn usage(
        &self,
        buffer: PluginCommandNodeBufferRef<'_>,
        root_children: PluginCommandRootChildrenRef<'_>,
    ) {
        let node_index = buffer.add_literal(self.name(), false);
        root_children.push(node_index);

        let arg_index = buffer.add_argument(steel_plugin_api::AbiStr::from("<arguments>"), true);
        buffer.link_child(node_index, arg_index);
    }

    extern "C" fn execute(
        &self,
        sender: PluginCommandSenderRef<'_>,
        args_ptr: *const steel_plugin_api::AbiStr<'static>,
        args_len: usize,
    ) {
        sender.send_message(steel_plugin_api::AbiStr::from(
            "Hello from the Example Plugin Command!",
        ));

        // Inspect the sender using the new plugin APIs
        let sender_type = sender.sender_type();
        let sender_name = sender.name();
        sender.send_message(steel_plugin_api::AbiStr::from(
            format!("Sender type: {:?}", sender_type).as_str(),
        ));
        sender.send_message(steel_plugin_api::AbiStr::from(
            format!("Sender name: {}", sender_name.as_str()).as_str(),
        ));

        if sender_type == steel_plugin_api::types::CommandSenderType::Player {
            let player_id = sender.player_id();
            let bytes = sender.uuid();
            sender.send_message(steel_plugin_api::AbiStr::from(
                format!("Player ID: {}", player_id).as_str(),
            ));
            sender.send_message(steel_plugin_api::AbiStr::from(format!("Player UUID: {:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                bytes.bytes[0], bytes.bytes[1], bytes.bytes[2], bytes.bytes[3],
                bytes.bytes[4], bytes.bytes[5],
                bytes.bytes[6], bytes.bytes[7],
                bytes.bytes[8], bytes.bytes[9],
                bytes.bytes[10], bytes.bytes[11], bytes.bytes[12], bytes.bytes[13], bytes.bytes[14], bytes.bytes[15]
            ).as_str()));
        }

        // SAFETY: The host guarantees args_ptr points to args_len valid AbiStr.
        let args = unsafe { std::slice::from_raw_parts(args_ptr, args_len) };
        if args.is_empty() {
            sender.send_message(steel_plugin_api::AbiStr::from("No arguments provided."));
        } else {
            // If the first argument is "kick", attempt to kick the player
            if args[0].as_str() == "kick" {
                if sender_type == steel_plugin_api::types::CommandSenderType::Player {
                    sender.send_message(steel_plugin_api::AbiStr::from("Kicking you now..."));
                    sender.kick(steel_plugin_api::AbiStr::from("Kicked by plugin command"));
                } else {
                    sender.send_message(steel_plugin_api::AbiStr::from(
                        "Cannot kick non-player sender.",
                    ));
                }
                return;
            }

            for (i, arg) in args.iter().enumerate() {
                sender.send_message(steel_plugin_api::AbiStr::from(
                    format!("Argument {}: {}", i + 1, arg.as_str()).as_str(),
                ));
            }
        }
    }
}

extern "C" fn on_command_init(args: &CommandInitAction) {
    let handler = PluginCommandHandlerRef(stabby::sync::Arc::new(ExampleCommand).into());
    args.register_handler(handler);
}

fn init(ctx: PluginInitContext) -> InitResult {
    let _ = HOOK_API.set(ctx.hook_api.clone());

    ctx.hook_api
        .add_action::<ServerStopAction>(on_server_stop)
        .register();

    ctx.hook_api
        .add_action::<BehaviorInitAction>(on_behavior_init)
        .register();

    ctx.hook_api
        .add_action::<CommandInitAction>(on_command_init)
        .register();

    log::info!("Hello plugin: {}", ctx.hook_api.plugin_id.as_str());
    InitResult::Ok
}

steel_plugin_api::declare_plugin!(init);
