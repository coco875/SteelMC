use steel_utils::BlockStateId;
use steel_utils::types::InteractionHand;
use crate::types::{InteractionResult, CommandSenderType, PluginUuid};
use crate::hook::Action;

#[stabby::stabby(checked)]
pub trait PluginWorld {
    extern "C" fn get_block_state(&self, x: i32, y: i32, z: i32) -> u16;
    extern "C" fn set_block_state(&self, x: i32, y: i32, z: i32, state: u16);
    extern "C" fn get_raw_world_ptr(&self) -> *const std::ffi::c_void {
        core::ptr::null()
    }
    extern "C" fn play_sound(
        &self,
        _sound_name: crate::AbiStr<'_>,
        _x: f64,
        _y: f64,
        _z: f64,
        _volume: f32,
        _pitch: f32,
        _exclude_player_id: i32,
    ) {}
}

pub type PluginWorldRef = stabby::dynptr!(&'static dyn PluginWorld);

impl PluginWorld for PluginWorldRef {
    extern "C" fn get_block_state(&self, x: i32, y: i32, z: i32) -> u16 {
        PluginWorldDyn::get_block_state(self, x, y, z)
    }

    extern "C" fn set_block_state(&self, x: i32, y: i32, z: i32, state: u16) {
        PluginWorldDyn::set_block_state(self, x, y, z, state)
    }

    extern "C" fn get_raw_world_ptr(&self) -> *const std::ffi::c_void {
        PluginWorldDyn::get_raw_world_ptr(self)
    }

    extern "C" fn play_sound(
        &self,
        sound_name: crate::AbiStr<'_>,
        x: f64,
        y: f64,
        z: f64,
        volume: f32,
        pitch: f32,
        exclude_player_id: i32,
    ) {
        PluginWorldDyn::play_sound(
            self,
            sound_name,
            x,
            y,
            z,
            volume,
            pitch,
            exclude_player_id,
        );
    }
}

#[stabby::stabby]
pub trait PluginBlockBehavior {
    extern "C" fn get_original(&self) -> PluginBlockBehaviorRef;
    extern "C" fn use_item_on(
        &self,
        state: BlockStateId,
        world: PluginWorldRef,
        player_id: i32,
        hand: InteractionHand,
        x: i32,
        y: i32,
        z: i32,
        face: u8,
        hit_x: f64,
        hit_y: f64,
        hit_z: f64,
    ) -> InteractionResult {
        self.get_original().use_item_on(
            state, world, player_id, hand, x, y, z, face, hit_x, hit_y, hit_z,
        )
    }

    extern "C" fn use_without_item(
        &self,
        state: BlockStateId,
        world: PluginWorldRef,
        player_id: i32,
        x: i32,
        y: i32,
        z: i32,
        face: u8,
        hit_x: f64,
        hit_y: f64,
        hit_z: f64,
    ) -> InteractionResult {
        self.get_original()
            .use_without_item(state, world, player_id, x, y, z, face, hit_x, hit_y, hit_z)
    }

    extern "C" fn step_on(
        &self,
        state: BlockStateId,
        world: PluginWorldRef,
        x: i32,
        y: i32,
        z: i32,
        entity_id: i32,
    ) {
        self.get_original()
            .step_on(state, world, x, y, z, entity_id)
    }

    extern "C" fn tick(&self, state: BlockStateId, world: PluginWorldRef, x: i32, y: i32, z: i32) {
        self.get_original().tick(state, world, x, y, z);
    }

    extern "C" fn random_tick(
        &self,
        state: BlockStateId,
        world: PluginWorldRef,
        x: i32,
        y: i32,
        z: i32,
    ) {
        self.get_original().random_tick(state, world, x, y, z);
    }

    extern "C" fn on_place(
        &self,
        state: BlockStateId,
        world: PluginWorldRef,
        x: i32,
        y: i32,
        z: i32,
        old_state: u16,
        moved_by_piston: bool,
    ) {
        self.get_original()
            .on_place(state, world, x, y, z, old_state, moved_by_piston);
    }

    extern "C" fn player_will_destroy(
        &self,
        state: BlockStateId,
        world: PluginWorldRef,
        x: i32,
        y: i32,
        z: i32,
        player_id: i32,
    ) -> BlockStateId {
        self.get_original()
            .player_will_destroy(state, world, x, y, z, player_id)
    }
}

#[stabby::stabby]
#[derive(Clone)]
pub struct PluginBlockBehaviorRef(pub stabby::dynptr!(stabby::sync::Arc<dyn PluginBlockBehavior>));

// SAFETY: The plugin guarantees that the behavior implementation is thread-safe.
unsafe impl Send for PluginBlockBehaviorRef {}
unsafe impl Sync for PluginBlockBehaviorRef {}

impl PluginBlockBehavior for PluginBlockBehaviorRef {
    extern "C" fn get_original(&self) -> PluginBlockBehaviorRef {
        PluginBlockBehaviorDyn::get_original(&self.0)
    }
    extern "C" fn use_item_on(
        &self,
        state: BlockStateId,
        world: PluginWorldRef,
        player_id: i32,
        hand: InteractionHand,
        x: i32,
        y: i32,
        z: i32,
        face: u8,
        hit_x: f64,
        hit_y: f64,
        hit_z: f64,
    ) -> InteractionResult {
        PluginBlockBehaviorDyn::use_item_on(
            &self.0, state, world, player_id, hand, x, y, z, face, hit_x, hit_y, hit_z,
        )
    }

    extern "C" fn use_without_item(
        &self,
        state: BlockStateId,
        world: PluginWorldRef,
        player_id: i32,
        x: i32,
        y: i32,
        z: i32,
        face: u8,
        hit_x: f64,
        hit_y: f64,
        hit_z: f64,
    ) -> InteractionResult {
        PluginBlockBehaviorDyn::use_without_item(
            &self.0, state, world, player_id, x, y, z, face, hit_x, hit_y, hit_z,
        )
    }

    extern "C" fn step_on(
        &self,
        state: BlockStateId,
        world: PluginWorldRef,
        x: i32,
        y: i32,
        z: i32,
        entity_id: i32,
    ) {
        PluginBlockBehaviorDyn::step_on(&self.0, state, world, x, y, z, entity_id)
    }

    extern "C" fn tick(&self, state: BlockStateId, world: PluginWorldRef, x: i32, y: i32, z: i32) {
        PluginBlockBehaviorDyn::tick(&self.0, state, world, x, y, z)
    }

    extern "C" fn random_tick(
        &self,
        state: BlockStateId,
        world: PluginWorldRef,
        x: i32,
        y: i32,
        z: i32,
    ) {
        PluginBlockBehaviorDyn::random_tick(&self.0, state, world, x, y, z)
    }

    extern "C" fn on_place(
        &self,
        state: BlockStateId,
        world: PluginWorldRef,
        x: i32,
        y: i32,
        z: i32,
        old_state: u16,
        moved_by_piston: bool,
    ) {
        PluginBlockBehaviorDyn::on_place(&self.0, state, world, x, y, z, old_state, moved_by_piston)
    }

    extern "C" fn player_will_destroy(
        &self,
        state: BlockStateId,
        world: PluginWorldRef,
        x: i32,
        y: i32,
        z: i32,
        player_id: i32,
    ) -> BlockStateId {
        PluginBlockBehaviorDyn::player_will_destroy(&self.0, state, world, x, y, z, player_id)
    }
}

#[stabby::stabby]
pub trait PluginItemBehavior {
    extern "C" fn get_original(&self) -> PluginItemBehaviorRef;
    extern "C" fn use_on(
        &self,
        world: PluginWorldRef,
        player_id: i32,
        hand: InteractionHand,
        x: i32,
        y: i32,
        z: i32,
        face: u8,
        hit_x: f64,
        hit_y: f64,
        hit_z: f64,
    ) -> InteractionResult {
        self.get_original().use_on(
            world, player_id, hand, x, y, z, face, hit_x, hit_y, hit_z,
        )
    }

    extern "C" fn use_item(
        &self,
        world: PluginWorldRef,
        player_id: i32,
        hand: InteractionHand,
    ) -> InteractionResult {
        self.get_original().use_item(world, player_id, hand)
    }
}

#[stabby::stabby]
#[derive(Clone)]
pub struct PluginItemBehaviorRef(pub stabby::dynptr!(stabby::sync::Arc<dyn PluginItemBehavior>));

// SAFETY: The plugin guarantees that the behavior implementation is thread-safe.
unsafe impl Send for PluginItemBehaviorRef {}
unsafe impl Sync for PluginItemBehaviorRef {}

impl PluginItemBehavior for PluginItemBehaviorRef {
    extern "C" fn get_original(&self) -> PluginItemBehaviorRef {
        PluginItemBehaviorDyn::get_original(&self.0)
    }
    extern "C" fn use_on(
        &self,
        world: PluginWorldRef,
        player_id: i32,
        hand: InteractionHand,
        x: i32,
        y: i32,
        z: i32,
        face: u8,
        hit_x: f64,
        hit_y: f64,
        hit_z: f64,
    ) -> InteractionResult {
        PluginItemBehaviorDyn::use_on(
            &self.0, world, player_id, hand, x, y, z, face, hit_x, hit_y, hit_z,
        )
    }

    extern "C" fn use_item(
        &self,
        world: PluginWorldRef,
        player_id: i32,
        hand: InteractionHand,
    ) -> InteractionResult {
        PluginItemBehaviorDyn::use_item(&self.0, world, player_id, hand)
    }
}

#[stabby::stabby(checked)]
pub trait PluginCommandNodeBuffer {
    extern "C" fn add_literal(&self, name: crate::AbiStr<'_>, is_executable: bool) -> i32;
    extern "C" fn add_argument(&self, name: crate::AbiStr<'_>, is_executable: bool) -> i32;
    extern "C" fn link_child(&self, parent_index: i32, child_index: i32);
}

pub type PluginCommandNodeBufferRef<'a> = stabby::dynptr!(&'a dyn PluginCommandNodeBuffer);

#[stabby::stabby(checked)]
pub trait PluginCommandRootChildren {
    extern "C" fn push(&self, node_index: i32);
}

pub type PluginCommandRootChildrenRef<'a> = stabby::dynptr!(&'a dyn PluginCommandRootChildren);

#[stabby::stabby(checked)]
pub trait PluginCommandSender {
    extern "C" fn send_message(&self, message: crate::AbiStr<'_>);
    extern "C" fn sender_type(&self) -> CommandSenderType;
    extern "C" fn name(&self) -> crate::AbiString;
    extern "C" fn player_id(&self) -> i32;
    extern "C" fn uuid(&self) -> PluginUuid;
    extern "C" fn kick(&self, reason: crate::AbiStr<'_>) -> bool;
}

pub type PluginCommandSenderRef<'a> = stabby::dynptr!(&'a dyn PluginCommandSender);

#[stabby::stabby]
pub trait PluginCommandHandler {
    extern "C" fn name(&self) -> crate::AbiStr<'static>;
    
    extern "C" fn description(&self) -> crate::AbiStr<'static> {
        crate::AbiStr::from("")
    }

    extern "C" fn permission(&self) -> crate::AbiStr<'static> {
        crate::AbiStr::from("")
    }

    extern "C" fn usage(
        &self,
        _buffer: PluginCommandNodeBufferRef<'_>,
        _root_children: PluginCommandRootChildrenRef<'_>,
    ) {}

    extern "C" fn execute(
        &self,
        sender: PluginCommandSenderRef<'_>,
        args_ptr: *const crate::AbiStr<'static>,
        args_len: usize,
    );
}

#[stabby::stabby]
#[derive(Clone)]
pub struct PluginCommandHandlerRef(pub stabby::dynptr!(stabby::sync::Arc<dyn PluginCommandHandler>));

// SAFETY: Host-side command handler is Send + Sync.
unsafe impl Send for PluginCommandHandlerRef {}
unsafe impl Sync for PluginCommandHandlerRef {}

impl std::ops::Deref for PluginCommandHandlerRef {
    type Target = stabby::dynptr!(stabby::sync::Arc<dyn PluginCommandHandler>);
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[stabby::stabby]
pub struct FnCommandHandler {
    pub name: crate::AbiStr<'static>,
    pub description: crate::AbiStr<'static>,
    pub permission: crate::AbiStr<'static>,
    pub usage: crate::AbiStr<'static>,
    pub callback: extern "C" fn(
        sender: PluginCommandSenderRef<'static>,
        args_ptr: *const crate::AbiStr<'static>,
        args_len: usize,
    ),
}

impl PluginCommandHandler for FnCommandHandler {
    extern "C" fn name(&self) -> crate::AbiStr<'static> {
        self.name
    }

    extern "C" fn description(&self) -> crate::AbiStr<'static> {
        self.description
    }

    extern "C" fn permission(&self) -> crate::AbiStr<'static> {
        self.permission
    }

    extern "C" fn usage(
        &self,
        buffer: PluginCommandNodeBufferRef<'_>,
        root_children: PluginCommandRootChildrenRef<'_>,
    ) {
        let has_args = !self.usage.as_str().is_empty();
        let node_index = buffer.add_literal(self.name, !has_args);
        root_children.push(node_index);

        if has_args {
            let arg_index = buffer.add_argument(self.usage, true);
            buffer.link_child(node_index, arg_index);
        }
    }

    extern "C" fn execute(
        &self,
        sender: PluginCommandSenderRef<'_>,
        args_ptr: *const crate::AbiStr<'static>,
        args_len: usize,
    ) {
        // SAFETY: The execution is synchronous, so the sender reference is guaranteed to remain valid.
        let sender_static = unsafe {
            std::mem::transmute::<PluginCommandSenderRef<'_>, PluginCommandSenderRef<'static>>(sender)
        };
        (self.callback)(sender_static, args_ptr, args_len);
    }
}

#[stabby::stabby(checked)]
pub trait PluginCommandDispatcher {
    extern "C" fn register_command(&self, handler: PluginCommandHandlerRef);
}

/// Action fired when the command dispatcher is initializing, allowing plugins to register commands directly.
#[stabby::stabby]
pub struct CommandInitAction {
    pub dispatcher: stabby::dynptr!(&'static dyn PluginCommandDispatcher),
}

impl Action for CommandInitAction {
    const TAG: &'static str = "steel:command_init";
}

impl CommandInitAction {
    /// Helper to register a command.
    pub fn register_command(
        &self,
        name: &'static str,
        callback: extern "C" fn(sender: PluginCommandSenderRef<'static>, args_ptr: *const crate::AbiStr<'static>, args_len: usize),
    ) {
        self.register_command_with_info(name, "", "", "", callback);
    }

    /// Helper to register a command with details.
    pub fn register_command_with_info(
        &self,
        name: &'static str,
        description: &'static str,
        permission: &'static str,
        usage: &'static str,
        callback: extern "C" fn(sender: PluginCommandSenderRef<'static>, args_ptr: *const crate::AbiStr<'static>, args_len: usize),
    ) {
        let handler = FnCommandHandler {
            name: crate::AbiStr::from(name),
            description: crate::AbiStr::from(description),
            permission: crate::AbiStr::from(permission),
            usage: crate::AbiStr::from(usage),
            callback,
        };
        let handler_ref = PluginCommandHandlerRef(stabby::sync::Arc::new(handler).into());
        self.dispatcher.register_command(handler_ref);
    }

    /// Helper to register a custom command handler directly.
    pub fn register_handler(&self, handler: PluginCommandHandlerRef) {
        self.dispatcher.register_command(handler);
    }
}

#[stabby::stabby(checked)]
pub trait PluginRegistryApiVtable {
    extern "C" fn get_block_state_id(
        &self,
        namespace: crate::AbiStr<'_>,
        path: crate::AbiStr<'_>,
    ) -> u16;
}

#[stabby::stabby]
#[derive(Copy, Clone)]
pub struct PluginRegistryApiVtableRef(pub stabby::dynptr!(&'static dyn PluginRegistryApiVtable));

// SAFETY: HostRegistryApi is Send + Sync, making the vtable reference thread-safe.
unsafe impl Send for PluginRegistryApiVtableRef {}
unsafe impl Sync for PluginRegistryApiVtableRef {}

impl std::ops::Deref for PluginRegistryApiVtableRef {
    type Target = stabby::dynptr!(&'static dyn PluginRegistryApiVtable);
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[stabby::stabby]
#[derive(Copy, Clone)]
pub struct PluginRegistryApi {
    pub api: PluginRegistryApiVtableRef,
}

impl PluginRegistryApi {
    pub fn get_block_state_id(&self, key: &steel_utils::Identifier) -> u16 {
        let namespace = crate::AbiStr::from(key.namespace.as_ref());
        let path = crate::AbiStr::from(key.path.as_ref());
        self.api.get_block_state_id(namespace, path)
    }
}

#[stabby::stabby]
pub struct RegistryInitAction {
    pub registry: *mut std::ffi::c_void,
}

impl Action for RegistryInitAction {
    const TAG: &'static str = "steel:registry_init";
}

#[stabby::stabby(checked)]
pub trait PluginBehaviorRegistry {
    extern "C" fn register_block_behavior(
        &self,
        namespace: crate::AbiStr<'_>,
        path: crate::AbiStr<'_>,
        behavior: extern "C" fn(old: PluginBlockBehaviorRef) -> PluginBlockBehaviorRef,
    ) -> PluginBlockBehaviorRef;

    extern "C" fn register_item_behavior(
        &self,
        namespace: crate::AbiStr<'_>,
        path: crate::AbiStr<'_>,
        behavior: extern "C" fn(old: PluginItemBehaviorRef) -> PluginItemBehaviorRef,
    ) -> PluginItemBehaviorRef;
}

/// Action fired when behavior registries are initializing, allowing plugins to register behavior directly.
#[stabby::stabby]
pub struct BehaviorInitAction {
    pub registry: stabby::dynptr!(&'static dyn PluginBehaviorRegistry),
}

impl Action for BehaviorInitAction {
    const TAG: &'static str = "steel:behavior_init";
}

impl BehaviorInitAction {
    /// Helper to register block behavior using standard Identifier.
    pub fn register_block_behavior(
        &self,
        key: &steel_utils::Identifier,
        behavior: extern "C" fn(old: PluginBlockBehaviorRef) -> PluginBlockBehaviorRef,
    ) -> PluginBlockBehaviorRef {
        let namespace = crate::AbiStr::from(key.namespace.as_ref());
        let path = crate::AbiStr::from(key.path.as_ref());
        self.registry
            .register_block_behavior(namespace, path, behavior)
    }

    /// Helper to register item behavior using standard Identifier.
    pub fn register_item_behavior(
        &self,
        key: &steel_utils::Identifier,
        behavior: extern "C" fn(old: PluginItemBehaviorRef) -> PluginItemBehaviorRef,
    ) -> PluginItemBehaviorRef {
        let namespace = crate::AbiStr::from(key.namespace.as_ref());
        let path = crate::AbiStr::from(key.path.as_ref());
        self.registry
            .register_item_behavior(namespace, path, behavior)
    }
}
