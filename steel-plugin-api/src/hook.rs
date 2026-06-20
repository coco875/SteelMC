use crate::AbiString;
pub use crate::logging::LogLevel;

pub use crate::hook_types::{
    BehaviorInitAction, CommandInitAction, PluginBehaviorRegistry, PluginBlockBehavior,
    PluginBlockBehaviorRef, PluginCommandDispatcher, PluginItemBehavior, PluginItemBehaviorRef,
    PluginRegistryApi, PluginRegistryApiVtable, PluginRegistryApiVtableRef, PluginWorld,
    PluginWorldRef, RegistryInitAction, PluginCommandHandler,
    PluginCommandHandlerRef, FnCommandHandler, PluginCommandHandlerDyn,
    PluginCommandNodeBuffer, PluginCommandNodeBufferRef, PluginCommandNodeBufferDyn,
    PluginCommandRootChildren, PluginCommandRootChildrenRef, PluginCommandRootChildrenDyn,
    PluginCommandSender, PluginCommandSenderRef, PluginCommandSenderDyn,
};
pub use crate::types::InteractionResult;

/// An Action represents a point of execution where callbacks can run side-effects.
/// All Action types must be FFI-stable.
pub trait Action: stabby::abi::IStable + Sized {
    /// Namespaced action tag, e.g. `"steel:player_join"`.
    const TAG: &'static str;
}

/// A Filter represents a value that can be modified by callbacks in sequence.
/// All Filter types must be FFI-stable.
pub trait Filter: stabby::abi::IStable + Sized {
    /// Namespaced filter tag, e.g. `"steel:format_chat"`.
    const TAG: &'static str;
}

#[stabby::stabby]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderingConstraint {
    /// This handler must run before the named plugin's handler.
    Before,
    /// This handler must run after the named plugin's handler.
    After,
}

/// A single ordering constraint: (kind, target_plugin_id).
#[stabby::stabby]
#[derive(Debug, Clone)]
pub struct HandlerOrdering {
    pub constraint: OrderingConstraint,
    pub plugin_id: AbiString,
}

/// Type-erased action callback. Receives a read-only pointer to action arguments.
pub type RawActionCallback = extern "C" fn(args: *const u8);

/// Type-erased filter callback. Receives a mutable pointer to the value being filtered.
pub type RawFilterCallback = extern "C" fn(value: *mut u8);

#[stabby::stabby]
pub struct RegisterActionArgs {
    pub plugin_id: AbiString,
    pub tag: AbiString,
    pub callback: RawActionCallback,
    pub orderings_ptr: *const HandlerOrdering,
    pub orderings_len: u32,
}

#[stabby::stabby]
pub struct RegisterFilterArgs {
    pub plugin_id: AbiString,
    pub tag: AbiString,
    pub callback: RawFilterCallback,
    pub orderings_ptr: *const HandlerOrdering,
    pub orderings_len: u32,
}

#[stabby::stabby]
pub struct DoActionArgs {
    pub tag: AbiString,
    pub args: *const u8,
}

#[stabby::stabby]
pub struct ApplyFiltersArgs {
    pub tag: AbiString,
    pub value: *mut u8,
}

#[stabby::stabby(checked)]
pub trait HookApiVtable {
    extern "C" fn register_action(&self, args: RegisterActionArgs);
    extern "C" fn do_action(&self, args: DoActionArgs);
    extern "C" fn register_filter(&self, args: RegisterFilterArgs);
    extern "C" fn apply_filters(&self, args: ApplyFiltersArgs);
    extern "C" fn log(&self, level: LogLevel, target: crate::AbiString, message: crate::AbiString);
}

#[stabby::stabby]
#[derive(Copy, Clone)]
pub struct HookApiVtableRef(pub stabby::dynptr!(&'static dyn HookApiVtable));

// SAFETY: HostHookRegistry is Send + Sync, making the vtable reference thread-safe.
unsafe impl Send for HookApiVtableRef {}
unsafe impl Sync for HookApiVtableRef {}

impl std::ops::Deref for HookApiVtableRef {
    type Target = stabby::dynptr!(&'static dyn HookApiVtable);
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[stabby::stabby]
pub struct PluginInitContext {
    pub hook_api: HookApi,
}

/// Scoped API for action and filter hook registrations and executions.
#[stabby::stabby]
#[derive(Clone)]
pub struct HookApi {
    pub host_api: HookApiVtableRef,
    pub plugin_id: crate::AbiString,
    pub registry: PluginRegistryApi,
}

impl HookApi {
    /// Register an action callback for Action `A`.
    pub fn add_action<A: Action>(&self, callback: extern "C" fn(args: &A)) -> ActionBuilder {
        // SAFETY: `&A` and `*const u8` have identical ABI representation.
        let raw: RawActionCallback = unsafe { core::mem::transmute(callback) };
        self.add_action_raw(raw, A::TAG)
    }

    pub fn add_action_raw(
        &self,
        callback: extern "C" fn(args: *const u8),
        tag: &'static str,
    ) -> ActionBuilder {
        ActionBuilder {
            host_api: self.host_api,
            plugin_id: self.plugin_id.as_str().to_string(),
            callback,
            tag,
            orderings: Vec::new(),
        }
    }

    /// Trigger an action hook.
    pub fn do_action<A: Action>(&self, args: &A) {
        self.do_action_raw(A::TAG, (args as *const A).cast::<u8>())
    }

    /// Trigger an action hook with a raw pointer.
    pub fn do_action_raw(&self, tag: &str, args: *const u8) {
        self.host_api.do_action(DoActionArgs {
            tag: AbiString::from(tag),
            args,
        })
    }

    /// Register a filter callback for Filter `F`.
    pub fn add_filter<F: Filter>(&self, callback: extern "C" fn(value: &mut F)) -> FilterBuilder {
        // SAFETY: `&mut F` and `*mut u8` have identical ABI representation.
        let raw: RawFilterCallback = unsafe { core::mem::transmute(callback) };
        self.add_filter_raw(raw, F::TAG)
    }

    /// Register a filter callback with a raw pointer.
    pub fn add_filter_raw(&self, callback: RawFilterCallback, tag: &'static str) -> FilterBuilder {
        FilterBuilder {
            host_api: self.host_api,
            plugin_id: self.plugin_id.as_str().to_string(),
            callback,
            tag,
            orderings: Vec::new(),
        }
    }

    /// Apply filters to a value.
    pub fn apply_filters<F: Filter>(&self, value: &mut F) {
        self.apply_filters_raw(F::TAG, (value as *mut F).cast::<u8>())
    }

    /// Apply filters to a value with a raw pointer.
    pub fn apply_filters_raw(&self, tag: &str, value: *mut u8) {
        self.host_api.apply_filters(ApplyFiltersArgs {
            tag: AbiString::from(tag),
            value,
        })
    }

    /// Get the stable host api reference.
    pub fn host_api(&self) -> HookApiVtableRef {
        self.host_api
    }
}

pub struct ActionBuilder {
    host_api: HookApiVtableRef,
    plugin_id: String,
    callback: RawActionCallback,
    tag: &'static str,
    orderings: Vec<HandlerOrdering>,
}

impl ActionBuilder {
    pub fn before(mut self, plugin_id: &str) -> Self {
        self.orderings.push(HandlerOrdering {
            constraint: OrderingConstraint::Before,
            plugin_id: AbiString::from(plugin_id),
        });
        self
    }

    pub fn after(mut self, plugin_id: &str) -> Self {
        self.orderings.push(HandlerOrdering {
            constraint: OrderingConstraint::After,
            plugin_id: AbiString::from(plugin_id),
        });
        self
    }

    pub fn register(self) {
        let ptr = if self.orderings.is_empty() {
            core::ptr::null()
        } else {
            self.orderings.as_ptr()
        };
        let len = self.orderings.len() as u32;

        self.host_api.register_action(RegisterActionArgs {
            plugin_id: AbiString::from(self.plugin_id.as_str()),
            tag: AbiString::from(self.tag),
            callback: self.callback,
            orderings_ptr: ptr,
            orderings_len: len,
        });
    }
}

pub struct FilterBuilder {
    host_api: HookApiVtableRef,
    plugin_id: String,
    callback: RawFilterCallback,
    tag: &'static str,
    orderings: Vec<HandlerOrdering>,
}

impl FilterBuilder {
    pub fn before(mut self, plugin_id: &str) -> Self {
        self.orderings.push(HandlerOrdering {
            constraint: OrderingConstraint::Before,
            plugin_id: AbiString::from(plugin_id),
        });
        self
    }

    pub fn after(mut self, plugin_id: &str) -> Self {
        self.orderings.push(HandlerOrdering {
            constraint: OrderingConstraint::After,
            plugin_id: AbiString::from(plugin_id),
        });
        self
    }

    pub fn register(self) {
        let ptr = if self.orderings.is_empty() {
            core::ptr::null()
        } else {
            self.orderings.as_ptr()
        };
        let len = self.orderings.len() as u32;

        self.host_api.register_filter(RegisterFilterArgs {
            plugin_id: AbiString::from(self.plugin_id.as_str()),
            tag: AbiString::from(self.tag),
            callback: self.callback,
            orderings_ptr: ptr,
            orderings_len: len,
        });
    }
}

/// Action fired on every game server tick.
#[stabby::stabby]
pub struct ServerTickAction {
    /// The current tick number.
    pub tick_count: u64,
}

impl Action for ServerTickAction {
    const TAG: &'static str = "steel:tick";
}

/// Action fired when the server is stopping.
#[stabby::stabby]
pub struct ServerStopAction {}

impl Action for ServerStopAction {
    const TAG: &'static str = "steel:stop";
}
