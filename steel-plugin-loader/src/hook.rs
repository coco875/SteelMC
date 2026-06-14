//! Host-side implementation of the WordPress-style action and filter hook system.
//!
//! This module provides the `HostHookRegistry` which maintains registration and execution
//! of plugin hooks, performing topological sorting to enforce dependency constraints.

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::slice;
use std::sync::OnceLock;
use steel_plugin_api::AbiString;
use steel_plugin_api::hook::{
    Action, ApplyFiltersArgs, DoActionArgs, Filter, HandlerOrdering, HookApiVtable, LogLevel,
    OrderingConstraint, RawActionCallback, RawFilterCallback, RegisterActionArgs,
    RegisterFilterArgs,
};
use steel_utils::locks::SyncRwLock;

pub use steel_plugin_api::hook::HookApiVtableRef;

/// A host-side handler for an Action hook.
#[derive(Clone)]
pub struct HostActionHandler {
    /// The unique identifier of the plugin that registered this handler.
    pub plugin_id: String,
    /// The FFI-stable raw callback to invoke.
    pub callback: RawActionCallback,
    /// The ordering constraints relative to other plugins.
    pub orderings: Vec<HandlerOrdering>,
}

/// A host-side handler for a Filter hook.
#[derive(Clone)]
pub struct HostFilterHandler {
    /// The unique identifier of the plugin that registered this handler.
    pub plugin_id: String,
    /// The FFI-stable raw callback to invoke.
    pub callback: RawFilterCallback,
    /// The ordering constraints relative to other plugins.
    pub orderings: Vec<HandlerOrdering>,
}

/// Host-side registry for action and filter hooks.
///
/// Keeps track of registered action and filter callbacks and executes them
/// while respecting any specified execution order constraints.
pub struct HostHookRegistry {
    actions: SyncRwLock<FxHashMap<String, Vec<HostActionHandler>>>,
    filters: SyncRwLock<FxHashMap<String, Vec<HostFilterHandler>>>,
}

impl Default for HostHookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl HostHookRegistry {
    /// Creates a new empty `HostHookRegistry`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            actions: SyncRwLock::new(FxHashMap::default()),
            filters: SyncRwLock::new(FxHashMap::default()),
        }
    }

    /// Helper to trigger a typed action on the registry.
    pub fn do_action_typed<A: Action>(&self, args: &A) {
        self.do_action(DoActionArgs {
            tag: AbiString::from(A::TAG),
            args: (args as *const A).cast::<u8>(),
        });
    }

    /// Helper to apply typed filters on the registry.
    pub fn apply_filters_typed<F: Filter>(&self, value: &mut F) {
        self.apply_filters(ApplyFiltersArgs {
            tag: AbiString::from(F::TAG),
            value: (value as *mut F).cast::<u8>(),
        });
    }
}

impl HookApiVtable for HostHookRegistry {
    extern "C" fn register_action(&self, args: RegisterActionArgs) {
        let plugin_id = args.plugin_id.as_str().to_string();
        let tag = args.tag.as_str().to_string();

        let orderings = if args.orderings_ptr.is_null() || args.orderings_len == 0 {
            Vec::new()
        } else {
            // SAFETY: The caller guarantees that `orderings_ptr` points to a valid allocation
            // of `orderings_len` elements, and that the elements are initialized and valid for the lifetime.
            unsafe {
                slice::from_raw_parts(args.orderings_ptr, args.orderings_len as usize).to_vec()
            }
        };

        let handler = HostActionHandler {
            plugin_id,
            callback: args.callback,
            orderings,
        };

        let mut actions = self.actions.write();
        let handlers = actions.entry(tag).or_default();
        handlers.push(handler);
        sort_handlers(handlers);
    }

    extern "C" fn do_action(&self, args: DoActionArgs) {
        let tag = args.tag.as_str();
        let handlers = {
            let actions = self.actions.read();
            actions.get(tag).cloned()
        };

        let Some(handlers) = handlers else {
            return;
        };

        for h in handlers {
            (h.callback)(args.args);
        }
    }

    extern "C" fn register_filter(&self, args: RegisterFilterArgs) {
        let plugin_id = args.plugin_id.as_str().to_string();
        let tag = args.tag.as_str().to_string();

        let orderings = if args.orderings_ptr.is_null() || args.orderings_len == 0 {
            Vec::new()
        } else {
            // SAFETY: The caller guarantees that `orderings_ptr` points to a valid allocation
            // of `orderings_len` elements, and that the elements are initialized and valid for the lifetime.
            unsafe {
                slice::from_raw_parts(args.orderings_ptr, args.orderings_len as usize).to_vec()
            }
        };

        let handler = HostFilterHandler {
            plugin_id,
            callback: args.callback,
            orderings,
        };

        let mut filters = self.filters.write();
        let handlers = filters.entry(tag).or_default();
        handlers.push(handler);
        sort_handlers(handlers);
    }

    extern "C" fn apply_filters(&self, args: ApplyFiltersArgs) {
        let tag = args.tag.as_str();
        let handlers = {
            let filters = self.filters.read();
            filters.get(tag).cloned()
        };

        let Some(handlers) = handlers else {
            return;
        };

        for h in handlers {
            (h.callback)(args.value);
        }
    }

    extern "C" fn log(&self, level: LogLevel, target: AbiString, message: AbiString) {
        let level = match level {
            LogLevel::Error => log::Level::Error,
            LogLevel::Warn => log::Level::Warn,
            LogLevel::Info => log::Level::Info,
            LogLevel::Debug => log::Level::Debug,
            LogLevel::Trace => log::Level::Trace,
        };
        log::log!(target: target.as_str(), level, "{}", message.as_str());
    }
}

trait OrderedHandler {
    fn plugin_id(&self) -> &str;
    fn orderings(&self) -> &[HandlerOrdering];
}

impl OrderedHandler for HostActionHandler {
    fn plugin_id(&self) -> &str {
        &self.plugin_id
    }
    fn orderings(&self) -> &[HandlerOrdering] {
        &self.orderings
    }
}

impl OrderedHandler for HostFilterHandler {
    fn plugin_id(&self) -> &str {
        &self.plugin_id
    }
    fn orderings(&self) -> &[HandlerOrdering] {
        &self.orderings
    }
}

fn sort_handlers<H: OrderedHandler>(handlers: &mut [H]) {
    let mut nodes = FxHashSet::default();
    let mut edges = FxHashSet::default();

    for h in &*handlers {
        nodes.insert(h.plugin_id().to_string());
        for ord in h.orderings() {
            let target = ord.plugin_id.as_str().to_string();
            nodes.insert(target.clone());
            match ord.constraint {
                OrderingConstraint::Before => {
                    edges.insert((h.plugin_id().to_string(), target));
                }
                OrderingConstraint::After => {
                    edges.insert((target, h.plugin_id().to_string()));
                }
            }
        }
    }

    let sorted_plugins = topological_sort(&nodes, &edges);
    let plugin_order: FxHashMap<String, usize> = sorted_plugins
        .into_iter()
        .enumerate()
        .map(|(i, name)| (name, i))
        .collect();

    handlers.sort_by_key(|h| {
        plugin_order
            .get(h.plugin_id())
            .copied()
            .unwrap_or(usize::MAX)
    });
}

fn topological_sort(nodes: &FxHashSet<String>, edges: &FxHashSet<(String, String)>) -> Vec<String> {
    let mut in_degree: FxHashMap<String, usize> = FxHashMap::default();
    let mut adj: FxHashMap<String, Vec<String>> = FxHashMap::default();

    for node in nodes {
        in_degree.insert(node.clone(), 0);
    }

    for (from, to) in edges {
        if nodes.contains(from) && nodes.contains(to) {
            adj.entry(from.clone()).or_default().push(to.clone());
            *in_degree.entry(to.clone()).or_default() += 1;
        }
    }

    let mut queue = VecDeque::new();
    for node in nodes {
        if in_degree[node] == 0 {
            queue.push_back(node.clone());
        }
    }

    let mut order = Vec::new();
    while let Some(u) = queue.pop_front() {
        order.push(u.clone());
        if let Some(neighbors) = adj.get(&u) {
            for v in neighbors {
                if let Some(deg) = in_degree.get_mut(v) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(v.clone());
                    }
                }
            }
        }
    }

    if order.len() < nodes.len() {
        for node in nodes {
            if !order.contains(node) {
                order.push(node.clone());
            }
        }
    }

    order
}

static HOST_HOOK_REGISTRY: OnceLock<HostHookRegistry> = OnceLock::new();

/// Retrieves the global static instance of the `HostHookRegistry`.
pub fn get_host_registry() -> &'static HostHookRegistry {
    HOST_HOOK_REGISTRY.get_or_init(HostHookRegistry::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use steel_plugin_api::AbiString;
    use steel_plugin_api::hook::{Action, Filter, HookApi};

    #[stabby::stabby]
    struct PlayerJoinAction {
        pub username: AbiString,
        pub score: i32,
    }

    impl Action for PlayerJoinAction {
        const TAG: &'static str = "steel:player_join";
    }

    #[stabby::stabby]
    struct FormatChatFilter {
        pub message: AbiString,
    }

    impl Filter for FormatChatFilter {
        const TAG: &'static str = "steel:format_chat";
    }

    extern "C" fn on_player_join(args: &PlayerJoinAction) {
        assert_eq!(args.username.as_str(), "Alice");
    }

    extern "C" fn on_format_chat_1(value: &mut FormatChatFilter) {
        let mut msg = value.message.as_str().to_string();
        msg.push_str(" world");
        value.message = AbiString::from(msg);
    }

    extern "C" fn on_format_chat_2(value: &mut FormatChatFilter) {
        let mut msg = value.message.as_str().to_string();
        msg.push('!');
        value.message = AbiString::from(msg);
    }

    #[expect(clippy::cast_ptr_alignment, reason = "FFI type-erased pointer cast")]
    extern "C" fn raw_action_callback(args: *const u8) {
        // SAFETY: The caller guarantees that `args` points to a valid `PlayerJoinAction`.
        let join_args = unsafe { &*args.cast::<PlayerJoinAction>() };
        assert_eq!(join_args.username.as_str(), "Alice");
    }

    #[expect(clippy::cast_ptr_alignment, reason = "FFI type-erased pointer cast")]
    extern "C" fn raw_filter_callback(value: *mut u8) {
        // SAFETY: The caller guarantees that `value` points to a valid `FormatChatFilter`.
        let chat_value = unsafe { &mut *value.cast::<FormatChatFilter>() };
        let mut msg = chat_value.message.as_str().to_string();
        msg.push_str(" raw");
        chat_value.message = AbiString::from(msg);
    }

    #[test]
    fn test_generic_actions_and_filters() {
        let registry = get_host_registry();
        let stable_api = HookApiVtableRef(registry.into());

        let api_a = HookApi {
            host_api: stable_api,
            plugin_id: AbiString::from("plugin_a"),
        };
        let api_b = HookApi {
            host_api: stable_api,
            plugin_id: AbiString::from("plugin_b"),
        };

        // Test Action Hook
        api_a
            .add_action::<PlayerJoinAction>(on_player_join)
            .register();
        let join_args = PlayerJoinAction {
            username: AbiString::from("Alice"),
            score: 100,
        };
        api_a.do_action(&join_args);

        // Test Filter Hook (with ordering)
        // Register filter_2 on plugin_b to run after plugin_a (filter_1)
        api_b
            .add_filter::<FormatChatFilter>(on_format_chat_2)
            .after("plugin_a")
            .register();

        api_a
            .add_filter::<FormatChatFilter>(on_format_chat_1)
            .register();

        let mut chat_value = FormatChatFilter {
            message: AbiString::from("hello"),
        };
        api_a.apply_filters(&mut chat_value);

        assert_eq!(chat_value.message.as_str(), "hello world!");

        // Test Raw Action and Filter Hook
        api_a
            .add_action_raw(raw_action_callback, "steel:player_join_raw")
            .register();
        api_a.do_action_raw("steel:player_join_raw", (&raw const join_args).cast::<u8>());

        api_a
            .add_filter_raw(raw_filter_callback, "steel:format_chat_raw")
            .register();
        let mut chat_value_raw = FormatChatFilter {
            message: AbiString::from("hello"),
        };
        api_a.apply_filters_raw(
            "steel:format_chat_raw",
            (&raw mut chat_value_raw).cast::<u8>(),
        );
        assert_eq!(chat_value_raw.message.as_str(), "hello raw");
    }
}
