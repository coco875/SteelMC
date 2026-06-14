/// Convenience macro for declaring Actions.
#[macro_export]
macro_rules! action {
    ($ty:ty, $tag:literal) => {
        impl $crate::hook::Action for $ty {
            const TAG: &'static str = $tag;
        }
    };
}

/// Convenience macro for declaring Filters.
#[macro_export]
macro_rules! filter {
    ($ty:ty, $tag:literal) => {
        impl $crate::hook::Filter for $ty {
            const TAG: &'static str = $tag;
        }
    };
}

/// Convenience macro for registering the plugin entry point and automatically initializing logging.
#[macro_export]
macro_rules! declare_plugin {
    ($init_fn:path) => {
        #[stabby::export]
        pub extern "C" fn init_plugin(ctx: $crate::PluginInitContext) -> $crate::InitResult {
            // Automatically initialize the plugin logger using the host's HookApi.
            $crate::logging::init_logger(ctx.hook_api.host_api);

            // Call the plugin's own initialization function.
            $init_fn(ctx)
        }
    };
}

