//! Example plugin implementation for SteelMC.

use steel_plugin_api::hook::{ServerStopAction, ServerTickAction};
use steel_plugin_api::{InitResult, PluginInitContext};

extern "C" fn on_server_tick(args: &ServerTickAction) {
    log::info!("Tick received in plugin: count = {}", args.tick_count);
}

extern "C" fn on_server_stop(_args: &ServerStopAction) {
    log::info!("Stop received in plugin");
}

fn init(ctx: PluginInitContext) -> InitResult {
    ctx.hook_api.add_action::<ServerTickAction>(on_server_tick)
        .register();
    ctx.hook_api.add_action::<ServerStopAction>(on_server_stop)
        .register();

    log::info!("Hello plugin: {}", ctx.hook_api.plugin_id.as_str());
    InitResult::Ok
}

steel_plugin_api::declare_plugin!(init);

