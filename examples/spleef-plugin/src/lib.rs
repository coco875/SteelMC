//! A standalone Spleef room. Game rules and AI live here; the host owns worlds and actors.
mod ai;
mod game;
mod map;
use std::sync::{Mutex, OnceLock};
use steel_plugin_api::{
    ApiResult, Command, Commands, GeneratorRegistry, Hooks, Server,
    stabby::{self, str::Str, string::String as ApiString},
};
#[expect(
    clippy::disallowed_types,
    reason = "plugin-private state must not link host-only steel-utils"
)]
static GAME: OnceLock<Mutex<game::Game>> = OnceLock::new();
macro_rules! api {
    ($value:expr) => {
        $value.match_owned(Ok, |error| Err(error.to_string()))
    };
}
pub(crate) use api;
fn pre_init(mut hooks: Hooks<'_>) -> ApiResult<()> {
    let result = (|| -> Result<(), String> {
        api!(hooks.register_commands(commands))?;
        api!(hooks.register_generator_registration(generators))?;
        hooks.register_server_tick(tick);
        Ok(())
    })();
    result.map_err(ApiString::from).into()
}
#[stabby::export(callback)]
fn commands(mut registry: Commands<'_>) -> ApiResult<()> {
    registry.register_player_command(
        "spleef:spleef".into(),
        [Str::new("join"), Str::new("leave"), Str::new("status")]
            .as_slice()
            .into(),
        command,
    )
}
#[stabby::export(callback)]
fn generators(mut registry: GeneratorRegistry<'_>) -> ApiResult<()> {
    let result = (|| -> Result<(), String> {
        api!(registry.register_terrain_generator(
            "spleef:lobby".into(),
            "minecraft:overworld".into(),
            "minecraft:plains".into(),
            0,
            map::lobby
        ))?;
        api!(registry.register_terrain_generator(
            "spleef:arena".into(),
            "minecraft:overworld".into(),
            "minecraft:plains".into(),
            0,
            map::arena
        ))?;
        Ok(())
    })();
    result.map_err(ApiString::from).into()
}
#[expect(
    clippy::disallowed_types,
    reason = "plugin-private state must not link host-only steel-utils"
)]
fn init(config: Str<'_>) -> ApiResult<()> {
    let result = (|| -> Result<(), String> {
        let config: game::Config = toml::from_str(config.as_str()).map_err(|e| e.to_string())?;
        config.validate()?;
        map::initialize()?;
        GAME.set(Mutex::new(game::Game::new(config)))
            .map_err(|_| "Spleef initialized twice")?;
        Ok(())
    })();
    result.map_err(ApiString::from).into()
}
#[stabby::export(callback)]
fn command(context: Command<'_>, action: Str<'_>) -> ApiResult<i32> {
    let result = (|| -> Result<i32, String> {
        let game = GAME.get().ok_or("Spleef is not initialized")?;
        let mut game = game.lock().map_err(|_| "Spleef state lock poisoned")?;
        let player = api!(context.player())?;
        let server = context.server();
        match action.as_str() {
            "join" => game.join(&server, player)?,
            "leave" => game.leave(player)?,
            "status" => {}
            _ => return Err("unknown Spleef action".into()),
        }
        let status = game.status();
        context.reply(status.as_str().into());
        Ok(1)
    })();
    result.map_err(ApiString::from).into()
}
#[stabby::export(callback)]
fn tick(server: Server<'_>, tick: u64) {
    let Some(game) = GAME.get() else {
        return;
    };
    let mut game = match game.lock() {
        Ok(game) => game,
        Err(error) => {
            eprintln!("Spleef state unavailable: {error}");
            return;
        }
    };
    if let Err(error) = game.tick(&server, tick) {
        eprintln!("Spleef stopped: {error}");
        game.begin_return();
    }
}
steel_plugin_api::export_plugin!("spleef-example", pre_init, init);
