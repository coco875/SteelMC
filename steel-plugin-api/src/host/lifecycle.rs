//! Plugin lifecycle state; registry callbacks run immediately at their native events.
use crate::{ApiResult, PluginInit};
use stabby::{str::Str, string::String as ApiString};
use std::sync::Arc;
use steel_core::behavior::{BLOCK_BEHAVIORS, FLUID_BEHAVIORS};
use steel_utils::locks::SyncMutex;

#[derive(Default)]
struct RegistrationStatus {
    pending: usize,
    error: Option<String>,
}

pub(crate) struct PluginState {
    pub name: &'static str,
    pub config: String,
    status: SyncMutex<RegistrationStatus>,
}
impl PluginState {
    pub(super) fn new(name: &'static str, config: &str) -> Self {
        Self {
            name,
            config: config.to_owned(),
            status: SyncMutex::new(RegistrationStatus::default()),
        }
    }
    pub(crate) fn begin_registration(&self) {
        self.status.lock().pending += 1;
    }
    pub(crate) fn cancel_registration(&self) {
        self.status.lock().pending -= 1;
    }
    pub(crate) fn run_registration(&self, phase: &str, callback: impl FnOnce() -> ApiResult<()>) {
        let run = self.status.lock().error.is_none();
        let result: Result<(), ApiString> = if run { callback().into() } else { Ok(()) };
        let mut status = self.status.lock();
        status.pending -= 1;
        if let Err(error) = result {
            status.error = Some(format!(
                "plugin {} {phase} registration failed: {error}",
                self.name
            ));
        }
    }
    fn check_registration(&self) -> Result<(), String> {
        let status = self.status.lock();
        if let Some(error) = &status.error {
            return Err(error.to_owned());
        }
        if status.pending != 0 {
            return Err(format!(
                "plugin {} still has pending registration events",
                self.name
            ));
        }
        Ok(())
    }
}

/// Loaded and pre-initialized plugin, awaiting registry events and final initialization.
pub struct LoadedPlugin {
    /// Diagnostic name of this pinned module.
    pub name: &'static str,
    pub(super) state: Arc<PluginState>,
    pub(super) initialize: PluginInit,
}

/// Runs init only after every module's registration has completed successfully.
/// Consuming the loaded modules prevents accidentally invoking init twice.
pub fn initialize_all(plugins: Vec<LoadedPlugin>) -> Result<Vec<&'static str>, String> {
    if BLOCK_BEHAVIORS.get().is_none() || FLUID_BEHAVIORS.0.get().is_none() {
        return Err("plugin init requires frozen behavior registries".into());
    }
    for plugin in &plugins {
        plugin.state.check_registration()?;
    }
    let mut names = Vec::with_capacity(plugins.len());
    for plugin in plugins {
        let result: Result<(), ApiString> =
            (plugin.initialize)(Str::new(&plugin.state.config)).into();
        result.map_err(|error| format!("plugin {} init failed: {error}", plugin.name))?;
        names.push(plugin.name);
    }
    Ok(names)
}
