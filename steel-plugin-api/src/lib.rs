//! Stable ABI Plugin API for SteelMC.

mod macros;
pub mod hook;
pub mod logging;

pub use stabby::alloc::string::String as AbiString;
pub use stabby::str::Str as AbiStr;

pub use hook::{
    Action, Filter, HookApi, HookApiVtable, PluginInitContext,
    OrderingConstraint, HandlerOrdering, RawActionCallback, RawFilterCallback,
};

#[stabby::stabby]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitResult {
    Ok,
    /// The init function panicked. Host should shut down gracefully.
    Panic,
}

pub type InitPluginFn = extern "C" fn(ctx: PluginInitContext) -> InitResult;
