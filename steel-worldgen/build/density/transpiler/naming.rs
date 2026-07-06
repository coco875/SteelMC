//! Identifier generation for transpiled Rust output.
//!
//! Converts namespaced density function and noise IDs into valid `Ident`s for
//! struct fields, cache grids, and generated `compute_*` / `router_*` functions.

use proc_macro2::Ident;
use quote::format_ident;

pub(super) fn noise_field_ident(noise_id: &str) -> Ident {
    format_ident!("n_{}", sanitize_name(noise_id))
}

pub(super) fn named_fn_field_ident(name: &str) -> Ident {
    format_ident!("df_{}", sanitize_name(name))
}

pub(super) fn named_fn_ident(name: &str) -> Ident {
    format_ident!("compute_{}", sanitize_name(name))
}

pub(super) fn named_fn_ident_4x(name: &str) -> Ident {
    format_ident!("compute_{}_4x", sanitize_name(name))
}

pub(super) fn grid_field_ident(name: &str) -> Ident {
    format_ident!("grid_df_{}", sanitize_name(name))
}

pub(super) fn router_cache_field_ident(name: &str) -> Ident {
    format_ident!("router_{}", sanitize_name(name))
}

pub(super) fn router_grid_field_ident(name: &str) -> Ident {
    format_ident!("grid_router_{}", sanitize_name(name))
}

pub(super) fn router_compute_fn_ident(name: &str) -> Ident {
    format_ident!("compute_router_{}", sanitize_name(name))
}

/// Converts a namespaced ID to a valid Rust identifier.
///
/// `"minecraft:overworld/continents"` → `"overworld__continents"`
/// `"mymod:custom/noise"` → `"custom__noise"`
pub(super) fn sanitize_name(id: &str) -> String {
    let id = steel_utils::Identifier::parse_or_vanilla(id)
        .unwrap_or_else(|error| panic!("invalid density function identifier {id}: {error}"));
    id.path.replace('/', "__").replace('-', "_")
}
