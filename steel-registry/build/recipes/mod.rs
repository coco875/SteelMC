//! Build-time generation of typed vanilla recipe declarations.

use proc_macro2::TokenStream;
use quote::quote;
use serde_json::Value;
use steel_utils::datapack_overlay::DatapackOverlay;

mod cooking;
mod crafting;
mod serializer;
mod shared;
mod smithing;
mod stonecutting;

use serializer::RecipeSerializer;
use shared::{recipe_ident, string_field};

type RecipeGenerator = fn(&Value) -> TokenStream;

struct ParsedRecipe {
    name: String,
    value: Value,
}

pub(crate) fn build(overlay: &DatapackOverlay) -> TokenStream {
    let mut recipes: Vec<_> = overlay
        .list_json_relative("minecraft/recipe")
        .into_iter()
        .map(|(name, source)| ParsedRecipe {
            value: serde_json::from_str(&source)
                .unwrap_or_else(|error| panic!("Cannot parse recipe {name}: {error}")),
            name,
        })
        .collect();
    recipes.sort_by(|left, right| left.name.cmp(&right.name));

    let declarations: Vec<_> = recipes.iter().map(generate_declaration).collect();
    let registrations: Vec<_> = recipes
        .iter()
        .map(|recipe| {
            let ident = recipe_ident(&recipe.name);
            quote! { registry.register(&#ident); }
        })
        .collect();

    quote! {
        use std::sync::LazyLock;

        use steel_utils::Identifier;

        use crate::{
            data_components::{DataComponentPatch, vanilla_components},
            item_stack_template::ItemStackTemplate,
            recipe::*,
            vanilla_items, vanilla_mob_effects, vanilla_trim_patterns,
        };
        use crate::data_components::vanilla_components::{
            FireworkExplosionShape, SuspiciousStewEffect, SuspiciousStewEffects,
        };

        #(#declarations)*

        /// Registers every extracted vanilla recipe using its hardcoded static declaration.
        pub fn register_recipes(registry: &mut RecipeRegistry) {
            #(#registrations)*
        }
    }
}

fn generate_declaration(recipe: &ParsedRecipe) -> TokenStream {
    let serializer_identifier = string_field(&recipe.value, "type");
    let Some(serializer) = RecipeSerializer::from_identifier(serializer_identifier) else {
        panic!(
            "Unsupported extracted recipe type {serializer_identifier} for {}",
            recipe.name
        );
    };
    let ident = recipe_ident(&recipe.name);
    let name = &recipe.name;
    let data = serializer.generate_data(&recipe.value);
    let (rust_type, operational_type) = serializer.recipe_type_tokens();

    quote! {
        pub static #ident: LazyLock<#rust_type> = LazyLock::new(|| {
            Recipe::new(
                Identifier::vanilla_static(#name),
                &#operational_type,
                #data,
            )
        });
    }
}
