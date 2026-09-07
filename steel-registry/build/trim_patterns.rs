use crate::generator_functions::{
    generate_identifier, generate_text_component, read_ordered_minecraft_datapack_entries,
};
use crate::shared_structs::TextComponentJson;
use heck::ToShoutySnakeCase;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Deserialize;
use steel_utils::Identifier;

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct TrimPatternJson {
    asset_id: Identifier,
    description: TextComponentJson,
    #[serde(default)]
    decal: bool,
}

pub(crate) fn build(overlay: &steel_utils::datapack_overlay::DatapackOverlay) -> TokenStream {
    // TrimPatterns.bootstrap defines registry insertion order in Vanilla.
    const VANILLA_ORDER: &[&str] = &[
        "sentry",
        "dune",
        "coast",
        "wild",
        "ward",
        "eye",
        "vex",
        "tide",
        "snout",
        "rib",
        "spire",
        "wayfinder",
        "shaper",
        "silence",
        "raiser",
        "host",
        "flow",
        "bolt",
    ];

    let trim_patterns: Vec<(String, TrimPatternJson)> =
        read_ordered_minecraft_datapack_entries(overlay, "trim_pattern", VANILLA_ORDER);

    let mut definitions = TokenStream::new();
    let mut registrations = TokenStream::new();
    for (name, pattern) in trim_patterns {
        let ident = Ident::new(&name.to_shouty_snake_case(), Span::call_site());
        let key = quote! { Identifier::vanilla_static(#name) };
        let asset_id = generate_identifier(&pattern.asset_id);
        let description = generate_text_component(&pattern.description);
        let decal = pattern.decal;

        definitions.extend(quote! {
            pub static #ident: TrimPattern = TrimPattern::new(
                #key,
                TrimPatternValue::new(#asset_id, #description, #decal),
            );
        });
        registrations.extend(quote! {
            registry.register(&#ident);
        });
    }
    quote! {
        use crate::trim_pattern::{TrimPattern, TrimPatternRegistry, TrimPatternValue};
        use steel_utils::Identifier;
        use std::borrow::Cow;
        use text_components::{TextComponent, translation::TranslatedMessage};

        #definitions

        pub fn register_trim_patterns(registry: &mut TrimPatternRegistry) {
            #registrations
        }
    }
}
