use crate::generator_functions::{generate_identifier, read_minecraft_datapack_entries};
use heck::ToShoutySnakeCase;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Deserialize;
use steel_utils::Identifier;

#[derive(Deserialize, Debug)]
pub struct TrimPatternJson {
    asset_id: Identifier,
    description: TextComponent,
    #[serde(default)]
    decal: bool,
}

#[derive(Deserialize, Debug)]
pub struct TextComponent {
    translate: String,
}

pub(crate) fn build(overlay: &steel_utils::datapack_overlay::DatapackOverlay) -> TokenStream {
    let trim_patterns: Vec<(String, TrimPatternJson)> =
        read_minecraft_datapack_entries(overlay, "trim_pattern");

    let mut stream = TokenStream::new();

    stream.extend(quote! {
        use crate::trim_pattern::{
            TrimPattern, TrimPatternRegistry,
        };
        use steel_utils::Identifier;
        use text_components::{TextComponent, translation::TranslatedMessage};
        use std::borrow::Cow;
    });

    // Generate static trim pattern definitions
    let mut register_stream = TokenStream::new();
    for (trim_pattern_name, trim_pattern) in &trim_patterns {
        let trim_pattern_ident =
            Ident::new(&trim_pattern_name.to_shouty_snake_case(), Span::call_site());
        let trim_pattern_name_str = trim_pattern_name.clone();

        let key = quote! { Identifier::vanilla_static(#trim_pattern_name_str) };
        let asset_id = generate_identifier(&trim_pattern.asset_id);
        let translate = &trim_pattern.description.translate;
        let decal = trim_pattern.decal;

        stream.extend(quote! {
            pub static #trim_pattern_ident: TrimPattern = TrimPattern {
                key: #key,
                asset_id: #asset_id,
                description: TextComponent::translated(TranslatedMessage::new(#translate, None)),
                decal: #decal,
            };
        });

        register_stream.extend(quote! {
            registry.register(&#trim_pattern_ident);
        });
    }

    stream.extend(quote! {
        pub fn register_trim_patterns(registry: &mut TrimPatternRegistry) {
            #register_stream
        }
    });

    stream
}
