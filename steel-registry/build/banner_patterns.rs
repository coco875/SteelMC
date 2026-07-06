use crate::generator_functions::{generate_identifier, read_minecraft_datapack_entries};
use heck::ToShoutySnakeCase;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Deserialize;
use steel_utils::Identifier;

#[derive(Deserialize, Debug)]
pub struct BannerPatternJson {
    asset_id: Identifier,
    translation_key: String,
}

pub(crate) fn build(overlay: &steel_utils::datapack_overlay::DatapackOverlay) -> TokenStream {
    let banner_patterns: Vec<(String, BannerPatternJson)> =
        read_minecraft_datapack_entries(overlay, "banner_pattern");

    let mut stream = TokenStream::new();

    stream.extend(quote! {
        use crate::banner_pattern::{BannerPattern, BannerPatternRegistry};
        use steel_utils::Identifier;
        use std::borrow::Cow;
    });

    // Generate static banner pattern definitions
    let mut register_stream = TokenStream::new();
    for (banner_pattern_name, banner_pattern) in &banner_patterns {
        let banner_pattern_ident = Ident::new(
            &banner_pattern_name.to_shouty_snake_case(),
            Span::call_site(),
        );
        let banner_pattern_name_str = banner_pattern_name.clone();

        let key = quote! { Identifier::vanilla_static(#banner_pattern_name_str) };
        let asset_id = generate_identifier(&banner_pattern.asset_id);
        let translation_key = banner_pattern.translation_key.as_str();

        stream.extend(quote! {
            pub static #banner_pattern_ident: BannerPattern = BannerPattern {
                key: #key,
                asset_id: #asset_id,
                translation_key: #translation_key,
            };
        });
        register_stream.extend(quote! {
            registry.register(&#banner_pattern_ident);
        });
    }

    stream.extend(quote! {
        pub fn register_banner_patterns(registry: &mut BannerPatternRegistry) {
            #register_stream
        }
    });

    stream
}
