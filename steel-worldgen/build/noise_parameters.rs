use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Deserialize;
use std::collections::BTreeMap;
use steel_utils::datapack_overlay::DatapackOverlay;

/// JSON structure for a noise parameter entry (matches datapack format).
#[derive(Deserialize, Debug)]
struct NoiseParamsJson {
    #[serde(rename = "firstOctave")]
    first_octave: i32,
    amplitudes: Vec<f64>,
}

/// Generate noise parameters code from the vanilla datapack.
pub(crate) fn build(overlay: &DatapackOverlay) -> TokenStream {
    let mut noises: BTreeMap<String, NoiseParamsJson> = BTreeMap::new();
    for (id, content) in overlay.list_json_registry_ids_with_suffix("worldgen/noise") {
        let params: NoiseParamsJson = serde_json::from_str(&content)
            .unwrap_or_else(|error| panic!("Failed to parse noise parameter {id}: {error}"));
        noises.insert(id, params);
    }

    let mut stream = TokenStream::new();

    stream.extend(quote! {
        //! Generated vanilla noise parameters from the datapack.
        //!
        //! Auto-generated from `steel-utils/build_assets/builtin_datapacks/minecraft/worldgen/noise/*.json`.
        //! Do not edit manually.

        use rustc_hash::FxHashMap;
        use steel_worldgen::density::NoiseParameters;
    });

    // Generate static amplitude arrays
    for (id, params) in &noises {
        let const_name = Ident::new(
            &format!("{}_AMPLITUDES", id.replace([':', '/'], "_").to_uppercase()),
            Span::call_site(),
        );
        let amplitudes = &params.amplitudes;

        stream.extend(quote! {
            static #const_name: &[f64] = &[#(#amplitudes),*];
        });
    }

    // Generate the get_noise_parameters function
    let entries: Vec<TokenStream> = noises
        .iter()
        .map(|(id, params)| {
            let amp_name = Ident::new(
                &format!("{}_AMPLITUDES", id.replace([':', '/'], "_").to_uppercase()),
                Span::call_site(),
            );
            let first_octave = params.first_octave;

            quote! {
                (String::from(#id), NoiseParameters::new(#first_octave, #amp_name.to_vec())),
            }
        })
        .collect();

    stream.extend(quote! {
        /// Get all vanilla noise parameters from the datapack.
        ///
        /// Returns a map keyed by namespaced noise ID (e.g., `"minecraft:temperature"`).
        #[must_use]
        pub fn get_noise_parameters() -> FxHashMap<String, NoiseParameters> {
            FxHashMap::from_iter([
                #(#entries)*
            ])
        }
    });

    stream
}
