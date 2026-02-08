//! Build-time code generator for noise parameters.

use std::fs;
use std::path::Path;

use heck::ToShoutySnakeCase;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Deserialize;

/// Represents the structure of each noise parameter JSON file.
#[derive(Deserialize)]
struct NoiseParams {
    #[serde(rename = "firstOctave")]
    first_octave: i32,
    amplitudes: Vec<f64>,
}

/// Generates noise parameter constants from datapack JSON files.
pub(crate) fn build() -> TokenStream {
    let noise_dir = Path::new(
        "../steel-registry/build_assets/builtin_datapacks/minecraft/data/minecraft/worldgen/noise",
    );

    println!("cargo:rerun-if-changed={}", noise_dir.display());

    // Collect all noise parameters from individual JSON files
    let mut params: Vec<(String, NoiseParams)> = Vec::new();

    for entry in fs::read_dir(noise_dir).expect("Failed to read noise directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();

        if path.extension().is_some_and(|ext| ext == "json") {
            let file_name = path
                .file_stem()
                .expect("Failed to get file stem")
                .to_str()
                .expect("Failed to convert file stem to str")
                .to_string();

            let json_content =
                fs::read_to_string(&path).expect("Failed to read noise parameter file");
            let noise_params: NoiseParams =
                serde_json::from_str(&json_content).expect("Failed to parse noise parameter file");

            params.push((file_name, noise_params));
        }
    }

    // Sort for deterministic output
    params.sort_by(|a, b| a.0.cmp(&b.0));

    let mut stream = TokenStream::new();

    // Generate the struct definition
    stream.extend(quote! {
        /// Parameters for double perlin noise generation.
        pub struct DoublePerlinNoiseParameters {
            /// The first octave level for the noise.
            pub first_octave: i32,
            /// Amplitude multipliers for each octave.
            pub amplitudes: &'static [f64],
            /// The identifier for this noise type.
            id: &'static str,
        }

        impl DoublePerlinNoiseParameters {
            /// Creates a new noise parameter set.
            pub const fn new(first_octave: i32, amplitudes: &'static [f64], id: &'static str) -> Self {
                Self {
                    first_octave,
                    amplitudes,
                    id,
                }
            }

            /// Returns the identifier for this noise type.
            pub const fn id(&self) -> &'static str {
                self.id
            }
        }
    });

    // Generate constants
    let mut const_names = Vec::new();
    let mut match_arms = Vec::new();

    for (key, noise_params) in &params {
        let first_octave = noise_params.first_octave;
        let amplitudes = &noise_params.amplitudes;

        let const_name_str = key.to_shouty_snake_case();
        let const_name = Ident::new(&const_name_str, Span::call_site());

        // Generate the amplitudes array
        let amps_tokens: Vec<_> = amplitudes.iter().map(|a| quote! { #a }).collect();

        // The minecraft: prefix for the id
        let id = format!("minecraft:{key}");

        stream.extend(quote! {
            /// Noise parameters for #key.
            pub const #const_name: DoublePerlinNoiseParameters =
                DoublePerlinNoiseParameters::new(#first_octave, &[#(#amps_tokens),*], #id);
        });

        const_names.push(const_name.clone());
        match_arms.push((key.clone(), const_name));
    }

    // Generate id_to_parameters function
    let match_arms_tokens: Vec<_> = match_arms
        .iter()
        .map(|(key, const_name)| {
            quote! {
                #key => &#const_name,
            }
        })
        .collect();

    stream.extend(quote! {
        impl DoublePerlinNoiseParameters {
            /// Looks up noise parameters by their identifier (without minecraft: prefix).
            pub fn id_to_parameters(id: &str) -> Option<&'static DoublePerlinNoiseParameters> {
                Some(match id {
                    #(#match_arms_tokens)*
                    _ => return None,
                })
            }
        }
    });

    stream
}
