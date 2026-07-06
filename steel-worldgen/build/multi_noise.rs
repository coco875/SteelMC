use std::collections::BTreeMap;
use std::fs;
use std::ops::Deref;

use steel_utils::datapack_overlay::DatapackOverlay;

use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Deserialize;

/// A biome entry from the extracted multi-noise biome source parameter list.
#[derive(Clone, Deserialize)]
struct BiomeEntry {
    biome: String,
    parameters: BiomeParameters,
}

/// A climate parameter range, which can be deserialized from either a single number or a pair.
#[derive(Clone, Debug)]
struct ParameterRange([f64; 2]);

impl<'de> Deserialize<'de> for ParameterRange {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum RawRange {
            Single(f64),
            Pair([f64; 2]),
        }
        match RawRange::deserialize(deserializer)? {
            RawRange::Single(val) => Ok(ParameterRange([val, val])),
            RawRange::Pair(pair) => Ok(ParameterRange(pair)),
        }
    }
}

impl Deref for ParameterRange {
    type Target = [f64; 2];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Climate parameters for a biome entry.
#[derive(Clone, Deserialize)]
struct BiomeParameters {
    temperature: ParameterRange,
    humidity: ParameterRange,
    continentalness: ParameterRange,
    erosion: ParameterRange,
    depth: ParameterRange,
    weirdness: ParameterRange,
    offset: f64,
}

#[derive(Deserialize)]
struct DatapackBiomeSourceParameters {
    preset: Option<String>,
    #[serde(default)]
    biomes: Vec<BiomeEntry>,
}

struct GeneratedEndTokens {
    biome_source_kind: TokenStream,
    biome_source_type: TokenStream,
    chunk_biome_sampler_type: TokenStream,
    new_biome_source: TokenStream,
    new_chunk_biome_sampler: TokenStream,
    possible_biome_refs: TokenStream,
}

impl GeneratedEndTokens {
    fn from_source_kind(end_uses_multi_noise: bool) -> Self {
        if end_uses_multi_noise {
            return Self {
                biome_source_kind: quote! { EndBiomeSourceKind::MultiNoise },
                biome_source_type: quote! { crate::biomes::biome_source::EndClimateSampler },
                chunk_biome_sampler_type: quote! {
                    crate::biomes::biome_source::EndMultiNoiseChunkBiomeSampler<'a>
                },
                new_biome_source: quote! {
                    crate::biomes::biome_source::EndClimateSampler::new(seed)
                },
                new_chunk_biome_sampler: quote! {
                    crate::biomes::biome_source::EndMultiNoiseChunkBiomeSampler {
                        source,
                        column_cache: crate::density_functions::end::EndColumnCache::new(),
                        biome_cache: None,
                    }
                },
                possible_biome_refs: quote! {
                    THE_END_BIOME_PARAMETERS
                        .values()
                        .iter()
                        .map(|(_, biome)| *biome)
                        .collect()
                },
            };
        }

        Self {
            biome_source_kind: quote! { EndBiomeSourceKind::Vanilla },
            biome_source_type: quote! { crate::noise::EndIslands },
            chunk_biome_sampler_type: quote! {
                crate::biomes::biome_source::VanillaEndChunkBiomeSampler<'a>
            },
            new_biome_source: quote! {
                crate::noise::EndIslands::new(seed)
            },
            new_chunk_biome_sampler: quote! {
                crate::biomes::biome_source::VanillaEndChunkBiomeSampler {
                    source,
                    cached_erosion: None,
                }
            },
            possible_biome_refs: quote! {
                vec![
                    &vanilla_biomes::THE_END,
                    &vanilla_biomes::END_HIGHLANDS,
                    &vanilla_biomes::END_MIDLANDS,
                    &vanilla_biomes::SMALL_END_ISLANDS,
                    &vanilla_biomes::END_BARRENS,
                ]
            },
        }
    }
}

fn apply_datapack_parameter_lists(
    overlay: &DatapackOverlay,
    presets: &mut BTreeMap<String, Vec<BiomeEntry>>,
) {
    for (id, json) in overlay
        .list_json_registry_ids_with_suffix("worldgen/multi_noise_biome_source_parameter_list")
    {
        let params: DatapackBiomeSourceParameters =
            serde_json::from_str(&json).unwrap_or_else(|error| {
                panic!("Failed to parse multi-noise parameter list {id}: {error}")
            });

        if !params.biomes.is_empty() {
            presets.insert(id.clone(), params.biomes);
        } else if let Some(preset) = &params.preset
            && !presets.contains_key(&id)
            && let Some(entries) = presets.get(preset).cloned()
        {
            presets.insert(id.clone(), entries);
        }
    }
}

/// Generate the Rust code for multi-noise biome parameter lists (all presets).
#[expect(clippy::too_many_lines, reason = "build function contains generated structures and parsing code")]
pub(crate) fn build(overlay: &DatapackOverlay) -> TokenStream {
    println!("cargo:rerun-if-changed=build_assets/multi_noise_biome_source_parameters.json");

    let content = fs::read_to_string("build_assets/multi_noise_biome_source_parameters.json")
        .expect("Failed to read multi_noise_biome_source_parameters.json");
    let mut presets: BTreeMap<String, Vec<BiomeEntry>> =
        serde_json::from_str(&content).expect("Failed to parse multi-noise biome parameters JSON");

    apply_datapack_parameter_lists(overlay, &mut presets);

    // Load custom End dimension from the datapack if it overrides biome_source with multi_noise
    let mut end_uses_multi_noise = false;
    if let Some(end_json) = overlay.read_string("minecraft/dimension/the_end.json") {
        #[derive(Deserialize)]
        struct DatapackDimension {
            generator: Option<DatapackGenerator>,
        }
        #[derive(Deserialize)]
        struct DatapackGenerator {
            #[serde(rename = "type")]
            gen_type: String,
            biome_source: Option<DatapackBiomeSource>,
        }
        #[derive(Deserialize)]
        struct DatapackBiomeSource {
            #[serde(rename = "type")]
            source_type: String,
            #[serde(default)]
            biomes: Vec<BiomeEntry>,
        }

        let source = serde_json::from_str::<DatapackDimension>(&end_json)
            .ok()
            .and_then(|dim| dim.generator)
            .filter(|g| g.gen_type == "minecraft:noise")
            .and_then(|g| g.biome_source)
            .filter(|s| s.source_type == "minecraft:multi_noise" && !s.biomes.is_empty());

        if let Some(source) = source {
            presets.insert("minecraft:the_end".to_string(), source.biomes);
            end_uses_multi_noise = true;
        }
    }

    if !presets.contains_key("minecraft:the_end") {
        presets.insert("minecraft:the_end".to_string(), Vec::new());
    }

    let mut stream = TokenStream::new();
    let end_tokens = GeneratedEndTokens::from_source_kind(end_uses_multi_noise);
    let end_biome_source_kind = end_tokens.biome_source_kind;
    let generated_end_biome_source = end_tokens.biome_source_type;
    let generated_end_chunk_biome_sampler = end_tokens.chunk_biome_sampler_type;
    let new_end_biome_source = end_tokens.new_biome_source;
    let new_end_chunk_biome_sampler = end_tokens.new_chunk_biome_sampler;
    let end_possible_biome_refs = end_tokens.possible_biome_refs;

    stream.extend(quote! {
        //! Generated multi-noise biome source parameters for all presets.
        //!
        //! Auto-generated from steel-worldgen/build_assets/multi_noise_biome_source_parameters.json.
        //! Do not edit manually.

        use steel_registry::biome::BiomeRef;
        use steel_registry::vanilla_biomes;
        use steel_utils::climate::{Parameter, ParameterList, ParameterPoint, TargetPoint};
        use std::sync::LazyLock;

        /// Generated End biome source kind after datapack overlays are applied.
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum EndBiomeSourceKind {
            Vanilla,
            MultiNoise,
        }

        pub const END_BIOME_SOURCE_KIND: EndBiomeSourceKind = #end_biome_source_kind;

        pub(crate) type GeneratedEndBiomeSource = #generated_end_biome_source;

        pub(crate) type GeneratedEndChunkBiomeSampler<'a> = #generated_end_chunk_biome_sampler;

        pub(crate) fn new_end_biome_source(seed: u64) -> GeneratedEndBiomeSource {
            #new_end_biome_source
        }

        pub(crate) fn new_end_chunk_biome_sampler(
            source: &GeneratedEndBiomeSource,
        ) -> GeneratedEndChunkBiomeSampler<'_> {
            #new_end_chunk_biome_sampler
        }

        pub fn end_possible_biome_refs() -> Vec<BiomeRef> {
            #end_possible_biome_refs
        }
    });

    // Generate each preset
    for (preset_name, entries) in &presets {
        let short_name = preset_name
            .strip_prefix("minecraft:")
            .unwrap_or(preset_name);
        let upper_name = short_name.to_uppercase();

        let static_ident = Ident::new(&format!("{upper_name}_BIOME_PARAMETERS"), Span::call_site());
        let points_ident = Ident::new(&format!("{upper_name}_BIOME_POINTS"), Span::call_site());
        let lookup_fn = Ident::new(&format!("lookup_{short_name}_biome"), Span::call_site());
        let get_fn = Ident::new(&format!("get_{short_name}_biome"), Span::call_site());
        let get_cached_fn =
            Ident::new(&format!("get_{short_name}_biome_cached"), Span::call_site());

        let (points_tokens, arms_tokens) = generate_biome_entries(entries);
        let doc_static = format!(
            "{} biome parameter list for multi-noise biome selection.",
            capitalize(short_name)
        );
        let doc_get = format!("Get the biome for a target point in the {short_name}.");
        let doc_cached = format!(
            "Get the biome with lastResult caching for the {short_name} (matches vanilla's ThreadLocal warm-start)."
        );

        // Emit climate points as a `static` of `const`-constructed values so they live
        // in `.rodata` instead of being built inside the LazyLock closure. This keeps
        // LLVM from having to optimize a single multi-megabyte function full of
        // inlined `Parameter::new` / `ParameterPoint::new` calls.
        stream.extend(quote! {
            static #points_ident: &[ParameterPoint] = &[
                #points_tokens
            ];

            fn #lookup_fn(i: usize) -> BiomeRef {
                match i {
                    #arms_tokens
                    _ => unreachable!(),
                }
            }

            #[doc = #doc_static]
            pub static #static_ident: LazyLock<ParameterList<BiomeRef>> = LazyLock::new(|| {
                let entries: Vec<(ParameterPoint, BiomeRef)> = #points_ident
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (*p, #lookup_fn(i)))
                    .collect();
                ParameterList::new(entries)
            });

            #[doc = #doc_get]
            #[inline]
            pub fn #get_fn(target: &steel_utils::climate::TargetPoint) -> BiomeRef {
                *#static_ident.find_value(target)
            }

            #[doc = #doc_cached]
            #[inline]
            pub fn #get_cached_fn(target: &steel_utils::climate::TargetPoint, cache: &mut Option<usize>) -> BiomeRef {
                *#static_ident.find_value_cached(target, cache)
            }
        });
    }

    stream
}

/// Quantize a float value matching vanilla's `(long)(float * 10000.0F)`.
///
/// The JSON values are f64, but vanilla uses f32 arithmetic for quantization.
/// Casting to f32 first ensures bit-exact matching with Java's float precision.
fn quantize(v: f64) -> i64 {
    ((v as f32) * 10000.0f32) as i64
}

/// Convert a biome name like `"minecraft:plains"` to the `vanilla_biomes` constant
/// identifier `PLAINS`.
fn biome_ident(name: &str) -> Ident {
    Ident::new(&super::biome_ident_str(name), Span::call_site())
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().to_string() + c.as_str(),
    }
}

/// Build (points-static body, lookup-function match arms).
///
/// Splitting these lets the climate ranges live as compile-time `const`-evaluated
/// data while the biome reference resolution (which deref's `LazyLock<Biome>` and
/// can't be const) stays in a small runtime function.
fn generate_biome_entries(entries: &[BiomeEntry]) -> (TokenStream, TokenStream) {
    let mut points = Vec::with_capacity(entries.len());
    let mut arms = Vec::with_capacity(entries.len());

    for (i, entry) in entries.iter().enumerate() {
        let p = &entry.parameters;
        let temp_min = quantize(p.temperature[0]);
        let temp_max = quantize(p.temperature[1]);
        let hum_min = quantize(p.humidity[0]);
        let hum_max = quantize(p.humidity[1]);
        let cont_min = quantize(p.continentalness[0]);
        let cont_max = quantize(p.continentalness[1]);
        let ero_min = quantize(p.erosion[0]);
        let ero_max = quantize(p.erosion[1]);
        let depth_min = quantize(p.depth[0]);
        let depth_max = quantize(p.depth[1]);
        let weird_min = quantize(p.weirdness[0]);
        let weird_max = quantize(p.weirdness[1]);
        let offset = quantize(p.offset);

        let biome = biome_ident(&entry.biome);

        points.push(quote! {
            ParameterPoint::new(
                Parameter::new(#temp_min, #temp_max),
                Parameter::new(#hum_min, #hum_max),
                Parameter::new(#cont_min, #cont_max),
                Parameter::new(#ero_min, #ero_max),
                Parameter::new(#depth_min, #depth_max),
                Parameter::new(#weird_min, #weird_max),
                #offset,
            ),
        });

        arms.push(quote! {
            #i => &*vanilla_biomes::#biome,
        });
    }

    let points_tokens = quote! { #(#points)* };
    let arms_tokens = quote! { #(#arms)* };
    (points_tokens, arms_tokens)
}
