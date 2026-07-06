use std::collections::BTreeMap;
use std::path::PathBuf;

use proc_macro2::TokenStream;
use quote::quote;
use serde::Deserialize;
use serde_json::Value;
use steel_utils::Identifier;
use steel_utils::datapack_overlay::{DatapackFileSource, DatapackOverlay};

// ── JSON structures ──

#[derive(Deserialize, Debug)]
struct PoolJson {
    fallback: String,
    elements: Vec<WeightedElementJson>,
}

#[derive(Deserialize, Debug)]
struct WeightedElementJson {
    element: ElementJson,
    weight: i32,
}

#[derive(Deserialize, Debug)]
struct ElementJson {
    element_type: String,
    #[serde(default)]
    location: Option<String>,
    #[serde(default)]
    processors: Option<ProcessorsJson>,
    #[serde(default)]
    projection: Option<String>,
    #[serde(default)]
    feature: Option<String>,
    #[serde(default)]
    elements: Option<Vec<ElementJson>>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub(crate) enum ProcessorsJson {
    Registry(String),
    Direct { processors: Vec<Value> },
}

// ── Code generation helpers ──

fn gen_identifier(id: &str) -> TokenStream {
    if id.is_empty() {
        panic!("Cannot generate an empty identifier");
    }
    let id = Identifier::parse_or_vanilla(id)
        .unwrap_or_else(|error| panic!("invalid template pool identifier {id}: {error}"));
    let namespace = id.namespace.as_ref();
    let path = id.path.as_ref();
    if namespace == Identifier::VANILLA_NAMESPACE {
        quote! { Identifier::vanilla_static(#path) }
    } else {
        quote! { Identifier::new_static(#namespace, #path) }
    }
}

fn identifier_parts(id: &str, context: &str) -> (String, String) {
    let id = Identifier::parse_or_vanilla(id)
        .unwrap_or_else(|error| panic!("invalid {context} identifier {id}: {error}"));
    (id.namespace.into_owned(), id.path.into_owned())
}

fn required<T>(value: Option<T>, context: &str, field: &str) -> T {
    value.unwrap_or_else(|| panic!("Missing required field {field} in {context}"))
}

fn gen_projection(proj: &Option<String>, context: &str) -> TokenStream {
    match proj.as_deref() {
        Some("rigid") => quote! { Projection::Rigid },
        Some("terrain_matching") => quote! { Projection::TerrainMatching },
        Some(other) => panic!("Unknown projection {other} in {context}"),
        None => panic!("Missing required field projection in {context}"),
    }
}

pub(crate) fn gen_processors(processors: Option<&ProcessorsJson>, context: &str) -> TokenStream {
    match processors {
        Some(ProcessorsJson::Registry(id)) => {
            let id = gen_identifier(id);
            quote! { ProcessorList::Registry(#id) }
        }
        Some(ProcessorsJson::Direct { processors }) => {
            if processors.is_empty() {
                return quote! { ProcessorList::Empty };
            }
            let processors = processors.iter().enumerate().map(|(index, processor)| {
                let data = serde_json::from_value(processor.clone()).unwrap_or_else(|err| {
                    panic!("Failed to parse direct processor {index} in {context}: {err}")
                });
                crate::structure_processors::generate_processor_kind(&data)
            });
            quote! { ProcessorList::Direct(vec![#(#processors),*]) }
        }
        None => panic!("Missing required field processors in {context}"),
    }
}

fn gen_element(elem: &ElementJson, context: &str) -> TokenStream {
    let element_type = Identifier::parse_or_vanilla(&elem.element_type).unwrap_or_else(|error| {
        panic!(
            "invalid template pool element type {} in {context}: {error}",
            elem.element_type
        )
    });
    match element_type.to_string().as_str() {
        "minecraft:single_pool_element" => {
            let location = gen_identifier(required(elem.location.as_deref(), context, "location"));
            let processors = gen_processors(elem.processors.as_ref(), context);
            let projection = gen_projection(&elem.projection, context);
            quote! { PoolElement::Single { location: #location, processors: #processors, projection: #projection } }
        }
        "minecraft:legacy_single_pool_element" => {
            let location = gen_identifier(required(elem.location.as_deref(), context, "location"));
            let processors = gen_processors(elem.processors.as_ref(), context);
            let projection = gen_projection(&elem.projection, context);
            quote! { PoolElement::LegacySingle { location: #location, processors: #processors, projection: #projection } }
        }
        "minecraft:empty_pool_element" => {
            quote! { PoolElement::Empty }
        }
        "minecraft:feature_pool_element" => {
            let feature = gen_identifier(required(elem.feature.as_deref(), context, "feature"));
            let projection = gen_projection(&elem.projection, context);
            quote! { PoolElement::Feature { feature: #feature, projection: #projection } }
        }
        "minecraft:list_pool_element" => {
            let elems = required(elem.elements.as_ref(), context, "elements");
            if elems.is_empty() {
                panic!("Field elements must be non-empty in {context}");
            }
            let sub_elements: Vec<TokenStream> = elems
                .iter()
                .enumerate()
                .map(|(index, elem)| gen_element(elem, &format!("{context}.elements[{index}]")))
                .collect();
            let projection = gen_projection(&elem.projection, context);
            quote! { PoolElement::List { elements: vec![#(#sub_elements),*], projection: #projection } }
        }
        other => panic!("Unknown pool element type: {other}"),
    }
}

// ── Main build function ──

pub(crate) fn build(overlay: &DatapackOverlay) -> TokenStream {
    // ── Parse template pools ──

    let mut pools: Vec<(String, PoolJson)> = overlay
        .list_json_registry_ids_with_suffix("worldgen/template_pool")
        .into_iter()
        .map(|(id, content)| {
            let pool: PoolJson = serde_json::from_str(&content)
                .unwrap_or_else(|err| panic!("Failed to parse template pool {id}: {err}"));
            (id, pool)
        })
        .collect();
    pools.sort_by(|a, b| a.0.cmp(&b.0));

    let mut pool_tokens = TokenStream::new();
    for (id, pool) in &pools {
        let key = gen_identifier(id);
        let fallback = gen_identifier(&pool.fallback);

        let elements: Vec<TokenStream> = pool
            .elements
            .iter()
            .enumerate()
            .map(|(index, we)| {
                if we.weight <= 0 {
                    panic!("Template pool {id} element {index} has non-positive weight");
                }
                let elem = gen_element(&we.element, &format!("{id}.elements[{index}]"));
                let weight = we.weight;
                quote! { (#elem, #weight) }
            })
            .collect();

        pool_tokens.extend(quote! {
            TemplatePoolData {
                key: #key,
                fallback: #fallback,
                elements: vec![#(#elements),*],
            },
        });
    }

    // ── Parse structure NBT files ──

    let mut templates: Vec<(String, DatapackFileSource)> = overlay
        .list_structure_template_nbt_with_sources()
        .into_iter()
        .map(|(id, nbt)| (id, nbt.source))
        .collect();
    templates.sort_by(|a, b| a.0.cmp(&b.0));

    let mut template_keys = TokenStream::new();
    let mut file_nbt_match_arms = TokenStream::new();
    let mut zip_sources: BTreeMap<PathBuf, Vec<(String, String)>> = BTreeMap::new();
    for (id, source) in &templates {
        let key = gen_identifier(id);
        template_keys.extend(quote! {
            #key,
        });

        let (namespace, path) = identifier_parts(id, "structure template");
        match source {
            DatapackFileSource::Path(source_path) => {
                let source_path = source_path
                    .canonicalize()
                    .unwrap_or_else(|err| {
                        panic!(
                            "failed to resolve structure template source {}: {err}",
                            source_path.display()
                        )
                    })
                    .to_string_lossy()
                    .replace('\\', "/");
                file_nbt_match_arms.extend(quote! {
                    (#namespace, #path) => Some(include_bytes!(#source_path)),
                });
            }
            DatapackFileSource::Zip {
                zip_path,
                entry_name,
            } => {
                zip_sources
                    .entry(zip_path.clone())
                    .or_default()
                    .push((id.clone(), entry_name.clone()));
            }
        }
    }

    let mut zip_loader_calls = TokenStream::new();
    for (zip_path, entries) in &zip_sources {
        let zip_path = zip_path
            .canonicalize()
            .unwrap_or_else(|err| {
                panic!(
                    "failed to resolve datapack zip source {}: {err}",
                    zip_path.display()
                )
            })
            .to_string_lossy()
            .replace('\\', "/");
        let mappings = entries.iter().map(|(id, entry_name)| {
            let (namespace, path) = identifier_parts(id, "structure template");
            quote! { ((#namespace, #path), #entry_name) }
        });
        zip_loader_calls.extend(quote! {
            load_zip(&mut entries, include_bytes!(#zip_path), &[#(#mappings),*]);
        });
    }

    let pool_count = pools.len();
    let template_count = templates.len();
    let (zip_support, zip_match_arm) = if zip_sources.is_empty() {
        (TokenStream::new(), quote! { _ => None, })
    } else {
        (
            quote! {
                use std::collections::HashMap;
                use std::io::{Cursor, Read};
                use std::sync::LazyLock;

                static ZIP_STRUCTURE_NBT: LazyLock<HashMap<(&'static str, &'static str), &'static [u8]>> = LazyLock::new(|| {
                    let mut entries = HashMap::new();

                    fn load_zip(
                        entries: &mut HashMap<(&'static str, &'static str), &'static [u8]>,
                        zip_bytes: &'static [u8],
                        mappings: &[((&'static str, &'static str), &'static str)],
                    ) {
                        let cursor = Cursor::new(zip_bytes);
                        let mut archive = zip::ZipArchive::new(cursor)
                            .unwrap_or_else(|err| panic!("failed to read embedded datapack zip: {err}"));

                        for ((namespace, path), entry_name) in mappings {
                            let mut entry = archive.by_name(entry_name).unwrap_or_else(|err| {
                                panic!("embedded datapack zip missing structure template {entry_name}: {err}")
                            });
                            let mut bytes = Vec::new();
                            entry.read_to_end(&mut bytes).unwrap_or_else(|err| {
                                panic!("failed to read structure template {entry_name} from embedded datapack zip: {err}")
                            });
                            entries.insert((*namespace, *path), Box::leak(bytes.into_boxed_slice()));
                        }
                    }

                    #zip_loader_calls

                    entries
                });
            },
            quote! { key => ZIP_STRUCTURE_NBT.get(&key).copied(), },
        )
    };

    quote! {
        use crate::template_pool::{
            TemplatePoolData, PoolElement, ProcessorList, Projection, TemplateData,
            load_template_data_from_nbt_keys,
        };
        use crate::structure_processor::{
            PosRuleTestData, ProcessorRuleData, RuleBlockEntityModifierData,
            StructureProcessorKind, StructureRuleTestData,
        };
        use steel_utils::Identifier;

        #zip_support

        /// Returns all vanilla template pools parsed from the datapack.
        pub fn vanilla_template_pools() -> Vec<TemplatePoolData> {
            vec![#pool_tokens]
        }

        /// Returns all vanilla structure templates with their jigsaw data.
        ///
        /// Each entry is (template_key, template_data).
        pub fn vanilla_templates() -> Vec<(Identifier, TemplateData)> {
            load_template_data_from_nbt_keys(&[#template_keys], vanilla_template_nbt_bytes)
        }

        /// Returns the compressed NBT bytes for a structure template bundled at build time.
        pub fn vanilla_template_nbt_bytes(key: &Identifier) -> Option<&'static [u8]> {
            match (key.namespace.as_ref(), key.path.as_ref()) {
                #file_nbt_match_arms
                #zip_match_arm
            }
        }

        /// Number of template pools.
        pub const POOL_COUNT: usize = #pool_count;

        /// Number of structure templates.
        pub const TEMPLATE_COUNT: usize = #template_count;
    }
}
