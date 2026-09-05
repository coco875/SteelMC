use crate::shared_structs::{
    BiomeCondition, BiomeConditionTarget, SpawnConditionEntry, TextComponentJson,
};
use heck::ToShoutySnakeCase;
use proc_macro2::TokenStream;
use proc_macro2::{Ident, Span};
use quote::quote;
use rustc_hash::FxHashMap;
use serde::Deserialize;
use std::fs;
use steel_utils::Identifier;
use steel_utils::datapack_overlay::DatapackOverlay;

pub fn read_json_asset<T: serde::de::DeserializeOwned>(path: &str) -> T {
    println!("cargo:rerun-if-changed={path}");
    let content = fs::read_to_string(path).unwrap_or_else(|e| panic!("Failed to read {path}: {e}"));
    serde_json::from_str(&content).unwrap_or_else(|e| panic!("Failed to parse {path}: {e}"))
}

pub fn sort_contiguous_registry_entries<T>(
    entries: &mut [T],
    path: &str,
    id: impl Fn(&T) -> usize,
) {
    entries.sort_by_key(&id);
    for (expected, entry) in entries.iter().enumerate() {
        let actual = id(entry);
        assert_eq!(
            actual, expected,
            "Expected contiguous ids in {path}: entry at position {expected} has id {actual}"
        );
    }
}

pub fn generate_identifier(resource: &Identifier) -> TokenStream {
    let namespace = resource.namespace.as_ref();
    let path = resource.path.as_ref();
    quote! { Identifier { namespace: Cow::Borrowed(#namespace), path: Cow::Borrowed(#path) } }
}

pub fn generate_static_identifier(resource: &Identifier) -> TokenStream {
    let namespace = resource.namespace.as_ref();
    let path = resource.path.as_ref();
    if namespace == Identifier::VANILLA_NAMESPACE {
        quote! { Identifier::vanilla_static(#path) }
    } else {
        quote! { Identifier::new_static(#namespace, #path) }
    }
}

pub fn generate_static_identifier_from_str(raw: &str, context: &str) -> TokenStream {
    let identifier = Identifier::parse_or_vanilla(raw)
        .unwrap_or_else(|error| panic!("invalid {context} identifier {raw}: {error}"));
    generate_static_identifier(&identifier)
}

pub fn generate_owned_identifier_from_str(raw: &str, context: &str) -> TokenStream {
    let identifier = Identifier::parse_or_vanilla(raw)
        .unwrap_or_else(|error| panic!("invalid {context} identifier {raw}: {error}"));
    let namespace = identifier.namespace.as_ref();
    let path = identifier.path.as_ref();
    if namespace == Identifier::VANILLA_NAMESPACE {
        quote! { Identifier::vanilla(#path.to_string()) }
    } else {
        quote! { Identifier::new(#namespace, #path) }
    }
}

pub fn registry_entry_ident(registry_id: &str) -> Ident {
    Ident::new(
        &registry_id.replace([':', '/'], "_").to_shouty_snake_case(),
        Span::call_site(),
    )
}

pub fn generate_sound_event_ref(resource: &Identifier) -> TokenStream {
    assert_eq!(
        resource.namespace.as_ref(),
        "minecraft",
        "vanilla sound event references must use the minecraft namespace: {resource}"
    );

    let ident = Ident::new(&resource.path.to_shouty_snake_case(), Span::call_site());
    quote! { &crate::sound_events::#ident }
}

pub fn generate_option<T, F>(opt: &Option<T>, f: F) -> TokenStream
where
    F: FnOnce(&T) -> TokenStream,
{
    if let Some(val) = opt {
        let inner = f(val);
        quote! { Some(#inner) }
    } else {
        quote! { None }
    }
}

pub fn generate_vec<T, F>(vec: &[T], f: F) -> TokenStream
where
    F: Fn(&T) -> TokenStream,
{
    let items: Vec<_> = vec.iter().map(f).collect();
    quote! { vec![#(#items),*] }
}

pub fn generate_biome_condition(condition: &BiomeCondition) -> TokenStream {
    let condition_type = condition.condition_type.as_str();
    let biomes = generate_biome_condition_target(&condition.biomes);

    quote! {
        BiomeCondition {
            condition_type: #condition_type,
            biomes: #biomes,
        }
    }
}

fn generate_biome_condition_target(target: &BiomeConditionTarget) -> TokenStream {
    match target {
        BiomeConditionTarget::Tag(tag) => {
            let tag = generate_identifier(tag);
            quote! { crate::shared_structs::BiomeConditionTarget::Tag(#tag) }
        }
        BiomeConditionTarget::Direct(biome) => {
            let biome = generate_identifier(biome);
            quote! { crate::shared_structs::BiomeConditionTarget::Direct(#biome) }
        }
    }
}

pub fn generate_spawn_condition_entry(entry: &SpawnConditionEntry) -> TokenStream {
    let priority = entry.priority;
    let condition = generate_option(&entry.condition, generate_biome_condition);

    quote! {
        SpawnConditionEntry {
            priority: #priority,
            condition: #condition,
        }
    }
}
pub fn generate_text_component(component: &TextComponentJson) -> TokenStream {
    let translate = component.translate.as_str();
    let Some(color) = component.color.as_deref() else {
        return quote! {
            TextComponent::translated(TranslatedMessage::new(#translate, None))
        };
    };
    let color = generate_text_color(color);
    quote! {
        TextComponent {
            content: text_components::content::Content::Translate(
                TranslatedMessage::new(#translate, None),
            ),
            format: text_components::format::Format {
                color: Some(#color),
                font: None,
                bold: None,
                italic: None,
                underlined: None,
                strikethrough: None,
                obfuscated: None,
                shadow_color: None,
            },
            children: vec![],
            interactions: text_components::interactivity::Interactivity::new(),
        }
    }
}

fn generate_text_color(color: &str) -> TokenStream {
    match color {
        "black" => quote! { text_components::format::Color::Black },
        "dark_blue" => quote! { text_components::format::Color::DarkBlue },
        "dark_green" => quote! { text_components::format::Color::DarkGreen },
        "dark_aqua" => quote! { text_components::format::Color::DarkAqua },
        "dark_red" => quote! { text_components::format::Color::DarkRed },
        "dark_purple" => quote! { text_components::format::Color::DarkPurple },
        "gold" => quote! { text_components::format::Color::Gold },
        "gray" => quote! { text_components::format::Color::Gray },
        "dark_gray" => quote! { text_components::format::Color::DarkGray },
        "blue" => quote! { text_components::format::Color::Blue },
        "green" => quote! { text_components::format::Color::Green },
        "aqua" => quote! { text_components::format::Color::Aqua },
        "red" => quote! { text_components::format::Color::Red },
        "light_purple" => quote! { text_components::format::Color::LightPurple },
        "yellow" => quote! { text_components::format::Color::Yellow },
        "white" => quote! { text_components::format::Color::White },
        _ => generate_rgb_text_color(color),
    }
}

fn generate_rgb_text_color(color: &str) -> TokenStream {
    let Some(hex) = color.strip_prefix('#').filter(|hex| {
        hex.len() == 6 && hex.is_ascii() && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    }) else {
        panic!("Unknown text color: {color}");
    };
    let red = u8::from_str_radix(&hex[0..2], 16)
        .unwrap_or_else(|_| panic!("Invalid red channel in text color: {color}"));
    let green = u8::from_str_radix(&hex[2..4], 16)
        .unwrap_or_else(|_| panic!("Invalid green channel in text color: {color}"));
    let blue = u8::from_str_radix(&hex[4..6], 16)
        .unwrap_or_else(|_| panic!("Invalid blue channel in text color: {color}"));
    quote! { text_components::format::Color::Rgb(#red, #green, #blue) }
}

pub fn read_minecraft_datapack_entries<T: serde::de::DeserializeOwned>(
    overlay: &DatapackOverlay,
    subdir: &str,
) -> Vec<(String, T)> {
    let dir = format!("minecraft/{subdir}");
    overlay
        .list_json_relative(&dir)
        .into_iter()
        .map(|(name, content)| {
            let value: T = serde_json::from_str(&content)
                .unwrap_or_else(|e| panic!("Failed to parse {dir}/{name}.json: {e}"));
            (name, value)
        })
        .collect()
}

pub fn read_ordered_minecraft_datapack_entries<T: serde::de::DeserializeOwned>(
    overlay: &DatapackOverlay,
    subdir: &str,
    vanilla_order: &[&str],
) -> Vec<(String, T)> {
    let mut entries: std::collections::BTreeMap<_, _> =
        read_minecraft_datapack_entries(overlay, subdir)
            .into_iter()
            .collect();
    let ordered = vanilla_order
        .iter()
        .map(|name| {
            let value = entries.remove(*name).unwrap_or_else(|| {
                panic!("Missing vanilla {subdir} registry entry: minecraft:{name}")
            });
            ((*name).to_owned(), value)
        })
        .collect();
    assert!(
        entries.is_empty(),
        "Unknown vanilla {subdir} registry entries: {:?}",
        entries.keys().collect::<Vec<_>>()
    );
    ordered
}

pub fn read_variants_from_dir<T: serde::de::DeserializeOwned>(
    overlay: &DatapackOverlay,
    subdir: &str,
) -> Vec<(String, T)> {
    let mut out = read_minecraft_datapack_entries(overlay, subdir);
    let order = extracted_variant_order(subdir);
    out.sort_by_key(|(name, _)| {
        order
            .iter()
            .position(|ordered| ordered == name)
            .unwrap_or_else(|| panic!("Unknown vanilla {subdir} variant in extracted data: {name}"))
    });
    assert_eq!(
        out.len(),
        order.len(),
        "Expected {} vanilla {subdir} variants, got {}",
        order.len(),
        out.len()
    );
    out
}

#[derive(Deserialize)]
struct ExtractedVariantRegistryEntry {
    id: usize,
    key: String,
}

fn extracted_variant_order(subdir: &str) -> Vec<String> {
    const ASSET: &str = "build_assets/entity_variant_registries.json";

    let mut registries: FxHashMap<String, Vec<ExtractedVariantRegistryEntry>> =
        read_json_asset(ASSET);
    let mut entries = registries
        .remove(subdir)
        .unwrap_or_else(|| panic!("Missing vanilla {subdir} registry in {ASSET}"));
    sort_contiguous_registry_entries(&mut entries, ASSET, |entry| entry.id);

    entries
        .into_iter()
        .map(|entry| {
            let Some(key) = entry.key.strip_prefix("minecraft:") else {
                panic!(
                    "Expected vanilla {subdir} registry key in {ASSET}, got {}",
                    entry.key
                );
            };
            key.to_owned()
        })
        .collect()
}
