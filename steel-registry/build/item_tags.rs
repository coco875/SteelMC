use rustc_hash::FxHashMap;
use std::{fs, path::Path};

use proc_macro2::TokenStream;
use serde::Deserialize;

use super::tag_utils;

#[derive(Deserialize)]
struct TagFile {
    item: FxHashMap<String, Vec<String>>,
}

fn read_all_fabric_tags(tag_file: &str) -> FxHashMap<String, Vec<String>> {
    if fs::exists(tag_file).unwrap_or(false)
        && Path::new(tag_file).is_file()
        && let Ok(content) = fs::read_to_string(tag_file)
    {
        let tag: TagFile = serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("Failed to parse {tag_file}: {e}"));
        return tag.item;
    }
    FxHashMap::default()
}

pub(crate) fn build(overlay: &steel_utils::datapack_overlay::DatapackOverlay) -> TokenStream {
    tag_utils::build_overlay_tags_with_extras(
        overlay,
        "item",
        "items",
        "ItemRegistry",
        "ItemTag",
        "register_item_tags",
        read_all_fabric_tags("build_assets/tags.json"),
    )
}
