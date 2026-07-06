use proc_macro2::TokenStream;

pub(crate) fn build(overlay: &steel_utils::datapack_overlay::DatapackOverlay) -> TokenStream {
    super::tag_utils::build_overlay_tags(
        overlay,
        "worldgen/structure",
        "structure",
        "StructureRegistry",
    )
}
