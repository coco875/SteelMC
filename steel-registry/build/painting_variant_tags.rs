use proc_macro2::TokenStream;

pub(crate) fn build(overlay: &steel_utils::datapack_overlay::DatapackOverlay) -> TokenStream {
    super::tag_utils::build_simple_tags(
        overlay,
        "painting_variant",
        "painting_variant",
        "PaintingVariantRegistry",
    )
}
