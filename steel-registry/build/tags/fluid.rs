use proc_macro2::TokenStream;

pub(crate) fn build(overlay: &steel_utils::datapack_overlay::DatapackOverlay) -> TokenStream {
    super::common::build_simple_tags(overlay, "fluid", "fluid", "FluidRegistry")
}
