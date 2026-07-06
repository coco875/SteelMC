use crate::generator_functions::read_minecraft_datapack_entries;
use heck::ToShoutySnakeCase;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
pub struct WorldClockJson {}

pub(crate) fn build(overlay: &steel_utils::datapack_overlay::DatapackOverlay) -> TokenStream {
    let world_clocks: Vec<String> =
        read_minecraft_datapack_entries::<WorldClockJson>(overlay, "world_clock")
            .into_iter()
            .map(|(name, _)| name)
            .collect();

    let mut stream = TokenStream::new();

    stream.extend(quote! {
        use crate::world_clock::{WorldClock, WorldClockRegistry};
        use steel_utils::Identifier;
    });

    // Generate static world_clock definitions
    let mut register_stream = TokenStream::new();
    for world_clock_name in &world_clocks {
        let world_clock_ident =
            Ident::new(&world_clock_name.to_shouty_snake_case(), Span::call_site());
        let world_clock_name_str = world_clock_name.clone();

        let key = quote! { Identifier::vanilla_static(#world_clock_name_str) };

        stream.extend(quote! {
            pub static #world_clock_ident: WorldClock = WorldClock {
                key: #key,
            };
        });

        register_stream.extend(quote! {
            registry.register(&#world_clock_ident);
        });
    }

    stream.extend(quote! {
        pub fn register_world_clocks(registry: &mut WorldClockRegistry) {
            #register_stream
        }
    });

    stream
}
