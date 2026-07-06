#![expect(
    clippy::unwrap_used,
    reason = "build script must fail immediately on invalid extracted instrument data"
)]

use crate::generator_functions::{
    generate_sound_event_ref, generate_text_component, read_minecraft_datapack_entries,
};
use crate::shared_structs::TextComponentJson;
use heck::ToShoutySnakeCase;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Deserialize;
use steel_utils::Identifier;

#[derive(Deserialize, Debug)]
pub struct InstrumentJson {
    sound_event: Identifier,
    use_duration: f32,
    range: f32,
    description: TextComponentJson,
}

pub(crate) fn build(overlay: &steel_utils::datapack_overlay::DatapackOverlay) -> TokenStream {
    let instruments: Vec<(String, InstrumentJson)> =
        read_minecraft_datapack_entries(overlay, "instrument");

    let mut stream = TokenStream::new();

    stream.extend(quote! {
        use crate::instrument::{
            Instrument, InstrumentRegistry, InstrumentValue,
        };
        use crate::sound_event::SoundEventHolder;
        use steel_utils::Identifier;
        use text_components::{TextComponent, translation::TranslatedMessage};
    });

    // Generate static instrument definitions
    let mut register_stream = TokenStream::new();
    for (instrument_name, instrument) in &instruments {
        let instrument_ident =
            Ident::new(&instrument_name.to_shouty_snake_case(), Span::call_site());
        let instrument_name_str = instrument_name.clone();

        let key = quote! { Identifier::vanilla_static(#instrument_name_str) };
        let sound_event = generate_sound_event_ref(&instrument.sound_event);
        let use_duration = instrument.use_duration;
        let range = instrument.range;
        assert!(
            use_duration > 0.0 && use_duration <= f32::MAX,
            "instrument {instrument_name} use_duration must be positive"
        );
        assert!(
            range > 0.0 && range <= f32::MAX,
            "instrument {instrument_name} range must be positive"
        );
        let description = generate_text_component(&instrument.description);

        stream.extend(quote! {
            pub static #instrument_ident: Instrument = Instrument::new(
                #key,
                InstrumentValue::from_validated_parts(
                    SoundEventHolder::registry(#sound_event),
                    #use_duration,
                    #range,
                    #description,
                ),
            );
        });
        let instrument_ident =
            Ident::new(&instrument_name.to_shouty_snake_case(), Span::call_site());
        register_stream.extend(quote! {
            registry.register(&#instrument_ident);
        });
    }

    stream.extend(quote! {
        pub fn register_instruments(registry: &mut InstrumentRegistry) {
            #register_stream
        }
    });

    stream
}
