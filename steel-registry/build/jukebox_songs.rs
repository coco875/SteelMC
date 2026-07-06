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
pub struct JukeboxSongJson {
    sound_event: Identifier,
    description: TextComponentJson,
    length_in_seconds: f32,
    comparator_output: i32,
}

pub(crate) fn build(overlay: &steel_utils::datapack_overlay::DatapackOverlay) -> TokenStream {
    let jukebox_songs: Vec<(String, JukeboxSongJson)> =
        read_minecraft_datapack_entries(overlay, "jukebox_song");

    let mut stream = TokenStream::new();

    stream.extend(quote! {
        use crate::jukebox_song::{
            JukeboxSong, JukeboxSongRegistry,
        };
        use steel_utils::Identifier;
        use text_components::{TextComponent, translation::TranslatedMessage};
    });

    // Generate static jukebox song definitions
    let mut register_stream = TokenStream::new();
    for (jukebox_song_name, jukebox_song) in &jukebox_songs {
        // Handle special case where song name is a number (e.g., "13" -> "MUSIC_DISC_13")
        let jukebox_song_ident = if jukebox_song_name.chars().next().unwrap().is_ascii_digit() {
            Ident::new(
                &format!("MUSIC_DISC_{}", jukebox_song_name.to_shouty_snake_case()),
                Span::call_site(),
            )
        } else {
            Ident::new(&jukebox_song_name.to_shouty_snake_case(), Span::call_site())
        };
        let jukebox_song_name_str = jukebox_song_name.clone();

        let key = quote! { Identifier::vanilla_static(#jukebox_song_name_str) };
        let sound_event = generate_sound_event_ref(&jukebox_song.sound_event);
        let description = generate_text_component(&jukebox_song.description);
        let length_in_seconds = jukebox_song.length_in_seconds;
        let comparator_output = jukebox_song.comparator_output;

        stream.extend(quote! {
            pub static #jukebox_song_ident: JukeboxSong = JukeboxSong {
                key: #key,
                sound_event: #sound_event,
                description: #description,
                length_in_seconds: #length_in_seconds,
                comparator_output: #comparator_output,
            };
        });

        register_stream.extend(quote! {
            registry.register(&#jukebox_song_ident);
        });
    }

    stream.extend(quote! {
        pub fn register_jukebox_songs(registry: &mut JukeboxSongRegistry) {
            #register_stream
        }
    });

    stream
}
