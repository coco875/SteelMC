use crate::generator_functions::{generate_text_component, read_minecraft_datapack_entries};
use crate::shared_structs::TextComponentJson;
use heck::ToShoutySnakeCase;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum DialogJson {
    #[serde(rename = "minecraft:dialog_list")]
    DialogList(DialogListJson),
    #[serde(rename = "minecraft:server_links")]
    ServerLinks(ServerLinksJson),
}

#[derive(Deserialize, Debug)]
pub struct DialogListJson {
    button_width: i32,
    columns: i32,
    dialogs: String,
    exit_action: ExitActionJson,
    external_title: TextComponentJson,
    title: TextComponentJson,
}

#[derive(Deserialize, Debug)]
pub struct ServerLinksJson {
    button_width: i32,
    columns: i32,
    exit_action: ExitActionJson,
    external_title: TextComponentJson,
    title: TextComponentJson,
}

#[derive(Deserialize, Debug)]
pub struct ExitActionJson {
    label: TextComponentJson,
    width: i32,
}

fn generate_exit_action(action: &ExitActionJson) -> TokenStream {
    let label = generate_text_component(&action.label);
    let width = action.width;
    quote! {
        ExitAction {
            label: #label,
            width: #width,
        }
    }
}

pub(crate) fn build(overlay: &steel_utils::datapack_overlay::DatapackOverlay) -> TokenStream {
    let dialogs: Vec<(String, DialogJson)> = read_minecraft_datapack_entries(overlay, "dialog");

    let mut stream = TokenStream::new();

    stream.extend(quote! {
        use crate::dialog::{Dialog, DialogVariant, DialogRegistry, ExitAction};
        use steel_utils::Identifier;
        use text_components::{TextComponent, translation::TranslatedMessage};
    });

    // Generate static dialog definitions
    let mut register_stream = TokenStream::new();
    for (dialog_name, dialog) in &dialogs {
        let dialog_ident = Ident::new(&dialog_name.to_shouty_snake_case(), Span::call_site());
        let dialog_name_str = dialog_name.clone();

        let key = quote! { Identifier::vanilla_static(#dialog_name_str) };

        match dialog {
            DialogJson::DialogList(dialog_list) => {
                let button_width = dialog_list.button_width;
                let columns = dialog_list.columns;
                let dialogs_ref = dialog_list.dialogs.as_str();
                let exit_action = generate_exit_action(&dialog_list.exit_action);
                let external_title = generate_text_component(&dialog_list.external_title);
                let title = generate_text_component(&dialog_list.title);

                stream.extend(quote! {
                    pub static #dialog_ident: Dialog = Dialog {
                        key: #key,
                        button_width: #button_width,
                        columns: #columns,
                        exit_action: #exit_action,
                        external_title: #external_title,
                        title: #title,
                        variant: DialogVariant::DialogList { dialogs: #dialogs_ref },
                    };
                });
            }
            DialogJson::ServerLinks(server_links) => {
                let button_width = server_links.button_width;
                let columns = server_links.columns;
                let exit_action = generate_exit_action(&server_links.exit_action);
                let external_title = generate_text_component(&server_links.external_title);
                let title = generate_text_component(&server_links.title);

                stream.extend(quote! {
                    pub static #dialog_ident: Dialog = Dialog {
                        key: #key,
                        button_width: #button_width,
                        columns: #columns,
                        exit_action: #exit_action,
                        external_title: #external_title,
                        title: #title,
                        variant: DialogVariant::ServerLinks,
                    };
                });
            }
        }
        register_stream.extend(quote! {
            registry.register(&#dialog_ident);
        });
    }

    stream.extend(quote! {
        pub fn register_dialogs(registry: &mut DialogRegistry) {
            #register_stream
        }
    });

    stream
}
