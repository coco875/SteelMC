use crate::generator_functions::generate_static_identifier_from_str;
use heck::{ToShoutySnakeCase, ToUpperCamelCase};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Deserialize;
use steel_utils::Identifier;
use steel_utils::datapack_overlay::DatapackOverlay;

#[derive(Deserialize, Debug)]
pub struct TagJson {
    #[serde(default)]
    pub replace: bool,
    pub values: Vec<TagValueJson>,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum TagValueJson {
    Plain(String),
    Optional { id: String, required: Option<bool> },
}

pub fn tag_value_strings(values: Vec<TagValueJson>) -> Vec<String> {
    values
        .into_iter()
        .filter_map(|value| match value {
            TagValueJson::Plain(id) => Some(id),
            TagValueJson::Optional { id, required } => {
                if required == Some(false) {
                    None
                } else {
                    Some(id)
                }
            }
        })
        .collect()
}

pub fn load_and_merge_tags(
    overlay: &DatapackOverlay,
    tag_subpath: &str,
) -> FxHashMap<String, Vec<String>> {
    let suffix = format!("tags/{tag_subpath}");
    let mut all_tags: FxHashMap<String, Vec<String>> = FxHashMap::default();
    for (tag_id, contents) in overlay.list_json_registry_ids_with_suffix_all_layers(&suffix) {
        let mut values = Vec::new();
        for content in contents {
            let tag: TagJson = serde_json::from_str(&content)
                .unwrap_or_else(|error| panic!("Failed to parse tag {tag_id}: {error}"));
            if tag.replace {
                values.clear();
            }
            let layer_values = tag_value_strings(tag.values);
            for val in layer_values {
                if !values.contains(&val) {
                    values.push(val);
                }
            }
        }
        all_tags.insert(tag_id, values);
    }
    all_tags
}

fn lookup_tag_values<'a>(
    tag_name: &str,
    all_tags: &'a FxHashMap<String, Vec<String>>,
) -> Option<&'a Vec<String>> {
    let mut candidates = vec![tag_name.to_string()];
    if tag_name.contains(':') {
        let identifier = Identifier::parse_or_vanilla(tag_name)
            .unwrap_or_else(|error| panic!("invalid tag identifier {tag_name}: {error}"));
        if identifier.namespace == Identifier::VANILLA_NAMESPACE {
            candidates.push(identifier.path.into_owned());
        } else {
            candidates.truncate(1);
        }
    } else {
        candidates.push(format!("minecraft:{tag_name}"));
    }

    for candidate in candidates {
        if let Some(values) = all_tags.get(&candidate) {
            return Some(values);
        }
    }

    None
}

fn tag_static_ident(tag_id: &str) -> Ident {
    let identifier = Identifier::parse_or_vanilla(tag_id)
        .unwrap_or_else(|error| panic!("invalid tag identifier {tag_id}: {error}"));
    let name = if identifier.namespace == Identifier::VANILLA_NAMESPACE {
        identifier.path.into_owned()
    } else {
        tag_id.replace([':', '/'], "_")
    };
    Ident::new(&name.to_shouty_snake_case(), Span::call_site())
}

/// Resolves tag references recursively and returns a flattened, deduplicated list of keys.
pub fn resolve_tag(
    tag_name: &str,
    all_tags: &FxHashMap<String, Vec<String>>,
    resolved_cache: &mut FxHashMap<String, Vec<String>>,
    visiting: &mut Vec<String>,
) -> Vec<String> {
    if let Some(cached) = resolved_cache.get(tag_name) {
        return cached.clone();
    }

    if visiting.contains(&tag_name.to_string()) {
        panic!("Circular tag dependency detected: {:?}", visiting);
    }

    visiting.push(tag_name.to_string());

    let values = lookup_tag_values(tag_name, all_tags)
        .unwrap_or_else(|| panic!("Tag not found: {}", tag_name));

    let mut resolved = Vec::new();

    for value in values {
        if let Some(nested_tag) = value.strip_prefix('#') {
            let nested_values = resolve_tag(nested_tag, all_tags, resolved_cache, visiting);
            resolved.extend(nested_values);
        } else {
            let key = value.strip_prefix("minecraft:").unwrap_or(value);
            resolved.push(key.to_string());
        }
    }

    visiting.pop();

    let mut seen = FxHashSet::default();
    resolved.retain(|x| seen.insert(x.clone()));

    resolved_cache.insert(tag_name.to_string(), resolved.clone());
    resolved
}

/// Resolves all tags and returns them sorted by name.
pub fn resolve_all_tags(all_tags: &FxHashMap<String, Vec<String>>) -> Vec<(String, Vec<String>)> {
    let mut resolved_tags: FxHashMap<String, Vec<String>> = FxHashMap::default();
    let mut resolved_cache = FxHashMap::default();

    for tag_name in all_tags.keys() {
        let mut visiting = Vec::new();
        let resolved = resolve_tag(tag_name, all_tags, &mut resolved_cache, &mut visiting);
        resolved_tags.insert(tag_name.clone(), resolved);
    }

    let mut sorted: Vec<_> = resolved_tags.into_iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    sorted
}

/// Builds a complete tag module for a vanilla-only registry.
///
/// Generates: static tag arrays, `pub const` tag identifiers, and a register function.
///
/// - `tag_subdir`: directory under `tags/` (e.g., `"damage_type"`)
/// - `registry_module`: crate module name (e.g., `"damage_type"`)
/// - `registry_type`: type name (e.g., `"DamageTypeRegistry"`)
/// - `register_fn`: function name (e.g., `"register_damage_type_tags"`)
pub fn build_simple_tags(
    overlay: &DatapackOverlay,
    tag_subdir: &str,
    registry_module: &str,
    registry_type: &str,
) -> TokenStream {
    let all_tags = load_and_merge_tags(overlay, tag_subdir);
    let sorted_tags = resolve_all_tags(&all_tags);

    let registry_module_ident = Ident::new(registry_module, Span::call_site());
    let registry_type_ident = Ident::new(registry_type, Span::call_site());
    let register_fn_ident = Ident::new(
        &format!("register_{}_tags", registry_module),
        Span::call_site(),
    );
    let tag_category_ident = Ident::new(
        &format!("{}Tag", registry_module.to_upper_camel_case()),
        Span::call_site(),
    );

    let mut stream = TokenStream::new();

    stream.extend(quote! {
        use crate::#registry_module_ident::#registry_type_ident;
        use crate::TaggedRegistryExt;
        use steel_utils::Identifier;
    });

    let mut static_arrays = TokenStream::new();
    let mut const_identifiers = TokenStream::new();
    let mut register_stream = TokenStream::new();

    for (tag_id, entries) in &sorted_tags {
        let tag_list_ident = Ident::new(
            &format!(
                "{}_TAG_LIST",
                tag_id.replace([':', '/'], "_").to_shouty_snake_case()
            ),
            Span::call_site(),
        );
        let tag_ident = tag_static_ident(tag_id);

        let entry_strs = entries.iter().map(|s| s.as_str());
        let tag_key = generate_static_identifier_from_str(tag_id, "tag");

        static_arrays.extend(quote! {
            static #tag_list_ident: &[&str] = &[#(#entry_strs),*];
        });

        const_identifiers.extend(quote! {
            pub const #tag_ident: Identifier = #tag_key;
        });

        register_stream.extend(quote! {
            registry.register_tag(Self::#tag_ident, #tag_list_ident);
        });
    }

    stream.extend(quote! {
        #static_arrays

        pub struct #tag_category_ident {}
        impl #tag_category_ident {
            #const_identifiers
            pub fn #register_fn_ident(registry: &mut #registry_type_ident) {
               #register_stream
            }

        }
    });

    stream
}

/// Like [`build_overlay_tags`], but merges additional tag maps after overlay load.
///
/// Extra entries override overlay tags with the same key (used for Fabric `c:` tags).
pub fn build_overlay_tags_with_extras(
    overlay: &DatapackOverlay,
    tag_subpath: &str,
    registry_module: &str,
    registry_type: &str,
    tag_category_ident: &str,
    register_fn_ident: &str,
    extra_tags: FxHashMap<String, Vec<String>>,
) -> TokenStream {
    let mut all_tags = load_and_merge_tags(overlay, tag_subpath);
    all_tags.extend(extra_tags);

    let sorted_tags = resolve_all_tags(&all_tags);

    let registry_module_ident = Ident::new(registry_module, Span::call_site());
    let registry_type_ident = Ident::new(registry_type, Span::call_site());
    let register_fn = Ident::new(register_fn_ident, Span::call_site());
    let tag_category = Ident::new(tag_category_ident, Span::call_site());

    let mut stream = TokenStream::new();
    stream.extend(quote! {
        use crate::#registry_module_ident::#registry_type_ident;
        use crate::TaggedRegistryExt;
        use steel_utils::Identifier;
    });

    let mut static_arrays = TokenStream::new();
    let mut const_identifiers = TokenStream::new();
    let mut register_stream = TokenStream::new();

    for (tag_id, entries) in &sorted_tags {
        let tag_list_ident = Ident::new(
            &format!(
                "{}_TAG_LIST",
                tag_id.replace([':', '/'], "_").to_shouty_snake_case()
            ),
            Span::call_site(),
        );
        let tag_ident = tag_static_ident(tag_id);
        let tag_key = generate_static_identifier_from_str(tag_id, "tag");
        let entry_strs = entries.iter().map(|s| s.as_str());

        static_arrays.extend(quote! {
            static #tag_list_ident: &[&str] = &[#(#entry_strs),*];
        });

        const_identifiers.extend(quote! {
            pub const #tag_ident: Identifier = #tag_key;
        });

        register_stream.extend(quote! {
            registry.register_tag(Self::#tag_ident, #tag_list_ident);
        });
    }

    stream.extend(quote! {
        #static_arrays

        pub struct #tag_category {}
        impl #tag_category {
            #const_identifiers
            pub fn #register_fn(registry: &mut #registry_type_ident) {
               #register_stream
            }

        }
    });

    stream
}

/// Builds a tag module from the datapack overlay, preserving each namespace as its own registry id.
pub fn build_overlay_tags(
    overlay: &DatapackOverlay,
    tag_subpath: &str,
    registry_module: &str,
    registry_type: &str,
) -> TokenStream {
    let all_tags = load_and_merge_tags(overlay, tag_subpath);

    let sorted_tags = resolve_all_tags(&all_tags);

    let registry_module_ident = Ident::new(registry_module, Span::call_site());
    let registry_type_ident = Ident::new(registry_type, Span::call_site());
    let register_fn_ident = Ident::new(
        &format!("register_{registry_module}_tags"),
        Span::call_site(),
    );
    let tag_category_ident = Ident::new(
        &format!("{}Tag", registry_module.to_upper_camel_case()),
        Span::call_site(),
    );

    let mut stream = TokenStream::new();
    stream.extend(quote! {
        use crate::#registry_module_ident::#registry_type_ident;
        use crate::TaggedRegistryExt;
        use steel_utils::Identifier;
    });

    let mut static_arrays = TokenStream::new();
    let mut const_identifiers = TokenStream::new();
    let mut register_stream = TokenStream::new();

    for (tag_id, entries) in &sorted_tags {
        let tag_list_ident = Ident::new(
            &format!(
                "{}_TAG_LIST",
                tag_id.replace([':', '/'], "_").to_shouty_snake_case()
            ),
            Span::call_site(),
        );
        let tag_ident = tag_static_ident(tag_id);
        let tag_key = generate_static_identifier_from_str(tag_id, "tag");
        let entry_strs = entries.iter().map(|s| s.as_str());

        static_arrays.extend(quote! {
            static #tag_list_ident: &[&str] = &[#(#entry_strs),*];
        });

        const_identifiers.extend(quote! {
            pub const #tag_ident: Identifier = #tag_key;
        });

        register_stream.extend(quote! {
            registry.register_tag(Self::#tag_ident, #tag_list_ident);
        });
    }

    stream.extend(quote! {
        #static_arrays

        pub struct #tag_category_ident {}
        impl #tag_category_ident {
            #const_identifiers
            pub fn #register_fn_ident(registry: &mut #registry_type_ident) {
               #register_stream
            }

        }
    });

    stream
}
