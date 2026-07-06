//! Build script for generating vanilla loot table definitions.

use std::str::FromStr;

use heck::{ToShoutySnakeCase, ToSnakeCase};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use rustc_hash::FxHashMap;
use serde::Deserialize;
use steel_utils::Identifier;
use steel_utils::datapack_overlay::DatapackOverlay;

mod conditions;
mod entries;
mod functions;
mod values;

use conditions::generate_condition;
use entries::generate_pool;
use functions::generate_function;
use values::{
    generate_damage_source_predicate, generate_enchantment_options, generate_entity_predicate,
    generate_instrument_options, generate_location_predicate, generate_loot_context_entity,
    generate_loot_type, generate_number_provider, generate_tool_predicate,
};

/// A number provider can be a constant number or an object with type.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum NumberProviderJson {
    Constant(f32),
    Object {
        #[serde(rename = "type")]
        provider_type: String,
        #[serde(default)]
        value: Option<f32>,
        #[serde(default)]
        min: Option<f32>,
        #[serde(default)]
        max: Option<f32>,
        #[serde(default)]
        n: Option<f32>, // Can be float in JSON, convert to i32 later
        #[serde(default)]
        p: Option<f32>,
    },
}

impl Default for NumberProviderJson {
    fn default() -> Self {
        Self::Constant(1.0)
    }
}

/// Enchantment options can be a tag string or list of enchantment IDs.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum EnchantmentOptionsJson {
    Tag(String),
    List(Vec<String>),
}

/// Loot table value can be a string reference or inline loot table.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum LootTableValueJson {
    Reference(String),
    Inline(Box<InlineLootTableJson>),
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct InlineLootTableJson {
    #[serde(default)]
    pools: Vec<LootPoolJson>,
}

/// Enchanted chance can be a constant or linear formula.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum EnchantedChanceJson {
    Constant(f32),
    Formula {
        #[serde(rename = "type")]
        formula_type: String,
        #[serde(default)]
        value: Option<f32>,
        #[serde(default)]
        base: Option<f32>,
        #[serde(default)]
        per_level_above_first: Option<f32>,
    },
}

/// Limit count can be an integer or object with min/max.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum LimitJson {
    Integer(i32),
    Object {
        #[serde(default)]
        min: Option<f32>,
        #[serde(default)]
        max: Option<f32>,
    },
}

/// Block state property value can be string or object with min/max.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum PropertyValueJson {
    Exact(String),
    Range {
        min: Option<String>,
        max: Option<String>,
    },
}

/// Stew effect entry.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct StewEffectJson {
    #[serde(rename = "type")]
    effect_type: String,
    #[serde(default)]
    duration: NumberProviderJson,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct LootTableJson {
    #[serde(rename = "type")]
    loot_type: Option<String>,
    #[serde(default)]
    pools: Vec<LootPoolJson>,
    #[serde(default)]
    functions: Vec<LootFunctionJson>,
    #[serde(default)]
    random_sequence: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct LootPoolJson {
    #[serde(default)]
    rolls: NumberProviderJson,
    #[serde(default)]
    bonus_rolls: f32,
    #[serde(default)]
    entries: Vec<LootEntryJson>,
    #[serde(default)]
    conditions: Vec<LootConditionJson>,
    #[serde(default)]
    functions: Vec<LootFunctionJson>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct LootEntryJson {
    #[serde(rename = "type")]
    entry_type: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    value: Option<LootTableValueJson>,
    #[serde(default = "default_weight")]
    weight: i32,
    #[serde(default)]
    quality: i32,
    #[serde(default)]
    expand: bool,
    #[serde(default)]
    conditions: Vec<LootConditionJson>,
    #[serde(default)]
    functions: Vec<LootFunctionJson>,
    #[serde(default)]
    children: Vec<LootEntryJson>,
}

const fn default_weight() -> i32 {
    1
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct LootConditionJson {
    condition: String,
    // block_state_property
    #[serde(default)]
    block: Option<String>,
    #[serde(default)]
    properties: Option<FxHashMap<String, PropertyValueJson>>,
    // match_tool / entity_properties predicate
    #[serde(default)]
    predicate: Option<PredicateJson>,
    // table_bonus / random_chance_with_enchanted_bonus
    #[serde(default)]
    enchantment: Option<String>,
    #[serde(default)]
    chances: Option<Vec<f32>>,
    // inverted
    #[serde(default)]
    term: Option<Box<LootConditionJson>>,
    // any_of / all_of
    #[serde(default)]
    terms: Option<Vec<LootConditionJson>>,
    // random_chance
    #[serde(default)]
    chance: Option<f32>,
    // random_chance_with_enchanted_bonus
    #[serde(default)]
    unenchanted_chance: Option<f32>,
    #[serde(default)]
    enchanted_chance: Option<EnchantedChanceJson>,
    // entity_properties / damage_source_properties
    #[serde(default)]
    entity: Option<String>,
    // location_check
    #[serde(default, rename = "offsetX")]
    offset_x: Option<i32>,
    #[serde(default, rename = "offsetY")]
    offset_y: Option<i32>,
    #[serde(default, rename = "offsetZ")]
    offset_z: Option<i32>,
}

/// Predicate can be a tool predicate (`match_tool`), location predicate (`location_check`),
/// entity predicate (`entity_properties`), or damage source predicate. We parse these specifically.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
#[expect(clippy::large_enum_variant)]
enum PredicateJson {
    Tool(ToolPredicateJson),
    Location(LocationPredicateJson),
    DamageSource(DamageSourcePredicateJson),
    Entity(EntityPredicateJson),
}

/// Damage source predicate for `damage_source_properties` condition.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct DamageSourcePredicateJson {
    #[serde(default)]
    tags: Option<Vec<DamageTagPredicateJson>>,
    #[serde(default)]
    source_entity: Option<EntityPredicateJson>,
    #[serde(default)]
    direct_entity: Option<EntityPredicateJson>,
    #[serde(default)]
    is_direct: Option<bool>,
}

/// A tag check for damage source.
#[derive(Deserialize, Debug, Clone)]
struct DamageTagPredicateJson {
    id: String,
    #[serde(default = "default_true")]
    expected: bool,
}

const fn default_true() -> bool {
    true
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct LocationPredicateJson {
    #[serde(default)]
    block: Option<BlockPredicateJson>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct BlockPredicateJson {
    #[serde(default)]
    blocks: Option<String>,
    #[serde(default)]
    state: Option<FxHashMap<String, String>>,
}

/// Entity predicate - can have many fields
#[derive(Deserialize, Debug, Clone)]
struct EntityPredicateJson {
    #[serde(rename = "type", alias = "minecraft:entity_type", default)]
    entity_type: Option<String>,
    #[serde(alias = "minecraft:flags", default)]
    flags: Option<EntityFlagsJson>,
    #[serde(alias = "minecraft:equipment", default)]
    equipment: Option<EntityEquipmentJson>,
    /// Entity data components (`minecraft:components`). Only `sheep/color` and
    /// `chicken/variant` are modeled.
    #[serde(rename = "minecraft:components", default)]
    components: Option<EntityComponentsJson>,
    /// Type-specific predicates. `minecraft:type_specific/sheep` is a single
    /// registry-style flat key, not a nested object.
    #[serde(rename = "minecraft:type_specific/sheep", default)]
    sheep_type_specific: Option<SheepTypeSpecificJson>,
}

/// Entity data-component predicates (`minecraft:components`).
#[derive(Deserialize, Debug, Clone)]
struct EntityComponentsJson {
    #[serde(rename = "minecraft:sheep/color", default)]
    sheep_color: Option<String>,
    #[serde(rename = "minecraft:chicken/variant", default)]
    chicken_variant: Option<String>,
}

/// Type-specific entity predicates (`minecraft:type_specific/sheep`).
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct SheepTypeSpecificJson {
    #[serde(default)]
    sheared: Option<bool>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct EntityFlagsJson {
    #[serde(default)]
    is_on_fire: Option<bool>,
    #[serde(default)]
    is_sneaking: Option<bool>,
    #[serde(default)]
    is_sprinting: Option<bool>,
    #[serde(default)]
    is_swimming: Option<bool>,
    #[serde(default)]
    is_baby: Option<bool>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct EntityEquipmentJson {
    #[serde(default)]
    mainhand: Option<EquipmentSlotJson>,
    #[serde(default)]
    offhand: Option<EquipmentSlotJson>,
    #[serde(default)]
    head: Option<EquipmentSlotJson>,
    #[serde(default)]
    chest: Option<EquipmentSlotJson>,
    #[serde(default)]
    legs: Option<EquipmentSlotJson>,
    #[serde(default)]
    feet: Option<EquipmentSlotJson>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct EquipmentSlotJson {
    #[serde(default)]
    items: Option<String>,
    #[serde(default)]
    predicates: Option<ToolPredicatesJson>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct ToolPredicateJson {
    #[serde(default)]
    items: Option<String>,
    #[serde(default)]
    predicates: Option<ToolPredicatesJson>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct ToolPredicatesJson {
    #[serde(rename = "minecraft:enchantments", default)]
    enchantments: Option<Vec<EnchantmentPredicateJson>>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct EnchantmentPredicateJson {
    #[serde(default)]
    enchantments: Option<String>,
    #[serde(default)]
    levels: Option<LevelRangeJson>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct LevelRangeJson {
    #[serde(default)]
    min: Option<i32>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct LootFunctionJson {
    function: String,
    #[serde(default)]
    count: Option<NumberProviderJson>,
    #[serde(default)]
    add: bool,
    // apply_bonus
    #[serde(default)]
    enchantment: Option<String>,
    #[serde(default)]
    formula: Option<String>,
    #[serde(default)]
    parameters: Option<BonusParametersJson>,
    // limit_count / enchanted_count_increase limit
    #[serde(default)]
    limit: Option<LimitJson>,
    // set_damage
    #[serde(default)]
    damage: Option<NumberProviderJson>,
    // enchant_randomly / enchant_with_levels / set_instrument
    #[serde(default)]
    options: Option<EnchantmentOptionsJson>,
    // enchant_with_levels
    #[serde(default)]
    levels: Option<NumberProviderJson>,
    // copy_components
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    include: Option<Vec<String>>,
    // copy_state
    #[serde(default)]
    block: Option<String>,
    // copy_state properties
    #[serde(default)]
    properties: Option<Vec<String>>,
    // set_components (keep as raw value since it's complex NBT)
    #[serde(default)]
    components: Option<serde_json::Value>,
    // furnace_smelt
    #[serde(default)]
    use_input_count: Option<bool>,
    // exploration_map
    #[serde(default)]
    destination: Option<String>,
    #[serde(default)]
    decoration: Option<String>,
    #[serde(default)]
    zoom: Option<i32>,
    #[serde(default)]
    skip_existing_chunks: Option<bool>,
    // set_name (keep as raw value for text component)
    #[serde(default)]
    name: Option<serde_json::Value>,
    #[serde(default)]
    target: Option<String>,
    // set_ominous_bottle_amplifier
    #[serde(default)]
    amplifier: Option<NumberProviderJson>,
    // set_potion
    #[serde(default)]
    id: Option<String>,
    // set_stew_effect
    #[serde(default)]
    effects: Option<Vec<StewEffectJson>>,
    // set_enchantments
    #[serde(default)]
    enchantments: Option<FxHashMap<String, NumberProviderJson>>,
    // conditions for conditional functions
    #[serde(default)]
    conditions: Option<Vec<LootConditionJson>>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct BonusParametersJson {
    #[serde(rename = "bonusMultiplier", default)]
    bonus_multiplier: Option<i32>,
    #[serde(default)]
    extra: Option<i32>,
    #[serde(default)]
    probability: Option<f32>,
}

struct LootTableData {
    /// Full registry id like `minecraft:blocks/acacia_button`.
    registry_id: String,
    /// Category bucket for generated convenience structs.
    category_key: String,
    /// Field name within the category struct.
    field_name: String,
    /// Rust identifier like `BLOCKS_ACACIA_BUTTON`.
    const_ident: Ident,
    /// The loot type as a `TokenStream`.
    loot_type: TokenStream,
    /// Generated pools.
    pools: Vec<TokenStream>,
    /// Table-level functions.
    functions: Vec<TokenStream>,
    /// Random sequence identifier.
    random_sequence: Option<String>,
}

fn parsed_loot_table_id(registry_id: &str) -> Identifier {
    Identifier::parse_or_vanilla(registry_id)
        .unwrap_or_else(|error| panic!("invalid loot table identifier {registry_id}: {error}"))
}

fn loot_table_category_key(registry_id: &str) -> String {
    let id = parsed_loot_table_id(registry_id);
    let top = id.path.split('/').next().unwrap_or("other");
    if id.namespace == Identifier::VANILLA_NAMESPACE {
        top.to_string()
    } else {
        format!("{}_{}", id.namespace.to_snake_case(), top.to_snake_case())
    }
}

fn loot_table_field_name(registry_id: &str) -> String {
    let id = parsed_loot_table_id(registry_id);
    let suffix = id
        .path
        .split('/')
        .skip(1)
        .collect::<Vec<_>>()
        .join("_")
        .to_snake_case();
    let base = if suffix.is_empty() {
        id.path.to_snake_case()
    } else {
        suffix
    };
    if id.namespace == Identifier::VANILLA_NAMESPACE {
        base
    } else {
        format!("{}_{}", id.namespace.to_snake_case(), base)
    }
}

fn loot_table_const_ident(registry_id: &str) -> Ident {
    let id = parsed_loot_table_id(registry_id);
    let name = if id.namespace == Identifier::VANILLA_NAMESPACE {
        id.path.into_owned()
    } else {
        registry_id.replace([':', '/'], "_")
    };
    Ident::new(&name.to_shouty_snake_case(), Span::call_site())
}

fn parse_loot_table(registry_id: &str, content: &str) -> LootTableData {
    let loot_table: LootTableJson = serde_json::from_str(content)
        .unwrap_or_else(|error| panic!("Failed to parse loot table {registry_id}: {error}"));

    LootTableData {
        registry_id: registry_id.to_string(),
        category_key: loot_table_category_key(registry_id),
        field_name: loot_table_field_name(registry_id),
        const_ident: loot_table_const_ident(registry_id),
        loot_type: generate_loot_type(loot_table.loot_type.as_deref().unwrap_or("minecraft:empty")),
        pools: loot_table.pools.iter().map(generate_pool).collect(),
        functions: loot_table.functions.iter().map(generate_function).collect(),
        random_sequence: loot_table.random_sequence,
    }
}

pub(crate) fn build(overlay: &DatapackOverlay) -> TokenStream {
    let tables: Vec<LootTableData> = overlay
        .list_json_registry_ids_with_suffix("loot_table")
        .into_iter()
        .map(|(registry_id, content)| parse_loot_table(&registry_id, &content))
        .collect();

    let mut stream = TokenStream::new();

    // Imports
    stream.extend(quote! {
        use crate::loot_table::{
            BlockPredicate, BonusFormula, ConditionalLootFunction, CopySource, DamageSourcePredicate,
            DamageTagPredicate, DyeColor, EnchantedChance, EnchantmentOptions, EntityEquipment,
            EntityFlags, EntityPredicate, EquipmentSlotGroup, InstrumentOptions, LocationPredicate,
            LootCondition, LootContextEntity, LootEntry, LootFunction, LootPool, LootTable,
            LootTableRef, LootTableRegistry, LootType, NameTarget, NumberProvider, PropertyCheck,
            StewEffect, ToolPredicate,
        };
        use steel_utils::Identifier;
    });

    // Generate static constants for each loot table
    for table in &tables {
        let const_ident = &table.const_ident;
        let key = crate::generator_functions::generate_static_identifier_from_str(
            &table.registry_id,
            "loot table",
        );
        let loot_type = &table.loot_type;
        let pools = &table.pools;
        let functions = &table.functions;

        let random_sequence = table.random_sequence.as_ref().map_or_else(
            || quote! { None },
            |sequence| {
                let sequence = crate::generator_functions::generate_static_identifier_from_str(
                    sequence,
                    "loot table random sequence",
                );
                quote! { Some(#sequence) }
            },
        );

        stream.extend(quote! {
            pub static #const_ident: LootTable = LootTable {
                key: #key,
                loot_type: #loot_type,
                pools: &[#(#pools),*],
                functions: &[#(#functions),*],
                random_sequence: #random_sequence,
            };
        });
    }

    // Generate registration function
    let register_calls: Vec<TokenStream> = tables
        .iter()
        .map(|t| {
            let const_ident = &t.const_ident;
            quote! { registry.register(&#const_ident); }
        })
        .collect();

    stream.extend(quote! {
        pub fn register_loot_tables(registry: &mut LootTableRegistry) {
            #(#register_calls)*
        }
    });

    // Generate a struct with categorized access for convenience
    // Group tables by their top-level directory
    let mut categories: std::collections::BTreeMap<String, Vec<(&LootTableData, Ident)>> =
        std::collections::BTreeMap::new();

    for table in &tables {
        let field_ident = Ident::new(&table.field_name, Span::call_site());
        categories
            .entry(table.category_key.clone())
            .or_default()
            .push((table, field_ident));
    }

    // Generate category structs
    for (category, items) in &categories {
        let struct_name = Ident::new(
            &format!(
                "{}LootTables",
                category
                    .to_snake_case()
                    .replace('_', " ")
                    .split_whitespace()
                    .map(|s| {
                        let mut c = s.chars();
                        match c.next() {
                            None => String::new(),
                            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                        }
                    })
                    .collect::<String>()
            ),
            Span::call_site(),
        );

        let fields: Vec<TokenStream> = items
            .iter()
            .map(|(_, field_ident)| {
                quote! { pub #field_ident: LootTableRef, }
            })
            .collect();

        let inits: Vec<TokenStream> = items
            .iter()
            .map(|(table, field_ident)| {
                let const_ident = &table.const_ident;
                quote! { #field_ident: &#const_ident, }
            })
            .collect();

        stream.extend(quote! {
            pub struct #struct_name {
                #(#fields)*
            }

            impl #struct_name {
                pub const fn new() -> Self {
                    Self {
                        #(#inits)*
                    }
                }
            }
        });
    }

    // Generate the main LOOT_TABLES struct
    let category_fields: Vec<TokenStream> = categories
        .keys()
        .map(|category| {
            let field_ident = Ident::new(&category.to_snake_case(), Span::call_site());
            let struct_name = Ident::new(
                &format!(
                    "{}LootTables",
                    category
                        .to_snake_case()
                        .replace('_', " ")
                        .split_whitespace()
                        .map(|s| {
                            let mut c = s.chars();
                            match c.next() {
                                None => String::new(),
                                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                            }
                        })
                        .collect::<String>()
                ),
                Span::call_site(),
            );
            quote! { pub #field_ident: #struct_name, }
        })
        .collect();

    let category_inits: Vec<TokenStream> = categories
        .keys()
        .map(|category| {
            let field_ident = Ident::new(&category.to_snake_case(), Span::call_site());
            let struct_name = Ident::new(
                &format!(
                    "{}LootTables",
                    category
                        .to_snake_case()
                        .replace('_', " ")
                        .split_whitespace()
                        .map(|s| {
                            let mut c = s.chars();
                            match c.next() {
                                None => String::new(),
                                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                            }
                        })
                        .collect::<String>()
                ),
                Span::call_site(),
            );
            quote! { #field_ident: #struct_name::new(), }
        })
        .collect();

    stream.extend(quote! {
        pub struct LootTables {
            #(#category_fields)*
        }

        impl LootTables {
            pub const fn new() -> Self {
                Self {
                    #(#category_inits)*
                }
            }
        }

        pub static LOOT_TABLES: LootTables = LootTables::new();
    });

    stream
}
