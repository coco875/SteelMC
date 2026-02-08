use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use std::{fs, path::PathBuf};

/// Parsed density function from datapack JSON.
///
/// Values in the datapack format are polymorphic:
/// - Bare number -> Constant
/// - Bare string -> Reference
/// - Object with "type" field -> typed density function
#[derive(Debug, Clone)]
pub enum DensityFunctionJson {
    Constant {
        value: f64,
    },
    Reference {
        id: String,
    },
    YClampedGradient {
        from_y: i32,
        to_y: i32,
        from_value: f64,
        to_value: f64,
    },
    Noise {
        xz_scale: f64,
        y_scale: f64,
        noise: String,
    },
    ShiftedNoise {
        shift_x: Box<DensityFunctionJson>,
        shift_y: Box<DensityFunctionJson>,
        shift_z: Box<DensityFunctionJson>,
        xz_scale: f64,
        y_scale: f64,
        noise: String,
    },
    ShiftA {
        noise: String,
    },
    ShiftB {
        noise: String,
    },
    Shift {
        noise: String,
    },
    Clamp {
        input: Box<DensityFunctionJson>,
        min: f64,
        max: f64,
    },
    Abs {
        input: Box<DensityFunctionJson>,
    },
    Square {
        input: Box<DensityFunctionJson>,
    },
    Cube {
        input: Box<DensityFunctionJson>,
    },
    HalfNegative {
        input: Box<DensityFunctionJson>,
    },
    QuarterNegative {
        input: Box<DensityFunctionJson>,
    },
    Squeeze {
        input: Box<DensityFunctionJson>,
    },
    Add {
        argument1: Box<DensityFunctionJson>,
        argument2: Box<DensityFunctionJson>,
    },
    Mul {
        argument1: Box<DensityFunctionJson>,
        argument2: Box<DensityFunctionJson>,
    },
    Min {
        argument1: Box<DensityFunctionJson>,
        argument2: Box<DensityFunctionJson>,
    },
    Max {
        argument1: Box<DensityFunctionJson>,
        argument2: Box<DensityFunctionJson>,
    },
    Spline {
        spline: SplineJson,
    },
    RangeChoice {
        input: Box<DensityFunctionJson>,
        min_inclusive: f64,
        max_exclusive: f64,
        when_in_range: Box<DensityFunctionJson>,
        when_out_of_range: Box<DensityFunctionJson>,
    },
    Interpolated {
        argument: Box<DensityFunctionJson>,
    },
    FlatCache {
        argument: Box<DensityFunctionJson>,
    },
    CacheOnce {
        argument: Box<DensityFunctionJson>,
    },
    Cache2d {
        argument: Box<DensityFunctionJson>,
    },
    CacheAllInCell {
        argument: Box<DensityFunctionJson>,
    },
    BlendOffset,
    BlendAlpha,
    BlendDensity {
        input: Box<DensityFunctionJson>,
    },
    Beardifier,
    EndIslands,
    WeirdScaledSampler {
        input: Box<DensityFunctionJson>,
        noise: String,
        rarity_value_mapper: String,
    },
    OldBlendedNoise {
        xz_scale: f64,
        y_scale: f64,
        xz_factor: f64,
        y_factor: f64,
        smear_scale_multiplier: f64,
    },
}

/// Parsed spline from datapack JSON.
///
/// In the datapack format, a spline value can be:
/// - A bare number -> Constant
/// - An object with {coordinate, points} -> Multipoint
#[derive(Debug, Clone)]
pub enum SplineJson {
    Constant {
        value: f32,
    },
    Multipoint {
        coordinate: String,
        points: Vec<SplinePointJson>,
    },
}

#[derive(Debug, Clone)]
pub struct SplinePointJson {
    pub location: f32,
    pub value: SplineJson,
    pub derivative: f32,
}

/// Parsed noise router from a noise_settings datapack file.
pub struct NoiseRouterJson {
    barrier: DensityFunctionJson,
    fluid_level_floodedness: DensityFunctionJson,
    fluid_level_spread: DensityFunctionJson,
    lava: DensityFunctionJson,
    temperature: DensityFunctionJson,
    vegetation: DensityFunctionJson,
    continents: DensityFunctionJson,
    erosion: DensityFunctionJson,
    depth: DensityFunctionJson,
    ridges: DensityFunctionJson,
    preliminary_surface_level: Option<DensityFunctionJson>,
    final_density: DensityFunctionJson,
    vein_toggle: DensityFunctionJson,
    vein_ridged: DensityFunctionJson,
    vein_gap: DensityFunctionJson,
}

// ── Datapack JSON parsing ───────────────────────────────────────────────────

fn get_f64(val: &Value) -> f64 {
    val.as_f64().unwrap_or(0.0)
}

fn get_f32(val: &Value) -> f32 {
    get_f64(val) as f32
}

fn get_i32(val: &Value) -> i32 {
    val.as_i64().unwrap_or(0) as i32
}

fn get_str(val: &Value) -> &str {
    val.as_str().unwrap_or("")
}

/// Parse a polymorphic density function value from datapack JSON.
///
/// A density function value can be:
/// - A bare number (constant)
/// - A bare string (reference to another density function)
/// - An object with a "type" field
fn parse_df(val: &Value) -> DensityFunctionJson {
    match val {
        Value::Number(n) => DensityFunctionJson::Constant {
            value: n.as_f64().unwrap_or(0.0),
        },
        Value::String(s) => DensityFunctionJson::Reference { id: s.clone() },
        Value::Object(map) => parse_df_object(map),
        _ => DensityFunctionJson::Constant { value: 0.0 },
    }
}

fn parse_df_object(map: &serde_json::Map<String, Value>) -> DensityFunctionJson {
    let type_str = map
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let type_name = type_str.strip_prefix("minecraft:").unwrap_or(type_str);

    match type_name {
        "constant" => DensityFunctionJson::Constant {
            value: get_f64(map.get("argument").or_else(|| map.get("value")).unwrap()),
        },

        "y_clamped_gradient" => DensityFunctionJson::YClampedGradient {
            from_y: get_i32(&map["from_y"]),
            to_y: get_i32(&map["to_y"]),
            from_value: get_f64(&map["from_value"]),
            to_value: get_f64(&map["to_value"]),
        },

        "noise" => DensityFunctionJson::Noise {
            xz_scale: get_f64(&map["xz_scale"]),
            y_scale: get_f64(&map["y_scale"]),
            noise: get_str(&map["noise"]).to_string(),
        },

        "shifted_noise" => DensityFunctionJson::ShiftedNoise {
            shift_x: Box::new(parse_df(&map["shift_x"])),
            shift_y: Box::new(parse_df(&map["shift_y"])),
            shift_z: Box::new(parse_df(&map["shift_z"])),
            xz_scale: get_f64(&map["xz_scale"]),
            y_scale: get_f64(&map["y_scale"]),
            noise: get_str(&map["noise"]).to_string(),
        },

        "shift_a" => DensityFunctionJson::ShiftA {
            noise: get_str(&map["argument"]).to_string(),
        },
        "shift_b" => DensityFunctionJson::ShiftB {
            noise: get_str(&map["argument"]).to_string(),
        },
        "shift" => DensityFunctionJson::Shift {
            noise: get_str(&map["argument"]).to_string(),
        },

        "clamp" => DensityFunctionJson::Clamp {
            input: Box::new(parse_df(&map["input"])),
            min: get_f64(&map["min"]),
            max: get_f64(&map["max"]),
        },

        "abs" => DensityFunctionJson::Abs {
            input: Box::new(parse_df(&map["argument"])),
        },
        "square" => DensityFunctionJson::Square {
            input: Box::new(parse_df(&map["argument"])),
        },
        "cube" => DensityFunctionJson::Cube {
            input: Box::new(parse_df(&map["argument"])),
        },
        "half_negative" => DensityFunctionJson::HalfNegative {
            input: Box::new(parse_df(&map["argument"])),
        },
        "quarter_negative" => DensityFunctionJson::QuarterNegative {
            input: Box::new(parse_df(&map["argument"])),
        },
        "squeeze" => DensityFunctionJson::Squeeze {
            input: Box::new(parse_df(&map["argument"])),
        },

        "add" => DensityFunctionJson::Add {
            argument1: Box::new(parse_df(&map["argument1"])),
            argument2: Box::new(parse_df(&map["argument2"])),
        },
        "mul" => DensityFunctionJson::Mul {
            argument1: Box::new(parse_df(&map["argument1"])),
            argument2: Box::new(parse_df(&map["argument2"])),
        },
        "min" => DensityFunctionJson::Min {
            argument1: Box::new(parse_df(&map["argument1"])),
            argument2: Box::new(parse_df(&map["argument2"])),
        },
        "max" => DensityFunctionJson::Max {
            argument1: Box::new(parse_df(&map["argument1"])),
            argument2: Box::new(parse_df(&map["argument2"])),
        },

        "spline" => DensityFunctionJson::Spline {
            spline: parse_spline(&map["spline"]),
        },

        "range_choice" => DensityFunctionJson::RangeChoice {
            input: Box::new(parse_df(&map["input"])),
            min_inclusive: get_f64(&map["min_inclusive"]),
            max_exclusive: get_f64(&map["max_exclusive"]),
            when_in_range: Box::new(parse_df(&map["when_in_range"])),
            when_out_of_range: Box::new(parse_df(&map["when_out_of_range"])),
        },

        "interpolated" => DensityFunctionJson::Interpolated {
            argument: Box::new(parse_df(&map["argument"])),
        },
        "flat_cache" => DensityFunctionJson::FlatCache {
            argument: Box::new(parse_df(&map["argument"])),
        },
        "cache_once" => DensityFunctionJson::CacheOnce {
            argument: Box::new(parse_df(&map["argument"])),
        },
        "cache_2d" => DensityFunctionJson::Cache2d {
            argument: Box::new(parse_df(&map["argument"])),
        },
        "cache_all_in_cell" => DensityFunctionJson::CacheAllInCell {
            argument: Box::new(parse_df(&map["argument"])),
        },

        "blend_offset" => DensityFunctionJson::BlendOffset,
        "blend_alpha" => DensityFunctionJson::BlendAlpha,
        "blend_density" => DensityFunctionJson::BlendDensity {
            input: Box::new(parse_df(&map["argument"])),
        },

        "beardifier" => DensityFunctionJson::Beardifier,
        "end_islands" => DensityFunctionJson::EndIslands,

        "weird_scaled_sampler" => DensityFunctionJson::WeirdScaledSampler {
            input: Box::new(parse_df(&map["input"])),
            noise: get_str(&map["noise"]).to_string(),
            rarity_value_mapper: get_str(&map["rarity_value_mapper"]).to_string(),
        },

        "old_blended_noise" => DensityFunctionJson::OldBlendedNoise {
            xz_scale: get_f64(&map["xz_scale"]),
            y_scale: get_f64(&map["y_scale"]),
            xz_factor: get_f64(&map["xz_factor"]),
            y_factor: get_f64(&map["y_factor"]),
            smear_scale_multiplier: get_f64(&map["smear_scale_multiplier"]),
        },

        // find_top_surface is only used in preliminary_surface_level which is unused
        "find_top_surface" => DensityFunctionJson::Constant { value: 0.0 },

        other => {
            panic!("Unknown density function type: {other}");
        }
    }
}

/// Parse a spline value from datapack JSON.
///
/// A spline can be:
/// - A bare number -> Constant
/// - An object with {coordinate, points} -> Multipoint
fn parse_spline(val: &Value) -> SplineJson {
    match val {
        Value::Number(n) => SplineJson::Constant {
            value: n.as_f64().unwrap_or(0.0) as f32,
        },
        Value::Object(map) => {
            let coordinate = get_str(&map["coordinate"]).to_string();
            let points = map["points"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .map(|p| SplinePointJson {
                            location: get_f32(&p["location"]),
                            value: parse_spline(&p["value"]),
                            derivative: get_f32(&p["derivative"]),
                        })
                        .collect()
                })
                .unwrap_or_default();
            SplineJson::Multipoint { coordinate, points }
        }
        _ => SplineJson::Constant { value: 0.0 },
    }
}

/// Parse the noise_router object from a noise_settings JSON file.
fn parse_noise_router(val: &Value) -> NoiseRouterJson {
    let router = &val["noise_router"];
    NoiseRouterJson {
        barrier: parse_df(&router["barrier"]),
        fluid_level_floodedness: parse_df(&router["fluid_level_floodedness"]),
        fluid_level_spread: parse_df(&router["fluid_level_spread"]),
        lava: parse_df(&router["lava"]),
        temperature: parse_df(&router["temperature"]),
        vegetation: parse_df(&router["vegetation"]),
        continents: parse_df(&router["continents"]),
        erosion: parse_df(&router["erosion"]),
        depth: parse_df(&router["depth"]),
        ridges: parse_df(&router["ridges"]),
        preliminary_surface_level: router.get("preliminary_surface_level").map(parse_df),
        final_density: parse_df(&router["final_density"]),
        vein_toggle: parse_df(&router["vein_toggle"]),
        vein_ridged: parse_df(&router["vein_ridged"]),
        vein_gap: parse_df(&router["vein_gap"]),
    }
}

// ── Datapack file reading ───────────────────────────────────────────────────

const DATAPACK_BASE: &str = "build_assets/builtin_datapacks/minecraft/data/minecraft/worldgen";

/// Recursively collect all .json files under a directory.
fn collect_json_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(collect_json_files(&path));
            } else if path.extension().is_some_and(|ext| ext == "json") {
                files.push(path);
            }
        }
    }
    files
}

/// Convert a density_function file path to a registry ID.
///
/// e.g. `.../density_function/overworld/continents.json` -> `minecraft:overworld/continents`
fn path_to_id(path: &Path, base_dir: &Path) -> String {
    let relative = path.strip_prefix(base_dir).unwrap();
    let without_ext = relative.with_extension("");
    // Convert OS path separators to forward slashes
    let id_path = without_ext
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect::<Vec<_>>()
        .join("/");
    format!("minecraft:{id_path}")
}

/// Read all density function files from the datapack into a registry.
fn read_density_function_registry() -> BTreeMap<String, DensityFunctionJson> {
    let df_dir = format!("{DATAPACK_BASE}/density_function");
    let df_path = Path::new(&df_dir);
    let mut registry = BTreeMap::new();

    for file in collect_json_files(df_path) {
        println!("cargo:rerun-if-changed={}", file.display());
        let content = fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", file.display()));
        let val: Value = serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", file.display()));
        let id = path_to_id(&file, df_path);
        let df = parse_df(&val);
        registry.insert(id, df);
    }

    registry
}

/// Read the overworld noise settings from the datapack.
fn read_overworld_noise_router() -> NoiseRouterJson {
    let path = format!("{DATAPACK_BASE}/noise_settings/overworld.json");
    println!("cargo:rerun-if-changed={path}");
    let content =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("Failed to read {path}: {e}"));
    let val: Value =
        serde_json::from_str(&content).unwrap_or_else(|e| panic!("Failed to parse {path}: {e}"));
    parse_noise_router(&val)
}

// ── Code generation ─────────────────────────────────────────────────────────

/// Generate the Rust code for density functions.
pub(crate) fn build() -> TokenStream {
    let router = read_overworld_noise_router();
    let registry = read_density_function_registry();

    let mut stream = TokenStream::new();

    stream.extend(quote! {
        //! Generated density functions and noise routers.
        //!
        //! This file is auto-generated from the vanilla datapack files.
        //! Do not edit manually.

        use std::sync::Arc;
    });

    stream.extend(generate_density_function_enum());
    stream.extend(generate_spline_types());
    stream.extend(generate_noise_router_struct());

    let router_code = generate_noise_router_instance("OVERWORLD", &router);
    stream.extend(router_code);

    let registry_code = generate_density_function_registry(&registry);
    stream.extend(registry_code);

    stream
}

fn generate_density_function_enum() -> TokenStream {
    quote! {
        /// Density function types for terrain generation.
        #[derive(Debug, Clone)]
        pub enum DensityFunction {
            /// Constant value
            Constant(f64),

            /// Reference to another density function by ID
            Reference(&'static str),

            /// Y clamped gradient
            YClampedGradient {
                from_y: i32,
                to_y: i32,
                from_value: f64,
                to_value: f64,
            },

            /// Noise sampler
            Noise {
                noise_id: &'static str,
                xz_scale: f64,
                y_scale: f64,
            },

            /// Shifted noise sampler
            ShiftedNoise {
                shift_x: Arc<DensityFunction>,
                shift_y: Arc<DensityFunction>,
                shift_z: Arc<DensityFunction>,
                xz_scale: f64,
                y_scale: f64,
                noise_id: &'static str,
            },

            /// Shift A (for offset noise)
            ShiftA { noise_id: &'static str },
            /// Shift B (for offset noise)
            ShiftB { noise_id: &'static str },
            /// Shift (for offset noise)
            Shift { noise_id: &'static str },

            /// Clamp value to range
            Clamp {
                input: Arc<DensityFunction>,
                min: f64,
                max: f64,
            },

            /// Absolute value
            Abs(Arc<DensityFunction>),
            /// Square
            Square(Arc<DensityFunction>),
            /// Cube
            Cube(Arc<DensityFunction>),
            /// Half negative (v > 0 ? v : v * 0.5)
            HalfNegative(Arc<DensityFunction>),
            /// Quarter negative (v > 0 ? v : v * 0.25)
            QuarterNegative(Arc<DensityFunction>),
            /// Squeeze function
            Squeeze(Arc<DensityFunction>),

            /// Add two functions
            Add(Arc<DensityFunction>, Arc<DensityFunction>),
            /// Multiply two functions
            Mul(Arc<DensityFunction>, Arc<DensityFunction>),
            /// Minimum of two functions
            Min(Arc<DensityFunction>, Arc<DensityFunction>),
            /// Maximum of two functions
            Max(Arc<DensityFunction>, Arc<DensityFunction>),

            /// Cubic spline
            Spline(CubicSpline),

            /// Range choice
            RangeChoice {
                input: Arc<DensityFunction>,
                min_inclusive: f64,
                max_exclusive: f64,
                when_in_range: Arc<DensityFunction>,
                when_out_of_range: Arc<DensityFunction>,
            },

            /// Interpolated (cache marker)
            Interpolated(Arc<DensityFunction>),
            /// Flat cache (cache marker)
            FlatCache(Arc<DensityFunction>),
            /// Cache once
            CacheOnce(Arc<DensityFunction>),
            /// Cache 2D
            Cache2d(Arc<DensityFunction>),
            /// Cache all in cell
            CacheAllInCell(Arc<DensityFunction>),

            /// Blend offset (for world blending)
            BlendOffset,
            /// Blend alpha (for world blending)
            BlendAlpha,
            /// Blend density
            BlendDensity(Arc<DensityFunction>),

            /// Beardifier (structure terrain modification)
            Beardifier,

            /// End islands
            EndIslands,

            /// Weird scaled sampler
            WeirdScaledSampler {
                input: Arc<DensityFunction>,
                noise_id: &'static str,
                rarity_value_mapper: RarityValueMapper,
            },

            /// Old blended noise (3D terrain noise)
            OldBlendedNoise {
                xz_scale: f64,
                y_scale: f64,
                xz_factor: f64,
                y_factor: f64,
                smear_scale_multiplier: f64,
            },
        }

        /// Rarity value mapper for weird scaled sampler
        #[derive(Debug, Clone, Copy)]
        pub enum RarityValueMapper {
            Tunnels,
            Caves,
        }
    }
}

fn generate_spline_types() -> TokenStream {
    quote! {
        /// A point in a cubic spline
        #[derive(Debug, Clone)]
        pub struct SplinePoint {
            pub location: f32,
            pub value: CubicSpline,
            pub derivative: f32,
        }

        /// Cubic spline for density function interpolation
        #[derive(Debug, Clone)]
        pub enum CubicSpline {
            /// Constant value
            Constant(f32),
            /// Multipoint spline with coordinate function and points
            Multipoint {
                coordinate: Arc<DensityFunction>,
                points: Vec<SplinePoint>,
            },
        }
    }
}

fn generate_noise_router_struct() -> TokenStream {
    quote! {
        /// Noise router containing all density functions for terrain generation.
        #[derive(Debug, Clone)]
        pub struct NoiseRouter {
            pub barrier: Arc<DensityFunction>,
            pub fluid_level_floodedness: Arc<DensityFunction>,
            pub fluid_level_spread: Arc<DensityFunction>,
            pub lava: Arc<DensityFunction>,
            pub temperature: Arc<DensityFunction>,
            pub vegetation: Arc<DensityFunction>,
            pub continents: Arc<DensityFunction>,
            pub erosion: Arc<DensityFunction>,
            pub depth: Arc<DensityFunction>,
            pub ridges: Arc<DensityFunction>,
            pub preliminary_surface_level: Option<Arc<DensityFunction>>,
            pub final_density: Arc<DensityFunction>,
            pub vein_toggle: Arc<DensityFunction>,
            pub vein_ridged: Arc<DensityFunction>,
            pub vein_gap: Arc<DensityFunction>,
        }
    }
}

fn generate_noise_router_instance(name: &str, router: &NoiseRouterJson) -> TokenStream {
    let name_ident = Ident::new(&format!("{}_NOISE_ROUTER", name), Span::call_site());

    let barrier = generate_df_code(&router.barrier);
    let fluid_floodedness = generate_df_code(&router.fluid_level_floodedness);
    let fluid_spread = generate_df_code(&router.fluid_level_spread);
    let lava = generate_df_code(&router.lava);
    let temperature = generate_df_code(&router.temperature);
    let vegetation = generate_df_code(&router.vegetation);
    let continents = generate_df_code(&router.continents);
    let erosion = generate_df_code(&router.erosion);
    let depth = generate_df_code(&router.depth);
    let ridges = generate_df_code(&router.ridges);
    let final_density = generate_df_code(&router.final_density);
    let vein_toggle = generate_df_code(&router.vein_toggle);
    let vein_ridged = generate_df_code(&router.vein_ridged);
    let vein_gap = generate_df_code(&router.vein_gap);

    let preliminary = if let Some(ref psl) = router.preliminary_surface_level {
        let code = generate_df_code(psl);
        quote! { Some(Arc::new(#code)) }
    } else {
        quote! { None }
    };

    quote! {
        use std::sync::LazyLock;

        /// Overworld noise router with all density functions.
        pub static #name_ident: LazyLock<NoiseRouter> = LazyLock::new(|| {
            NoiseRouter {
                barrier: Arc::new(#barrier),
                fluid_level_floodedness: Arc::new(#fluid_floodedness),
                fluid_level_spread: Arc::new(#fluid_spread),
                lava: Arc::new(#lava),
                temperature: Arc::new(#temperature),
                vegetation: Arc::new(#vegetation),
                continents: Arc::new(#continents),
                erosion: Arc::new(#erosion),
                depth: Arc::new(#depth),
                ridges: Arc::new(#ridges),
                preliminary_surface_level: #preliminary,
                final_density: Arc::new(#final_density),
                vein_toggle: Arc::new(#vein_toggle),
                vein_ridged: Arc::new(#vein_ridged),
                vein_gap: Arc::new(#vein_gap),
            }
        });
    }
}

fn generate_df_code(df: &DensityFunctionJson) -> TokenStream {
    match df {
        DensityFunctionJson::Constant { value } => {
            quote! { DensityFunction::Constant(#value) }
        }

        DensityFunctionJson::Reference { id } => {
            quote! { DensityFunction::Reference(#id) }
        }

        DensityFunctionJson::YClampedGradient {
            from_y,
            to_y,
            from_value,
            to_value,
        } => {
            quote! {
                DensityFunction::YClampedGradient {
                    from_y: #from_y,
                    to_y: #to_y,
                    from_value: #from_value,
                    to_value: #to_value,
                }
            }
        }

        DensityFunctionJson::Noise {
            xz_scale,
            y_scale,
            noise,
        } => {
            quote! {
                DensityFunction::Noise {
                    noise_id: #noise,
                    xz_scale: #xz_scale,
                    y_scale: #y_scale,
                }
            }
        }

        DensityFunctionJson::ShiftedNoise {
            shift_x,
            shift_y,
            shift_z,
            xz_scale,
            y_scale,
            noise,
        } => {
            let shift_x_code = generate_df_code(shift_x);
            let shift_y_code = generate_df_code(shift_y);
            let shift_z_code = generate_df_code(shift_z);
            quote! {
                DensityFunction::ShiftedNoise {
                    shift_x: Arc::new(#shift_x_code),
                    shift_y: Arc::new(#shift_y_code),
                    shift_z: Arc::new(#shift_z_code),
                    xz_scale: #xz_scale,
                    y_scale: #y_scale,
                    noise_id: #noise,
                }
            }
        }

        DensityFunctionJson::ShiftA { noise } => {
            quote! { DensityFunction::ShiftA { noise_id: #noise } }
        }

        DensityFunctionJson::ShiftB { noise } => {
            quote! { DensityFunction::ShiftB { noise_id: #noise } }
        }

        DensityFunctionJson::Shift { noise } => {
            quote! { DensityFunction::Shift { noise_id: #noise } }
        }

        DensityFunctionJson::Clamp { input, min, max } => {
            let input_code = generate_df_code(input);
            quote! {
                DensityFunction::Clamp {
                    input: Arc::new(#input_code),
                    min: #min,
                    max: #max,
                }
            }
        }

        DensityFunctionJson::Abs { input } => {
            let input_code = generate_df_code(input);
            quote! { DensityFunction::Abs(Arc::new(#input_code)) }
        }

        DensityFunctionJson::Square { input } => {
            let input_code = generate_df_code(input);
            quote! { DensityFunction::Square(Arc::new(#input_code)) }
        }

        DensityFunctionJson::Cube { input } => {
            let input_code = generate_df_code(input);
            quote! { DensityFunction::Cube(Arc::new(#input_code)) }
        }

        DensityFunctionJson::HalfNegative { input } => {
            let input_code = generate_df_code(input);
            quote! { DensityFunction::HalfNegative(Arc::new(#input_code)) }
        }

        DensityFunctionJson::QuarterNegative { input } => {
            let input_code = generate_df_code(input);
            quote! { DensityFunction::QuarterNegative(Arc::new(#input_code)) }
        }

        DensityFunctionJson::Squeeze { input } => {
            let input_code = generate_df_code(input);
            quote! { DensityFunction::Squeeze(Arc::new(#input_code)) }
        }

        DensityFunctionJson::Add {
            argument1,
            argument2,
        } => {
            let arg1 = generate_df_code(argument1);
            let arg2 = generate_df_code(argument2);
            quote! { DensityFunction::Add(Arc::new(#arg1), Arc::new(#arg2)) }
        }

        DensityFunctionJson::Mul {
            argument1,
            argument2,
        } => {
            let arg1 = generate_df_code(argument1);
            let arg2 = generate_df_code(argument2);
            quote! { DensityFunction::Mul(Arc::new(#arg1), Arc::new(#arg2)) }
        }

        DensityFunctionJson::Min {
            argument1,
            argument2,
        } => {
            let arg1 = generate_df_code(argument1);
            let arg2 = generate_df_code(argument2);
            quote! { DensityFunction::Min(Arc::new(#arg1), Arc::new(#arg2)) }
        }

        DensityFunctionJson::Max {
            argument1,
            argument2,
        } => {
            let arg1 = generate_df_code(argument1);
            let arg2 = generate_df_code(argument2);
            quote! { DensityFunction::Max(Arc::new(#arg1), Arc::new(#arg2)) }
        }

        DensityFunctionJson::Spline { spline } => {
            let spline_code = generate_spline_code(spline);
            quote! { DensityFunction::Spline(#spline_code) }
        }

        DensityFunctionJson::RangeChoice {
            input,
            min_inclusive,
            max_exclusive,
            when_in_range,
            when_out_of_range,
        } => {
            let input_code = generate_df_code(input);
            let in_range_code = generate_df_code(when_in_range);
            let out_of_range_code = generate_df_code(when_out_of_range);
            quote! {
                DensityFunction::RangeChoice {
                    input: Arc::new(#input_code),
                    min_inclusive: #min_inclusive,
                    max_exclusive: #max_exclusive,
                    when_in_range: Arc::new(#in_range_code),
                    when_out_of_range: Arc::new(#out_of_range_code),
                }
            }
        }

        DensityFunctionJson::Interpolated { argument } => {
            let arg_code = generate_df_code(argument);
            quote! { DensityFunction::Interpolated(Arc::new(#arg_code)) }
        }

        DensityFunctionJson::FlatCache { argument } => {
            let arg_code = generate_df_code(argument);
            quote! { DensityFunction::FlatCache(Arc::new(#arg_code)) }
        }

        DensityFunctionJson::CacheOnce { argument } => {
            let arg_code = generate_df_code(argument);
            quote! { DensityFunction::CacheOnce(Arc::new(#arg_code)) }
        }

        DensityFunctionJson::Cache2d { argument } => {
            let arg_code = generate_df_code(argument);
            quote! { DensityFunction::Cache2d(Arc::new(#arg_code)) }
        }

        DensityFunctionJson::CacheAllInCell { argument } => {
            let arg_code = generate_df_code(argument);
            quote! { DensityFunction::CacheAllInCell(Arc::new(#arg_code)) }
        }

        DensityFunctionJson::BlendOffset => {
            quote! { DensityFunction::BlendOffset }
        }

        DensityFunctionJson::BlendAlpha => {
            quote! { DensityFunction::BlendAlpha }
        }

        DensityFunctionJson::BlendDensity { input } => {
            let input_code = generate_df_code(input);
            quote! { DensityFunction::BlendDensity(Arc::new(#input_code)) }
        }

        DensityFunctionJson::Beardifier => {
            quote! { DensityFunction::Beardifier }
        }

        DensityFunctionJson::EndIslands => {
            quote! { DensityFunction::EndIslands }
        }

        DensityFunctionJson::WeirdScaledSampler {
            input,
            noise,
            rarity_value_mapper,
        } => {
            let input_code = generate_df_code(input);
            let mapper = match rarity_value_mapper.as_str() {
                "type_1" => quote! { RarityValueMapper::Tunnels },
                "type_2" => quote! { RarityValueMapper::Caves },
                _ => quote! { RarityValueMapper::Caves },
            };
            quote! {
                DensityFunction::WeirdScaledSampler {
                    input: Arc::new(#input_code),
                    noise_id: #noise,
                    rarity_value_mapper: #mapper,
                }
            }
        }

        DensityFunctionJson::OldBlendedNoise {
            xz_scale,
            y_scale,
            xz_factor,
            y_factor,
            smear_scale_multiplier,
        } => {
            quote! {
                DensityFunction::OldBlendedNoise {
                    xz_scale: #xz_scale,
                    y_scale: #y_scale,
                    xz_factor: #xz_factor,
                    y_factor: #y_factor,
                    smear_scale_multiplier: #smear_scale_multiplier,
                }
            }
        }
    }
}

fn generate_spline_code(spline: &SplineJson) -> TokenStream {
    match spline {
        SplineJson::Constant { value } => {
            let v = *value;
            quote! { CubicSpline::Constant(#v) }
        }

        SplineJson::Multipoint { coordinate, points } => {
            let coord_code = quote! { Arc::new(DensityFunction::Reference(#coordinate)) };

            let points_code: Vec<TokenStream> = points
                .iter()
                .map(|p| {
                    let loc = p.location;
                    let val_code = generate_spline_code(&p.value);
                    let deriv = p.derivative;
                    quote! {
                        SplinePoint {
                            location: #loc,
                            value: #val_code,
                            derivative: #deriv,
                        }
                    }
                })
                .collect();

            quote! {
                CubicSpline::Multipoint {
                    coordinate: #coord_code,
                    points: vec![#(#points_code),*],
                }
            }
        }
    }
}

fn generate_density_function_registry(
    registry: &BTreeMap<String, DensityFunctionJson>,
) -> TokenStream {
    let entries: Vec<TokenStream> = registry
        .iter()
        .map(|(id, df)| {
            let df_code = generate_df_code(df);
            quote! { #id => Some(Arc::new(#df_code)) }
        })
        .collect();

    quote! {
        /// Get density function by ID from the registry.
        pub fn get_density_function(id: &str) -> Option<Arc<DensityFunction>> {
            match id {
                #(#entries,)*
                _ => None,
            }
        }
    }
}
