# Exotic example plugin

This plugin depends on the default, non-host API and contains no Steel core,
registry, worldgen, or utility dependencies.

Pre-init only subscribes to hooks. The block and generator registration
callbacks resolve their keys and edit the live registries immediately before they
freeze. Init runs afterward, resolves the remaining static values, and prepares
the Game of Life configuration.

```sh
cargo build --manifest-path examples/exotic-plugin/Cargo.toml --release
cargo test --manifest-path examples/exotic-plugin/Cargo.toml
```

Add this to the server's `config/config.toml` (Linux library path shown):

```toml
[[plugins]]
path = "examples/exotic-plugin/target/release/libsteel_exotic_plugin.so"

[plugins.config.life]
world = "minecraft:overworld"
x = 0
y = 96
z = 0
width = 32
depth = 32
interval = 10
```

Use `steel_exotic_plugin.dll` on Windows or `libsteel_exotic_plugin.dylib` on macOS.
Paths are relative to the server working directory. Start the server after building
the plugin. Loading errors stop startup before worlds begin ticking.

Right-click any full wool block with shears to replace it with oak wood and drop
one wool item of the original color. Drops respect the block-drops game rule.
Survival uses one durability; creative respects the host's existing infinite-materials rule.
Each wool block receives a complete `BlockBehavior` implementation that retains
its `OriginalBlockBehavior` handle. The shearing override explicitly delegates
other interactions; omitted methods retain the original behavior. Registration is
manual in the block-registration callback. `pre_init` subscribes that callback
and the separate world-tick callback through `Hooks`.

The configured X/Z board runs Conway's Game of Life. Black concrete is alive;
every other state is dead. Cells that die become white concrete. Place black
concrete to seed a pattern. Updates use one complete snapshot, with dead boundaries,
and pause if any board chunk is unavailable or normal ticking is frozen. The
example limits each board dimension to 1–128 cells to bound synchronous tick work.

To use the strange-world generator, select `example:strange` for a new world in
`config/worlds.toml`, for example:

```toml
[[domains.minecraft.worlds]]
name = "strange"
generator = "example:strange"
```

It generates floating stone islands, hollow end-stone shells, and twisting purpur
pillars with glowstone. It uses overworld dimension rules and an End biome, with no
structures or decorators. Seeded positional samples select global geometry offsets;
integer shape evaluation makes chunk output independent of invocation order.
Existing worlds continue to use their configured generator.
