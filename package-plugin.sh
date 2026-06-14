#!/bin/bash
set -e

# Default to release mode
MODE="release"
CRATE="steel-example-plugin"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) MODE="debug"; shift ;;
        --release) MODE="release"; shift ;;
        *) CRATE="$1"; shift ;;
    esac
done

echo "Packaging crate '$CRATE' in $MODE mode..."

# Build the crate
cargo build -p "$CRATE" $( [ "$MODE" = "release" ] && echo "--release" )

TARGET_DIR="target/$MODE"
PLUGINS_DIR="plugins"
mkdir -p "$PLUGINS_DIR"

# Find the mods.toml in the crate directory
MODS_TOML="$CRATE/mods.toml"
if [ ! -f "$MODS_TOML" ]; then
    echo "Error: $MODS_TOML not found."
    exit 1
fi

# Create temp packaging dir
TEMP_PACK_DIR=$(mktemp -d)
cp "$MODS_TOML" "$TEMP_PACK_DIR/"

# Find the built library file(s) for the current platform
# On Linux: lib<crate_name_underscored>.so
# On macOS: lib<crate_name_underscored>.dylib
# On Windows: <crate_name_underscored>.dll
UNDERSCORED_NAME=$(echo "$CRATE" | tr '-' '_')

LIB_FOUND=false

for EXT in so dylib dll; do
    # Check directly in target directory or in target/deps/
    for PREFIX in "lib" ""; do
        LIB_FILE="${PREFIX}${UNDERSCORED_NAME}.${EXT}"
        if [ -f "$TARGET_DIR/$LIB_FILE" ]; then
            cp "$TARGET_DIR/$LIB_FILE" "$TEMP_PACK_DIR/"
            LIB_FOUND=true
            echo "Added $LIB_FILE to package."
        elif [ -f "$TARGET_DIR/deps/$LIB_FILE" ]; then
            cp "$TARGET_DIR/deps/$LIB_FILE" "$TEMP_PACK_DIR/"
            LIB_FOUND=true
            echo "Added $LIB_FILE to package."
        fi
    done
done

if [ "$LIB_FOUND" = false ]; then
    echo "Error: Could not find compiled library for crate '$CRATE' in $TARGET_DIR"
    rm -rf "$TEMP_PACK_DIR"
    exit 1
fi

# Zip everything inside the temp pack directory
ZIP_FILE="$PLUGINS_DIR/${UNDERSCORED_NAME}.zip"
# Remove old zip if exists
rm -f "$ZIP_FILE"

# Make relative path resolve correctly
ABS_ZIP_PATH="$(pwd)/$ZIP_FILE"

(cd "$TEMP_PACK_DIR" && zip -r "$ABS_ZIP_PATH" .)

rm -rf "$TEMP_PACK_DIR"

echo "Successfully created plugin archive at: $ZIP_FILE"
