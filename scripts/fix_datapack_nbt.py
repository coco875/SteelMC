#!/usr/bin/env python3
"""Create fixed copies of datapack zips by patching structure NBT block ids.

The script never overwrites the source zip. By default it processes every
non-fixed zip in the script directory and writes `<name>-nbt-fixed.zip`.
"""

from __future__ import annotations

import argparse
import gzip
import io
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BLOCK_RENAMES = {
    "minecraft:grass": "minecraft:short_grass",
    "minecraft:grass_path": "minecraft:dirt_path",
    "minecraft:chain": "minecraft:iron_chain",
}

TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12


@dataclass
class Tag:
    tag_type: int
    value: Any


class NbtReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def read_exact(self, length: int) -> bytes:
        end = self.offset + length
        if end > len(self.data):
            raise ValueError("unexpected end of NBT data")
        out = self.data[self.offset:end]
        self.offset = end
        return out

    def read_u8(self) -> int:
        return self.read_exact(1)[0]

    def read_i8(self) -> int:
        return struct.unpack(">b", self.read_exact(1))[0]

    def read_i16(self) -> int:
        return struct.unpack(">h", self.read_exact(2))[0]

    def read_i32(self) -> int:
        return struct.unpack(">i", self.read_exact(4))[0]

    def read_i64(self) -> int:
        return struct.unpack(">q", self.read_exact(8))[0]

    def read_f32(self) -> float:
        return struct.unpack(">f", self.read_exact(4))[0]

    def read_f64(self) -> float:
        return struct.unpack(">d", self.read_exact(8))[0]

    def read_string(self) -> str:
        length = self.read_i16()
        if length < 0:
            raise ValueError("negative NBT string length")
        return self.read_exact(length).decode("utf-8")

    def read_payload(self, tag_type: int) -> Any:
        if tag_type == TAG_BYTE:
            return self.read_i8()
        if tag_type == TAG_SHORT:
            return self.read_i16()
        if tag_type == TAG_INT:
            return self.read_i32()
        if tag_type == TAG_LONG:
            return self.read_i64()
        if tag_type == TAG_FLOAT:
            return self.read_f32()
        if tag_type == TAG_DOUBLE:
            return self.read_f64()
        if tag_type == TAG_BYTE_ARRAY:
            length = self.read_i32()
            if length < 0:
                raise ValueError("negative byte array length")
            return self.read_exact(length)
        if tag_type == TAG_STRING:
            return self.read_string()
        if tag_type == TAG_LIST:
            item_type = self.read_u8()
            length = self.read_i32()
            if length < 0:
                raise ValueError("negative list length")
            return item_type, [self.read_payload(item_type) for _ in range(length)]
        if tag_type == TAG_COMPOUND:
            entries: list[tuple[str, Tag]] = []
            while True:
                child_type = self.read_u8()
                if child_type == TAG_END:
                    return entries
                name = self.read_string()
                entries.append((name, Tag(child_type, self.read_payload(child_type))))
        if tag_type == TAG_INT_ARRAY:
            length = self.read_i32()
            if length < 0:
                raise ValueError("negative int array length")
            return [self.read_i32() for _ in range(length)]
        if tag_type == TAG_LONG_ARRAY:
            length = self.read_i32()
            if length < 0:
                raise ValueError("negative long array length")
            return [self.read_i64() for _ in range(length)]
        raise ValueError(f"unsupported NBT tag type {tag_type}")

    def read_root(self) -> tuple[int, str, Any]:
        root_type = self.read_u8()
        if root_type == TAG_END:
            return root_type, "", None
        name = self.read_string()
        value = self.read_payload(root_type)
        if self.offset != len(self.data):
            raise ValueError("trailing bytes after NBT root")
        return root_type, name, value


class NbtWriter:
    def __init__(self) -> None:
        self.out = io.BytesIO()

    def write(self, data: bytes) -> None:
        self.out.write(data)

    def write_u8(self, value: int) -> None:
        self.write(struct.pack(">B", value))

    def write_i8(self, value: int) -> None:
        self.write(struct.pack(">b", value))

    def write_i16(self, value: int) -> None:
        self.write(struct.pack(">h", value))

    def write_i32(self, value: int) -> None:
        self.write(struct.pack(">i", value))

    def write_i64(self, value: int) -> None:
        self.write(struct.pack(">q", value))

    def write_f32(self, value: float) -> None:
        self.write(struct.pack(">f", value))

    def write_f64(self, value: float) -> None:
        self.write(struct.pack(">d", value))

    def write_string(self, value: str) -> None:
        data = value.encode("utf-8")
        if len(data) > 0x7FFF:
            raise ValueError("NBT string is too long")
        self.write_i16(len(data))
        self.write(data)

    def write_payload(self, tag_type: int, value: Any) -> None:
        if tag_type == TAG_BYTE:
            self.write_i8(value)
        elif tag_type == TAG_SHORT:
            self.write_i16(value)
        elif tag_type == TAG_INT:
            self.write_i32(value)
        elif tag_type == TAG_LONG:
            self.write_i64(value)
        elif tag_type == TAG_FLOAT:
            self.write_f32(value)
        elif tag_type == TAG_DOUBLE:
            self.write_f64(value)
        elif tag_type == TAG_BYTE_ARRAY:
            self.write_i32(len(value))
            self.write(value)
        elif tag_type == TAG_STRING:
            self.write_string(value)
        elif tag_type == TAG_LIST:
            item_type, items = value
            self.write_u8(item_type)
            self.write_i32(len(items))
            for item in items:
                self.write_payload(item_type, item)
        elif tag_type == TAG_COMPOUND:
            for name, tag in value:
                self.write_u8(tag.tag_type)
                self.write_string(name)
                self.write_payload(tag.tag_type, tag.value)
            self.write_u8(TAG_END)
        elif tag_type == TAG_INT_ARRAY:
            self.write_i32(len(value))
            for item in value:
                self.write_i32(item)
        elif tag_type == TAG_LONG_ARRAY:
            self.write_i32(len(value))
            for item in value:
                self.write_i64(item)
        else:
            raise ValueError(f"unsupported NBT tag type {tag_type}")

    def write_root(self, tag_type: int, name: str, value: Any) -> bytes:
        self.write_u8(tag_type)
        if tag_type != TAG_END:
            self.write_string(name)
            self.write_payload(tag_type, value)
        return self.out.getvalue()


def patch_block_state(value: str) -> tuple[str, bool]:
    replacement = BLOCK_RENAMES.get(value)
    if replacement is not None:
        return replacement, True

    if "[" in value:
        block, rest = value.split("[", 1)
        replacement = BLOCK_RENAMES.get(block)
        if replacement is not None:
            return f"{replacement}[{rest}", True

    return value, False


def patch_tag(tag_type: int, value: Any) -> bool:
    changed = False
    if tag_type == TAG_STRING:
        return False
    if tag_type == TAG_LIST:
        item_type, items = value
        if item_type == TAG_STRING:
            for index, item in enumerate(items):
                patched, did_change = patch_block_state(item)
                if did_change:
                    items[index] = patched
                    changed = True
            return changed
        for item in items:
            changed |= patch_tag(item_type, item)
        return changed
    if tag_type == TAG_COMPOUND:
        for index, (name, child) in enumerate(value):
            if child.tag_type == TAG_STRING:
                patched, did_change = patch_block_state(child.value)
                if did_change:
                    value[index] = (name, Tag(TAG_STRING, patched))
                    changed = True
            else:
                changed |= patch_tag(child.tag_type, child.value)
    return changed


def patch_nbt_bytes(content: bytes) -> tuple[bytes, bool]:
    compressed = content.startswith(b"\x1f\x8b")
    raw = gzip.decompress(content) if compressed else content

    reader = NbtReader(raw)
    root_type, root_name, root_value = reader.read_root()
    changed = patch_tag(root_type, root_value)
    if not changed:
        return content, False

    writer = NbtWriter()
    patched_raw = writer.write_root(root_type, root_name, root_value)
    if compressed:
        return gzip.compress(patched_raw), True
    return patched_raw, True


def output_path_for(source: Path, suffix: str) -> Path:
    if source.name.endswith(".zip"):
        return source.with_name(f"{source.name[:-4]}{suffix}.zip")
    return source.with_name(f"{source.name}{suffix}")


def copy_info(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    copied = zipfile.ZipInfo(info.filename, info.date_time)
    copied.comment = info.comment
    copied.extra = info.extra
    copied.internal_attr = info.internal_attr
    copied.external_attr = info.external_attr
    copied.create_system = info.create_system
    copied.compress_type = info.compress_type
    return copied


def fix_zip(source: Path, suffix: str, force: bool) -> None:
    destination = output_path_for(source, suffix)
    if destination.exists() and not force:
        print(f"skip {source.name}: {destination.name} already exists", file=sys.stderr)
        return

    changed_entries = 0
    failed_entries: list[str] = []

    with zipfile.ZipFile(source, "r") as zin, zipfile.ZipFile(destination, "w") as zout:
        for info in zin.infolist():
            content = zin.read(info.filename)
            output = content
            if not info.is_dir() and info.filename.endswith(".nbt"):
                try:
                    output, changed = patch_nbt_bytes(content)
                    if changed:
                        changed_entries += 1
                except Exception as error:  # noqa: BLE001 - keep the datapack copy usable.
                    failed_entries.append(f"{info.filename}: {error}")
                    output = content

            zout.writestr(copy_info(info), output)

    print(f"{source.name} -> {destination.name}: fixed {changed_entries} NBT files")
    for failure in failed_entries:
        print(f"  warning: kept original {failure}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zips", nargs="*", type=Path, help="datapack zip(s) to fix")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory to scan when no zip paths are provided",
    )
    parser.add_argument("--suffix", default="-nbt-fixed", help="output suffix before .zip")
    parser.add_argument("--force", action="store_true", help="overwrite existing fixed copies")
    args = parser.parse_args()

    zips = args.zips
    if not zips:
        zips = sorted(
            path
            for path in args.dir.glob("*.zip")
            if not path.name.endswith(f"{args.suffix}.zip")
        )

    if not zips:
        print("no datapack zips found", file=sys.stderr)
        return 1

    for source in zips:
        if not source.is_file():
            print(f"missing zip: {source}", file=sys.stderr)
            return 1
        fix_zip(source, args.suffix, args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
