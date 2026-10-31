#!/usr/bin/env python3
"""Import a folder of .cube / .3dl LUT files into CapCut on macOS."""
import argparse
import json
import os
import shutil
import uuid

LUT_DIR = "/Users/daho100/Movies/CapCut/User Data/Resources/Lut"
CFG_PATH = os.path.join(LUT_DIR, "LutImportCfg.json")


def main():
    parser = argparse.ArgumentParser(description="Import .cube/.3dl LUTs into CapCut macOS.")
    parser.add_argument("source_dir", help="Directory containing .cube or .3dl LUT files (searched recursively).")
    args = parser.parse_args()

    os.makedirs(LUT_DIR, exist_ok=True)
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = []

    existing_titles = {item.get("title") for item in cfg}
    added = 0

    for root, _, files in os.walk(args.source_dir):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in (".cube", ".3dl"):
                continue
            src_path = os.path.join(root, fname)
            title = fname
            if title in existing_titles:
                rel_parent = os.path.basename(root)
                title = f"{rel_parent}_{fname}"
            if title in existing_titles:
                continue

            new_name = f"{uuid.uuid4().hex[:32]}{ext}"
            dst_path = os.path.join(LUT_DIR, new_name)
            shutil.copy2(src_path, dst_path)

            cfg.append({
                "group_id": -1,
                "path": dst_path,
                "title": title,
            })
            existing_titles.add(title)
            added += 1

    with open(CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)

    print(f"Imported {added} new LUTs into CapCut (Total LUTs: {len(cfg)}).")


if __name__ == "__main__":
    main()
