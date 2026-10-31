#!/usr/bin/env python3
"""Install capcut-ai-colorist & capcut-draft-tech skills and register core 35mm LUTs in CapCut."""
import argparse
import datetime
import filecmp
import json
import os
import pathlib
import shutil
import tempfile

SKILL_NAMES = ["capcut-ai-colorist", "capcut-draft-tech"]
LUT_CFG_PATH = pathlib.Path.home() / "Movies/CapCut/User Data/Resources/Lut/LutImportCfg.json"


def same_tree(left: pathlib.Path, right: pathlib.Path) -> bool:
    def included(path: pathlib.Path, base: pathlib.Path) -> bool:
        return (
            path.is_file()
            and "__pycache__" not in path.relative_to(base).parts
            and path.suffix != ".pyc"
            and path.name != ".DS_Store"
        )

    a = {p.relative_to(left) for p in left.rglob("*") if included(p, left)}
    b = {p.relative_to(right) for p in right.rglob("*") if included(p, right)}
    return a == b and all(filecmp.cmp(left / p, right / p, shallow=False) for p in a)


def install_one_skill(source: pathlib.Path, skills_dir: pathlib.Path, name: str):
    source = source.resolve()
    skills_dir = skills_dir.expanduser().resolve()
    if not (source / "SKILL.md").is_file():
        raise ValueError(f"Source does not contain SKILL.md: {source}")
    destination = skills_dir / name
    if destination.is_symlink():
        raise ValueError(f"Refusing to replace symlink: {destination}")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Destination is not a directory: {destination}")
    if destination.exists() and same_tree(source, destination):
        return destination, None, False
    skills_dir.mkdir(parents=True, exist_ok=True)
    backup = None
    with tempfile.TemporaryDirectory(prefix=f".{name}-install-", dir=skills_dir) as temp:
        staged = pathlib.Path(temp) / name
        shutil.copytree(source, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        if destination.exists():
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            backup = skills_dir.parent / "skill-backups" / f"{name}-{stamp}"
            backup.parent.mkdir(parents=True, exist_ok=True)
            destination.rename(backup)
        try:
            staged.rename(destination)
        except OSError:
            if backup and not destination.exists():
                backup.rename(destination)
            raise
    return destination, backup, True


def register_bundled_luts(luts_src_dir: pathlib.Path) -> int:
    if not luts_src_dir.is_dir():
        return 0
    lut_dir = LUT_CFG_PATH.parent
    lut_dir.mkdir(parents=True, exist_ok=True)
    cfg = []
    if LUT_CFG_PATH.is_file():
        try:
            cfg = json.loads(LUT_CFG_PATH.read_text(encoding="utf-8"))
        except Exception:
            cfg = []
    existing = {it.get("title"): it.get("path") for it in cfg if it.get("path") and os.path.exists(it["path"])}
    added = 0
    for cube in sorted(luts_src_dir.iterdir()):
        if cube.suffix.lower() not in (".cube", ".3dl"):
            continue
        if cube.name not in existing:
            dst = lut_dir / cube.name
            if not dst.exists():
                shutil.copy2(cube, dst)
            cfg = [it for it in cfg if it.get("title") != cube.name]
            cfg.append({"path": str(dst), "title": cube.name})
            added += 1
    if added > 0 or not LUT_CFG_PATH.is_file():
        LUT_CFG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".agents" / "skills",
        help="Skills parent directory; defaults to ~/.agents/skills",
    )
    args = parser.parse_args()
    repo_root = pathlib.Path(__file__).resolve().parent
    for name in SKILL_NAMES:
        src = repo_root / "skills" / name
        dest, backup, changed = install_one_skill(src, args.dest, name)
        print(("Installed: " if changed else "Already current: ") + str(dest))
        if backup:
            print("  Previous version backed up: " + str(backup))

    added_luts = register_bundled_luts(repo_root / "skills" / "capcut-draft-tech" / "resources" / "luts")
    print(f"Core 35mm LUTs verified in CapCut ({added_luts} newly registered).")
    print("Invoke $capcut-ai-colorist in Codex / Antigravity.")


if __name__ == "__main__":
    main()
