#!/usr/bin/env python3
"""Install capcut-ai-colorist & capcut-draft-tech skills and register all 1,636 LUTs (Hồ Minh Quân pack) in CapCut."""
import argparse
import datetime
import filecmp
import json
import os
import pathlib
import shutil
import tarfile
import tempfile
import urllib.request

SKILL_NAMES = ["capcut-ai-colorist", "capcut-draft-tech"]
LUT_CFG_PATH = pathlib.Path.home() / "Movies/CapCut/User Data/Resources/Lut/LutImportCfg.json"
FULL_LUT_RELEASE_URL = "https://github.com/Ductoii/capcut-color-grading/releases/download/v1.0.0/luts_full_1636.tar.gz"


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


def register_all_luts(repo_root: pathlib.Path, download_full: bool = True) -> tuple[int, int]:
    lut_dir = LUT_CFG_PATH.parent
    lut_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy bundled core LUTs first
    luts_src_dir = repo_root / "skills" / "capcut-draft-tech" / "resources" / "luts"
    if luts_src_dir.is_dir():
        for cube in sorted(luts_src_dir.iterdir()):
            if cube.suffix.lower() in (".cube", ".3dl"):
                dst = lut_dir / cube.name
                if not dst.exists():
                    shutil.copy2(cube, dst)

    # 2. Check how many LUT files currently exist in CapCut's Lut directory
    current_files = [
        p for p in lut_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".cube", ".3dl")
    ]
    if download_full and len(current_files) < 1600:
        local_tar = repo_root / "luts_full_1636.tar.gz"
        tar_path = str(local_tar) if local_tar.is_file() else ""
        tmp_tar = None
        if not tar_path:
            try:
                print(f"Downloading full 1,636 LUT pack from {FULL_LUT_RELEASE_URL} ...")
                fd, tmp_tar = tempfile.mkstemp(suffix=".tar.gz")
                os.close(fd)
                req = urllib.request.Request(FULL_LUT_RELEASE_URL, headers={"User-Agent": "capcut-color-grading"})
                with urllib.request.urlopen(req, timeout=300) as resp, open(tmp_tar, "wb") as out_f:
                    shutil.copyfileobj(resp, out_f)
                tar_path = tmp_tar
            except Exception as exc:
                print(f"Note: Could not download full LUT release archive ({exc}); using bundled LUTs.")
        if tar_path and os.path.exists(tar_path):
            print("Extracting full 1,636 LUT pack into CapCut Resources/Lut ...")
            with tarfile.open(tar_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    bname = os.path.basename(member.name)
                    if bname.lower().endswith((".cube", ".3dl")):
                        dst_file = lut_dir / bname
                        if not dst_file.exists():
                            src_f = tf.extractfile(member)
                            if src_f:
                                with open(dst_file, "wb") as out_f:
                                    shutil.copyfileobj(src_f, out_f)
        if tmp_tar and os.path.exists(tmp_tar):
            os.unlink(tmp_tar)

    # 3. Sync LutImportCfg.json with all .cube / .3dl files in lut_dir
    cfg = []
    if LUT_CFG_PATH.is_file():
        try:
            cfg = json.loads(LUT_CFG_PATH.read_text(encoding="utf-8"))
        except Exception:
            cfg = []
    cfg = [it for it in cfg if it.get("path") and os.path.exists(it["path"])]
    existing_titles = {it.get("title") for it in cfg}
    added = 0
    for cube in sorted(lut_dir.iterdir(), key=lambda x: x.name.lower()):
        if cube.is_file() and cube.suffix.lower() in (".cube", ".3dl"):
            if cube.name not in existing_titles:
                cfg.append({"path": str(cube), "title": cube.name})
                existing_titles.add(cube.name)
                added += 1
    LUT_CFG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(cfg), added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".agents" / "skills",
        help="Skills parent directory; defaults to ~/.agents/skills",
    )
    parser.add_argument(
        "--no-download-luts",
        action="store_true",
        help="Skip downloading the full 1,636 LUT release pack if not already present locally.",
    )
    args = parser.parse_args()
    repo_root = pathlib.Path(__file__).resolve().parent
    for name in SKILL_NAMES:
        src = repo_root / "skills" / name
        dest, backup, changed = install_one_skill(src, args.dest, name)
        print(("Installed: " if changed else "Already current: ") + str(dest))
        if backup:
            print("  Previous version backed up: " + str(backup))

    total_luts, added_luts = register_all_luts(repo_root, download_full=not args.no_download_luts)
    print(f"CapCut LUT library ready: {total_luts} total LUTs registered ({added_luts} newly added).")
    print("Invoke $capcut-ai-colorist in Codex / Antigravity.")


if __name__ == "__main__":
    main()
