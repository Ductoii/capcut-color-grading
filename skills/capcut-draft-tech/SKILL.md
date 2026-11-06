---
name: capcut-draft-tech
description: Technical execution engine for CapCut macOS projects: deterministic source-duration model detection (Seedance, Grok, Omni, Hybrid), ground-truth BANGTRA/PROMPT_ANH_REF/sf-board.json scene & lighting regime mapping, parallel segment-level source_timerange ffmpeg diagnostics with automatic /tmp caching, 3D-trilinear optical LUT selection, non-canceling 14-parameter CapCut materials.effects JSON injection, and full synchronization across Root, Timelines, mini_draft.json, and Compound Clip subdrafts. Pair with capcut-ai-colorist.
---

# CapCut Draft Technical Engine (`capcut-draft-tech`)

## Overview

`capcut-draft-tech` provides the deterministic execution engine for grading CapCut macOS timelines in **under 30 seconds**. Pair with **`capcut-ai-colorist`** for colorist principles and model-specific conditioning.

---

## 1. Bundled Scripts & Optical Database

All scripts and resources live inside this skill folder:
- [`scripts/analyze_project.py`](scripts/analyze_project.py):
  - Auto-detects the AI video model (`seedance`, `grok`, `omni`, `hybrid`) from un-cut source video durations (`video suộc gốc`).
  - Discovers ground-truth scene maps (`*BANGTRA*.txt`, `*PROMPT_ANH_REF*.txt`, `*PROMPT*.txt`, `sf-board.json`), scoring candidate tables by clip count and duration column (`30s` vs `10s`).
  - Merges reverse camera angles (`"góc"` $\rightarrow$ `BG<N>`) while keeping distinct sub-locations separate, and classifies each scene's lighting regime (`DAWN_DUSK_GOLD`, `DAY_CRISP`, `WARM_INT`, `NIGHT_WARM`, `NIGHT_COOL`, `COOL_TECH`, `is_flashback`).
  - Extracts 15-metric segment diagnostics at exact `source_timerange` in parallel (`ThreadPoolExecutor(max_workers=12)`), caches to `/tmp/capcut_metrics_<slug>.json`, and renders `/tmp/<Project>_raw_scenes.jpg`.
- [`scripts/apply_grade.py`](scripts/apply_grade.py):
  - Reuses `/tmp/capcut_metrics_<slug>.json` automatically (or extracts missing segments in parallel).
  - Applies **Stage 1 (LUT at `zorder: 2000` with `enable_skin_tone_correction: true`)** + **Stage 2 (Non-Canceling 12 Adjust Sliders at `zorder: 8004`)** to every child video segment (`video con`), never double-stacking onto Compound Clip wrappers or photo overlays.
  - Promotes active LUTs into `LutImportCfg.json[0:100]` (`index <= 99`) using fast regex scanning so CapCut's Adjustment $\rightarrow$ LUT dropdown never displays `"None"`.
  - Synchronizes `draft_info.json`, `draft_info.json.bak`, `template-2.tmp`, `Timelines/*/attachment/patch/mini_draft.json` (`patch.json` reset), and `subdraft/*/draft_content.json`.
  - Renders `/tmp/<Project>_graded_comparison.jpg` using cached 3D trilinear `.cube` interpolation in ~2–4 seconds.
- [`scripts/import_luts.py`](scripts/import_luts.py):
  - Bulk-imports `.cube` / `.3dl` LUTs into `/Users/daho100/Movies/CapCut/User Data/Resources/Lut/LutImportCfg.json`.
- [`resources/lut_optical_db.json`](resources/lut_optical_db.json):
  - Pre-measured 3D-trilinear optical transfer database for all 1,636 `.cube` LUTs (`blk`, `shd`, `mid`, `hi`, `shdRB`, `midRB`, `hiRB`, `midGM`, `hiGM`, `sknRB`, `sknGM`, true multi-patch `satGain`, and `tier`).

---

## 2. Deterministic Source-Duration Model Detection (`detect_source_models`)

`detect_source_models(all_seg_tuples, proj_dir)` inspects the un-cut duration of every unique source video (`materials.videos[].duration` in microseconds, falling back to `ffprobe`):
- **No source video `> 10.5s`** (`max <= 10.5s`, typically `10s`) $\rightarrow$ **`omni`** (`Google Omni 1.1`).
- **Source videos span `6s / 10s / 15s`** (has `10.5s < dur <= 15.5s`, none `> 15.5s`) $\rightarrow$ **`grok`** (`Grok Video`).
- **Has source video(s) `> 15.5s` (`~25s–30s`)**:
  - If `>= 65%` of unique source videos are `<= 10.5s` (`10s` body clips with a `30s` opening hook) $\rightarrow$ **`hybrid`**.
  - Otherwise $\rightarrow$ **`seedance`** (with per-clip `clip_model` tagging any `6s–15s` replacement clips inside the timeline).

---

## 3. Standard CLI Execution (< 30 Seconds End-to-End)

> **CRITICAL RULE (YAGNI)**: Never edit or rewrite `analyze_project.py` or `apply_grade.py` during a project grading session. Use CLI flags (`--emotion`, `--model`, `--match-clips`, `--plan`, `--allow-stylized`, `--allow-bw`, `--relaunch`) for any customization.

```bash
# Step 1: Analyze project, auto-detect model, map scenes & cache metrics (~15-30s)
python3 "/Users/daho100/Documents/SF Board/.agents/skills/capcut-draft-tech/scripts/analyze_project.py" "<PROJECT_NAME_OR_ID>"

# Step 2: Apply 35mm grade, sync all CapCut draft mirrors & render comparison sheet (~2-5s)
python3 "/Users/daho100/Documents/SF Board/.agents/skills/capcut-draft-tech/scripts/apply_grade.py" "<PROJECT_NAME_OR_ID>" \
  --emotion narrative-arc \
  --auto-match-outliers \
  --relaunch
```

### Optional CLI Flags for `apply_grade.py`
- `--model {auto,seedance,grok,omni,hybrid}`: Defaults to `auto` (source-duration detection).
- `--emotion {narrative-arc,warm-human,dramatic-contrast,luxury-crisp,vintage-35mm}`: Emotional grading direction (default: `narrative-arc`).
- `--auto-match-outliers`: Boosts local neighbor convergence on intra-scene outlier cuts.
- `--match-clips "12.mp4,15.mp4"`: Forces specific clips to strongly match immediate neighbors.
- `--allow-stylized`: Unlocks Tier 2 (`stylized_rec709`) LUTs alongside Tier 1 (`core_rec709`).
- `--allow-bw`: Unlocks Tier 3 (`bw_mono`) LUTs for flashback/noir scenes (`is_flashback=True`).
- `--plan /path/to/plan.json`: Applies custom per-segment LUT/Adjust overrides without modifying code.

---

## 4. CapCut `materials.effects` & Draft Synchronization Reference

1. **14-Parameter Schema & Version Map**:
   - Every effect uses `effect_id = resource_id = "7501974767453474064"`.
   - Exact `version` strings: `lut: ""`, `brightness: "v2"`, `contrast: "v3"`, `highlight: "v3"`, `shadow: "v3"`, `light_sensation: ""`, `saturation: "v1"`, `temperature: "v3"`, `tone: "v3"`, `fade: ""`, `sharpen: "v1"`, `clear: ""`, `particle: "v2"`, `vignetting: "v1"`.
   - On every `lut` effect, `enable_skin_tone_correction: True` activates CapCut's neural `tt_skin_seg` skin protection mask.
2. **Top-100 `LutImportCfg.json` Promotion (`index 0..99`)**:
   - CapCut's C++ `ImportLUTDataManager::loadImportedLutList` (`cmp x8, #0x63; b.hi`) and `VELutSpinComboBoxViewModel::createPopupItems` (`cmp w27, #0x63; b.gt`) only load indices `0..99` into the UI dropdown. `promote_active_luts_in_cfg()` automatically places all active project LUTs at `index 0..K-1`.
3. **Full Draft Mirror Sync**:
   - Writes `draft_info.json`, `draft_info.json.bak`, `template-2.tmp` in both root and `Timelines/<UUID>/`, updates `Timelines/<UUID>/attachment/patch/mini_draft.json` (`patch.json` reset to `{"patch_data":[]}`), and syncs `subdraft/<UUID>/draft_content.json` for Compound Clips.
