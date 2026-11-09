---
name: capcut-draft-tech
description: Technical execution engine for CapCut macOS projects: deterministic source-duration model detection (Seedance, Grok, Omni, Hybrid), ground-truth BANGTRA/PROMPT_ANH_REF/sf-board.json scene & lighting regime mapping, parallel 17-metric source_timerange scope diagnostics with neutral_rb white-balance probe and 3-Shot Scene Strip rendering, AI Grading Plan (--plan) execution, 3D-trilinear optical LUT selection with CapCut skin protection, and full synchronization across Root, Timelines, mini_draft.json, and Compound Clip subdrafts. Pair with capcut-ai-colorist.
---

# CapCut Draft Technical Engine (`capcut-draft-tech`)

## Overview

`capcut-draft-tech` is the precision DI (Digital Intermediate) console and CapCut draft synchronization engine used by **`capcut-ai-colorist`**. It measures per-segment scopes, renders Multi-Shot Scene Strip contact sheets for the AI Colorist to visually inspect, executes the AI Colorist's scene & shot grading plan (`--plan`), solves intra-scene shot-to-shot continuity, enables CapCut's neural skin-tone protection (`enable_skin_tone_correction: true`), and synchronizes all CapCut draft mirrors.

---

## 1. Bundled Scripts & Optical Database

All scripts and resources live inside this skill folder:
- [`scripts/analyze_project.py`](scripts/analyze_project.py):
  - Auto-detects the AI video model (`seedance`, `grok`, `omni`, `hybrid`) from un-cut source video durations (`video suộc gốc`).
  - Discovers ground-truth scene maps (`*BANGTRA*.txt`, `*PROMPT_ANH_REF*.txt`, `*PROMPT*.txt`, `sf-board.json`), scoring candidate tables by clip count and duration column (`30s` vs `10s`).
  - Merges reverse camera angles (`"góc"` $\rightarrow$ `BG<N>`) while keeping distinct sub-locations separate, and classifies each scene's initial lighting regime (`DAWN_DUSK_GOLD`, `DAY_CRISP`, `WARM_INT`, `NIGHT_WARM`, `NIGHT_COOL`, `COOL_TECH`, `is_flashback`).
  - Extracts 17-metric segment diagnostics at exact `source_timerange` in parallel (`ThreadPoolExecutor(max_workers=16)` with single-threaded `-threads 1` FFmpeg decoders at `216x122`), sampling **3 temporal points (`20% Entry`, `50% Center`, `80% Exit`)** per segment in a single FFmpeg pass to reject transient flashes/occlusions and record cut-boundary metrics (`entry_mid`, `exit_mid`, `entry_nrb`, `exit_nrb`, `intra_drift`), plus `neutral_rb` and `neutral_gm` (achromatic white-balance probe on lowest-30% chroma pixels) to separate true ambient lighting casts from colorful props/costumes/brick walls.
  - Identifies each scene's `first_seg`, `anchor_seg`, and `outlier_seg` (`outlier_delta_mid`, `outlier_delta_rb`), caches scopes to `/tmp/capcut_metrics_<slug>.json` (with backwards-compatible top-level aliases) and decoded `uint8` frames to `/tmp/capcut_frames_<slug>.npy`, renders the **3-Shot Scene Strip Contact Sheet** (`/tmp/<Project>_raw_scenes.jpg` + high-res paginated `_p1.jpg.._pN.jpg`), and automatically generates:
    1. **`/tmp/<Project>_scene_report.txt`**: Complete human-readable Per-Segment Scope Report (`Seg #0..#N-1` with `Mid`, `P5`, `P95`, `RB`, `NRB`, `SkRB`, `Sat`, and `[OUTLIER]` flags) + Curated Clean Neutral 35mm LUT Palette table from `lut_optical_db.json` (eliminating any need for ad-hoc `python3` inspection commands).
    2. **`/tmp/capcut_plan_<slug>.json`**: Pre-populated Starter AI Grading Plan JSON for all scenes with location-locked clean 35mm LUTs, `target_mid`, `preserve_warm_props`, and automatic biases so the AI Colorist can refine it via fast `edit_file` calls (`~200–500` tokens) instead of writing 250 lines of JSON from scratch.
- [`scripts/apply_grade.py`](scripts/apply_grade.py):
  - Reuses `/tmp/capcut_metrics_<slug>.json` and `/tmp/capcut_frames_<slug>.npy` automatically and accepts the AI Colorist's bespoke **Scene & Shot Grading Plan (`--plan /tmp/capcut_plan_<slug>.json`)**.
  - Enforces regime-aware LUT tint guardrails (blocking heavy cyan/teal LUTs with `midRB < -0.055` or `midGM > +0.018` from warm/daylight scenes) and gates orange-cast cooling on `neutral_rb > 0.082`.
  - Automatically handles **3-Axis Sub-Space Partitioning & Cross-Space Cutaway Isolation (`partition_intra_scene_subspaces`)**:
    - Axis 1 (Exposure/Midtone bimodal split with `>= 2` segments per mode AND internal `std <= 0.048`),
    - Axis 2 (Ambient color `neutral_rb` split with `>= 2` segments per mode AND internal `std <= 0.025`),
    - Axis 3 (**Cross-Space Cutaway Isolation** for parallel/transition scenes with `↔`, `→`, `song song`, or multi-axis room jumps where `|dMid| > 0.145` and `|dContrast| > 0.10` or `|dNRB| > 0.032` from the main room, isolating 1-2 shot cutaways onto their own sub-space anchor while pulling ordinary 1-shot same-room AI glitches back into the main space).
  - Applies **Stage 1 (Scene-Locked LUT at `zorder: 2000` with `enable_skin_tone_correction: true` and micro-adaptive `±2.5%` intensity)** + **Stage 2 (Bounded Scene Baselines + Optical LUT Midtone Shift Compensation `lut_mid_err` + Continuous Piecewise-Linear Two-Tier Shot Continuity using `65%` Sub-Space Anchor + `35%` Same-Space Boundary Neighbor & Multi-Cue Illumination/Skin Probe `d_nrb + d_skin_rb + d_shd_rb` + Closed-Loop Pixel-Level 3D LUT Feedback Trim `refine_grades_closed_loop_on_frames` at `zorder: 8004`)** to every child video segment (`video con`), preserving explicit AI per-segment `bright_offset` / `temp_offset` overrides.
  - Promotes active LUTs into `LutImportCfg.json[0:100]` (`index <= 99`) using parallel multi-threaded regex scanning across all local CapCut drafts so CapCut's Adjustment $\rightarrow$ LUT dropdown never displays `"None"`.
  - Synchronizes `draft_info.json`, `draft_info.json.bak`, `template-2.tmp`, `Timelines/*/attachment/patch/mini_draft.json` (`patch.json` reset), and `subdraft/*/draft_content.json`.
  - Renders both the **4-Tile Before/After Comparison Sheet** (`/tmp/<Project>_graded_comparison.jpg` + `_p1.jpg.._pN.jpg`: `1.Anc RAW | 2.Anc GRD | 3.Out RAW | 4.Out GRD` per scene) and the **Consecutive Shot-to-Shot Continuity Strip** (`/tmp/<Project>_shot_continuity.jpg`: up to 9 consecutive cuts `RAW` vs `GRADED` for the Opening Hook + top multi-shot scenes) in `~2.9s` total with 0 extra FFmpeg calls.
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

## 3. End-to-End 5-Step CLI & Visual Workflow

> **CRITICAL RULES (YAGNI & 10–20 MINUTE PROFESSIONAL DI BUDGET)**:
> 1. **Time Budget (`10–20 Minutes`, Strictly `<= 20 Minutes`)**: Dedicate the 10–20 minute window to thorough visual inspection of all high-resolution paginated scene strips (`/tmp/<Project>_raw_scenes_p*.jpg`), `/tmp/<Project>_scene_report.txt`, screenplay context, targeted `edit_file` refinements on `/tmp/capcut_plan_<slug>.json`, and multi-page visual QA (`/tmp/<Project>_graded_comparison_p*.jpg` + `/tmp/<Project>_shot_continuity.jpg`).
> 2. **ZERO-PYTHON-PROBING RULE**: Never edit or rewrite `analyze_project.py` or `apply_grade.py` during a project grading session, and **NEVER** run `python3 -c` or heredocs to inspect `/tmp/capcut_metrics_<slug>.json`, `lut_optical_db.json`, or `sf-board.json`. Read `/tmp/<Project>_scene_report.txt` and `/tmp/capcut_plan_<slug>.json` directly with `read_file`.
> 3. **Use `edit_file` on `/tmp/capcut_plan_<slug>.json`**: Modify the pre-generated starter plan via `edit_file` rather than rewriting 250 lines of JSON from scratch with `write_file`.

```bash
# Step 1: Analyze project, auto-detect model, map scenes, cache 17-metric scopes,
# render 3-shot strips (/tmp/<Project>_raw_scenes_p*.jpg), write /tmp/<Project>_scene_report.txt
# and pre-generate /tmp/capcut_plan_<slug>.json
python3 "/Users/daho100/Documents/SF Board/.agents/skills/capcut-draft-tech/scripts/analyze_project.py" "<PROJECT_NAME_OR_ID>"

# Step 2 & 3: Call read_file in parallel on /tmp/<Project>_raw_scenes_p*.jpg,
# /tmp/<Project>_scene_report.txt, and /tmp/capcut_plan_<slug>.json,
# then use edit_file on /tmp/capcut_plan_<slug>.json to refine scene & outlier-shot grading direction.

# Step 4: Apply 35mm grade with --plan, sync all CapCut draft mirrors & render 4-tile comparison + continuity sheets
python3 "/Users/daho100/Documents/SF Board/.agents/skills/capcut-draft-tech/scripts/apply_grade.py" "<PROJECT_NAME_OR_ID>" \
  --plan "/tmp/capcut_plan_<slug>.json" \
  --emotion narrative-arc \
  --auto-match-outliers \
  --relaunch

# Step 5: Visually inspect /tmp/<Project>_shot_continuity.jpg and /tmp/<Project>_graded_comparison_p*.jpg with read_file.
```

### Schema for `--plan /tmp/capcut_plan_<slug>.json`
- `"global"`: `{"temp_bias": 0.0, "bright_bias": 0.0, "contrast_bias": 0.0, "sat_bias": 0.0, "lut_val_bias": 0.0}`
- `"scenes"`: Keyed by cluster ID (`"0"`, `"1"`, or `"S00"`) or background code (`"S06:BG4"`, `"BG4"`):
  - `"regime"`: Override lighting regime (`"DAWN_DUSK_GOLD"`, `"DAY_CRISP"`, `"WARM_INT"`, `"NIGHT_WARM"`, `"NIGHT_COOL"`, `"COOL_TECH"`).
  - `"lut"` / `"lut_val"`: Explicit Primary `.cube` LUT title and intensity (`0.18..0.24`), OR `"preferred_families": ["SKINTONE", "THE ROMANCE", "FILM LOOK"]`.
  - `"target_mid"`, `"target_rb"`, `"target_p5"`, `"target_hi"`, `"target_sat"` (or relative shifts `"mid_shift"`, `"rb_shift"`, `"sat_shift"`).
  - `"preserve_warm_props"`: `true`/`false` (prevents warm hats/brick/wood/costumes from triggering orange-cast cooling).
  - `"temp_bias"`, `"bright_bias"`, `"contrast_bias"`, `"sat_bias"`, `"shadow_bias"`, `"highlight_bias"`, `"tone_bias"`: Additive post-LUT Adjust biases for all shots in the scene.
  - `"note"`: AI Colorist visual diagnosis note printed in the summary and comparison sheet.
- `"segments"`: Keyed by segment index (`"59"`) or clip name (`"43.mp4"`):
  - `"force_anchor_match": true`, `"lut"`, `"lut_val"`, `"bright_offset"`, `"temp_offset"`, `"sat_offset"`, `"contrast_offset"`, `"shadow_offset"`, `"highlight_offset"`, `"tone_offset"`.

---

## 4. CapCut `materials.effects` & Draft Synchronization Reference

1. **14-Parameter Schema & Version Map**:
   - Every effect uses `effect_id = resource_id = "7501974767453474064"`.
   - Exact `version` strings: `lut: ""`, `brightness: "v2"`, `contrast: "v3"`, `highlight: "v3"`, `shadow: "v3"`, `light_sensation: ""`, `saturation: "v1"`, `temperature: "v3"`, `tone: "v3"`, `fade: ""`, `sharpen: "v1"`, `clear: ""`, `particle: "v2"`, `vignetting: "v1"`.
   - On every `lut` effect, `"enable_skin_tone_correction": true` activates CapCut's neural `tt_skin_seg` skin protection mask (`Bảo vệ tông màu da`).
2. **Top-100 `LutImportCfg.json` Promotion (`index 0..99`)**:
   - CapCut's C++ `ImportLUTDataManager::loadImportedLutList` (`cmp x8, #0x63; b.hi`) and `VELutSpinComboBoxViewModel::createPopupItems` (`cmp w27, #0x63; b.gt`) only load indices `0..99` into the UI dropdown. `promote_active_luts_in_cfg()` automatically places all active project LUTs at `index 0..K-1`.
3. **Full Draft Mirror Sync**:
   - Writes `draft_info.json`, `draft_info.json.bak`, `template-2.tmp` in both root and `Timelines/<UUID>/`, updates `Timelines/<UUID>/attachment/patch/mini_draft.json` (`patch.json` reset to `{"patch_data":[]}`), and syncs `subdraft/<UUID>/draft_content.json` for Compound Clips.
