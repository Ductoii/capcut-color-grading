---
name: capcut-draft-tech
description: >-
  Technical execution engine for CapCut macOS projects: bulk-importing .cube/.3dl LUTs
  into LutImportCfg.json, extracting multi-frame 16-metric ffmpeg video diagnostics,
  constructing 14-parameter CapCut materials.effects JSON payloads, and safely syncing
  root drafts, Timelines, and Compound Clip subdrafts. Pair with capcut-ai-colorist
  for pre-grading user intake, emotional direction, and colorist decisions.
---

# CapCut macOS Technical Engine: Multi-Frame Video Diagnostics, 14-Parameter Shader Schema & Draft Sync

This skill covers **only** the technical mechanics of analyzing videos and reading/writing CapCut macOS project files (`com.lveditor.draft/<PROJECT_ID>`). Always follow the **Pre-Grading User Intake** and colorist rules in **`capcut-ai-colorist`** before running `apply_grade.py`.

---

## 1. CLI Workflow (`/Users/daho100/Documents/SF Board/.agents/skills/capcut-draft-tech/scripts/`)

### Step 1 — Run Read-Only Multi-Frame Video Analysis (Before Asking User)
```bash
python3 "/Users/daho100/Documents/SF Board/.agents/skills/capcut-draft-tech/scripts/analyze_project.py" \
  "1026 - THU SKILL MOI" \
  --out "/tmp/metrics_1026.json"
```
*(After running Step 1, **STOP and ask the user** the 2 mandatory questions defined in `capcut-ai-colorist`: Model Source/Outliers & Emotional Direction).*

### Step 2 — Apply Empirical Per-Clip Grading + 12-Slider Adjust Emulation (After User Confirms)
```bash
python3 "/Users/daho100/Documents/SF Board/.agents/skills/capcut-draft-tech/scripts/apply_grade.py" \
  "1026 - THU SKILL MOI" \
  --metrics "/tmp/metrics_1026.json" \
  --emotion "narrative-arc" \
  --relaunch
```
- Supported `--emotion` modes:
  - `narrative-arc` (default: adapts target exposure, warmth, and LUT dynamically across Opening -> Conflict -> Climax -> Finale)
  - `warm-human` (warm, empathetic, rosy skin & golden highlights)
  - `dramatic-contrast` (deep shadows, crisp teal/warm separation, subtle vignette)
  - `luxury-crisp` (clean whites, golden brass warmth, high clarity)
  - `vintage-35mm` (rich Kodak 35mm grain, warm halation, soft contrast roll-off)
- Optional flags:
  - `--match-clips "1.mp4,17(2).mp4"`: Explicitly color-match listed clips to their immediate same-scene neighbors.
  - `--auto-match-outliers`: Automatically blend any clip whose `|dRB| > 0.055` or `|dMid| > 0.065` jumps sharply against both immediate neighbors.
  - `--plan "/path/to/custom_plan.json"`: Override specific segments or the entire timeline with an explicit AI-authored JSON grading plan.

### Utility — Bulk-Import `.cube` / `.3dl` LUTs into CapCut
```bash
python3 "/Users/daho100/Documents/SF Board/.agents/skills/capcut-draft-tech/scripts/import_luts.py" \
  "/path/to/lut_folder"
```

---

## 2. Complete 14-Parameter CapCut Adjust & LUT Shader Schema (`materials.effects`)

CapCut's `7501974767453474064` (`lumi_hub_path` / `DefaultAdjustBundle`) color engine supports **14 distinct effect types** on each segment (`segment["extra_material_refs"]`, with `enable_lut = True` and `enable_adjust = True`). Every effect MUST carry its exact shader `version` string:

| `type` | Mandatory `version` | `value` Range | Visual Parameter Controlled |
| :--- | :---: | :---: | :--- |
| `"lut"` | `""` | `0.0 .. 1.0` | 3D `.cube` LUT intensity (`path = "<CUBE_PATH>;<EFFECT_CACHE>"`, `enable_skin_tone_correction = true`). |
| `"brightness"` | `"v2"` | `-1.0 .. 1.0` | Overall exposure / midtone luma shift (`brightness_v2`). |
| `"contrast"` | `"v3"` | `-1.0 .. 1.0` | Sigmoid S-curve contrast around pivot `0.435` (`contrast_v3`). |
| `"highlight"` | `"v3"` | `-1.0 .. 1.0` | Highlight shoulder compression (`<0` pulls down hot `P95/P99`; `>0` boosts highlights; `shadow_highlight_v3_gles3`). |
| `"shadow"` | `"v3"` | `-1.0 .. 1.0` | Shadow toe control (`<0` anchors milky `P5/P25` blacks; `>0` lifts crushed shadows; `shadow_highlight_v3_gles3`). |
| `"light_sensation"` | `""` | `-1.0 .. 1.0` | Brilliance / midtone micro-illumination (`light_sensation`). |
| `"saturation"` | `"v1"` | `-1.0 .. 1.0` | Global chroma multiplier (`saturation_v1`). |
| `"temperature"` | `"v3"` | `-1.0 .. 1.0` | Planckian locus Blue (`<0`) ↔ Amber (`>0`) color temperature (`temperature_tone_v3`). |
| `"tone"` | `"v3"` | `-1.0 .. 1.0` | Green (`<0`) ↔ Magenta/Rosy (`>0`) tint axis (`temperature_tone_v3`). |
| `"fade"` | `""` | `0.0 .. 1.0` | Matte black-point fade (`fade`). |
| `"sharpen"` | `"v1"` | `0.0 .. 1.0` | Edge acutance / micro-detail (`sharpen_v1`). |
| `"clear"` | `""` | `0.0 .. 1.0` | Local midtone clarity / micro-contrast (`clear`). |
| `"particle"` | `"v2"` | `0.0 .. 1.0` | Organic 35mm film grain (`particle_v2`). |
| `"vignetting"` | `"v1"` | `-1.0 .. 1.0` | Optical lens vignette fall-off (`>0` darkens corners; `vignetting_v1`). |

---

## 3. CapCut macOS File Synchronization Topology

1. **Always Quit CapCut Before Writing**:
   - Run `osascript -e 'tell application "CapCut" to quit'` before writing JSON files.
2. **Sync All 6 Root & Timeline Files Identically**:
   - `draft_info.json`, `draft_info.json.bak`, `template-2.tmp`
   - `Timelines/<TIMELINE_UUID>/draft_info.json`, `draft_info.json.bak`, `template-2.tmp` *(never touch `template.tmp`)*.
3. **Sync Compound Clip (`subdraft`)**:
   - Copy `root_draft["materials"]["drafts"][0]["draft"]` into the matching `subdraft/<SUBDRAFT_UUID>/draft_content.json` (`sub_d["materials"]["drafts"][0]["draft"]`).
4. **Reset Patch Queue**:
   - Write `{"patch_data":[]}` to `Timelines/<TIMELINE_UUID>/attachment/patch/patch.json`.
