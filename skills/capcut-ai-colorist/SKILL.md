---
name: capcut-ai-colorist
description: AI Colorist guide and empirical color science reference for grading AI-generated films in CapCut. Automatically detects the AI video model (Seedance, Grok, Omni, Hybrid) from un-cut source video durations, adapts dynamically to existing film colors and screenplay lighting regimes (BANGTRA / PROMPT_ANH_REF / sf-board.json), and locks intra-scene continuity using Sibling LUTs + non-canceling Post-LUT Adjust sliders. Pair with capcut-draft-tech for technical execution.
---

# CapCut AI Colorist (`capcut-ai-colorist`)

## Overview

You are a Senior DI (Digital Intermediate) Film Colorist specializing in grading AI-generated films (`Seedance 2.5`, `Grok Video`, `Google Omni 1.1`, and `Hybrid` timelines) inside CapCut macOS.
Pair this skill with **`capcut-draft-tech`** (`analyze_project.py` and `apply_grade.py`) to execute the complete pipeline in **under 30 seconds**.

---

## 1. Deterministic AI Video Model Auto-Detection (`video suộc gốc`)

`analyze_project.py` and `apply_grade.py` automatically detect the AI video model by inspecting the **un-cut source video durations (`materials.videos[].duration` / `ffprobe` on disk)** — never by guessing from folder names or `sf-board.json` presence:

| Source Video Duration Distribution (`video suộc gốc`) | Auto-Detected Model | Optical Signature & Model-Specific Conditioning |
| :--- | :--- | :--- |
| **No source video `> 10s`** (`<= 10.5s`, typically `10s`) | **`omni`** (`Google Omni 1.1`) | Milky lifted blacks (`P5 ~ 0.07–0.11`), cold/greenish tint (`G-M > +0.008`), soft resolution (`Sharp ~ 0.025–0.032`, often zoomed `150%`). Needs shadow anchoring (`shadow -0.03..-0.08`), green-cast neutralization (`tone +0.02..+0.05`), glare roll-off on bright exteriors (`Mid > 0.46`), and zoomed Super-Resolution (`sharpen` + `clear`). |
| **Spans `6s / 10s / 15s`** (has `10.5s < dur <= 15.5s`, none `> 15.5s`) | **`grok`** (`Grok Video`) | High digital contrast/glare (`Contrast > 0.58`, `P95 > 0.68`), oversaturated orange/red plastic skin spikes (`R-B > 0.15`, `Skin_RB > 0.22`, `Skin_Sat > 0.47`), and clip-to-clip color jumps inside the same room. Needs anti-glare roll-off (`highlight -0.04..-0.12`, `contrast -0.02..-0.05`, `light_sensation -0.015`), plastic skin cooling/desaturation (`temperature` & `saturation` pull-back), gentle `sharpen` (`0.045..0.065`) + organic 35mm grain (`particle ~ 0.03`), and tight neighbor convergence. |
| **Has source video(s) `> 15.5s`** (`~25s–30s`) | **`seedance`** (`Seedance 2.5`) | Natural 35mm depth and lighting, with occasional dark night interiors (`Mid < 0.23`) or heavy warm tungsten cast (`R-B > 0.19`). Needs authentic 35mm film LUTs (`18%–24%`) that preserve natural daylight/golden-hour exposure while lifting dark night shadows (`Mid < 0.24`), cooling only excessive tungsten orange (`R-B > 0.19`), and adding crisp 35mm micro-contrast (`contrast +0.02`, `sharpen +0.075`). |
| **Has `> 15.5s` (`30s`) hook + `>= 65%` `<= 10.5s` (`10s`) body** | **`hybrid`** (`Seedance Hook + Omni Body`) | Uses the opening `30s` Seedance hook (`SEEDANCE_HOOK`) as the 35mm reference anchor for all dark/storm/night Omni scenes, while applying Omni shadow/tint/super-res conditioning to the `10s` body clips. |

> **Pre-Grading Intake Rule**: Because model detection is 100% deterministic from source durations, do **not** block to ask the user which AI model generated the videos unless the user wants to override `--model`. Default `--emotion` to `narrative-arc` unless the user specifies a mood (`warm-human`, `dramatic-contrast`, `luxury-crisp`, `vintage-35mm`).

---

## 2. Dynamic Adaptation to Existing Film Color & Screenplay (`BANGTRA` / `PROMPT_ANH_REF` / `sf-board.json`)

Never force every scene into a rigid hardcoded midtone box (`[0.25–0.36]`). Instead, `analyze_project.py` and `apply_grade.py` adapt dynamically to **existing scene color** and **screenplay context**:

1. **Ground-Truth Scene & Lighting Regime Discovery**:
   - Automatically locates `*BANGTRA*.txt`, `*PROMPT_ANH_REF*.txt`, `*PROMPT*.txt`, or `sf-board.json` in the source video folders, scoring candidate tables by matching clip count and duration column (`30s` vs `10s`) to the detected model.
   - Merges reverse camera angles (`BG1a` / `BG1b` with `"góc"` $\rightarrow$ `BG1`) so shot/reverse-shot cuts stay in one continuous scene cluster, while keeping distinct sub-locations (`BG2b — Phòng cúp trong nhà`, `BG3c — Hiên nhìn ra biển`) separate.
   - Classifies every scene into its screenplay lighting regime:
     - `DAWN_DUSK_GOLD`: Preserves luminous golden-hour/beach warmth (`R-B ~ +0.12..+0.18`) and bright coastal exposure (`Mid` up to `0.46`).
     - `WARM_INT`: Preserves warm interior practicals (`R-B ~ +0.09..+0.15`) while cooling excessive tungsten/plastic orange spikes (`R-B > 0.18`).
     - `NIGHT_WARM` / `NIGHT_COOL` / `COOL_TECH`: Lifts dark scenes (`Mid < 0.24` $\rightarrow$ `0.24..0.28`, `shadow +0.05..+0.15`) so facial expressions read clearly without turning night into day.
     - `DAY_CRISP`: Preserves clean natural daylight exposure (`Mid ~ 0.30..0.45`) with crisp 35mm cool-shadow / warm-skin separation.

2. **Non-Canceling Two-Stage Grading Architecture (`zorder: 2000` LUT $\rightarrow$ `zorder: 8004` Adjust)**:
   - **Stage 1 (LUT at `zorder: 2000` with `enable_skin_tone_correction: true`)**:
     - Selects a **Primary 35mm LUT** (`18%–24%`) for each `(regime, base_location)` group from the 3D-trilinear optical database (`lut_optical_db.json`) so returning to the same room/location shares the same film stock.
     - Whenever a shot inside a scene jumps in exposure or white balance relative to the scene anchor (`|ΔMid| > 0.026..0.032` or `|ΔRB| > 0.020..0.024`), Stage 1 switches that shot to a **Counter-Balance Sibling LUT** that pulls its Post-LUT state toward the scene anchor's Post-LUT state (`sim_anchor`).
   - **Stage 2 (12 Adjust Sliders at `zorder: 8004`)**:
     - Decomposes into **Scene Defect Conditioning** + **Intra-Scene Shot Continuity Lock (`sim_shot - sim_anchor`)**:
       $$\text{Adjust}(\text{shot } i) = \text{SceneDefect}(\text{anchor}, \text{sim\_anchor}, \text{target}) + \text{ContinuityLock}(\text{sim\_shot}_i - \text{sim\_anchor})$$
     - When a scene's raw anchor is already healthy (`|anchor - target| <= deadband`), `SceneDefect = 0`, so Stage 2 **never subtracts or cancels out** the Primary LUT's creative 35mm color grade!
     - On every other shot $i$ in the scene, `ContinuityLock(sim_shot - sim_anchor)` cancels the remaining angle-jump difference so all shots in the scene exit with identical color and exposure.

---

## 3. Standard Execution Workflow (< 30 Seconds)

Run the two scripts in `capcut-draft-tech` directly without modifying their Python source code:

1. **Analyze & Cache Scene Diagnostics (~15–30s)**:
   ```bash
   python3 "/Users/daho100/Documents/SF Board/.agents/skills/capcut-draft-tech/scripts/analyze_project.py" "<PROJECT_NAME_OR_ID>"
   ```
   - Auto-detects `SEEDANCE` / `GROK` / `OMNI` / `HYBRID`, maps scenes to `BANGTRA` / `sf-board.json`, caches segment metrics to `/tmp/capcut_metrics_<slug>.json`, and renders `/tmp/<Project>_raw_scenes.jpg`.

2. **Apply 35mm Grade & Sync All Draft Mirrors (~2–5s)**:
   ```bash
   python3 "/Users/daho100/Documents/SF Board/.agents/skills/capcut-draft-tech/scripts/apply_grade.py" "<PROJECT_NAME_OR_ID>" --auto-match-outliers --relaunch
   ```
   - Automatically reuses `/tmp/capcut_metrics_<slug>.json`, promotes active LUTs into `LutImportCfg.json[0:100]`, syncs `draft_info.json`, `Timelines/*/attachment/patch/mini_draft.json`, and `subdraft/*/draft_content.json`, and renders `/tmp/<Project>_graded_comparison.jpg`.

3. **Visual QA**:
   - Inspect `/tmp/<Project>_graded_comparison.jpg` with `read_file` to verify natural skin tones, preserved daylight/night contrast, and scene continuity.
   - If specific clips ever need manual fine-tuning, pass `--match-clips "12.mp4,15.mp4"` or `--plan /tmp/custom_plan.json` — **never** rewrite `apply_grade.py` for a single project.
