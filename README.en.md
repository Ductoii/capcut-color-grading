# CapCut AI Colorist & Draft Tech

**A two-skill 35mm film color grading suite for CapCut macOS — featuring per-segment (`source_timerange`) empirical video diagnostics, ground-truth script/BANGTRA scene mapping, intra-scene camera-angle jump counter-balancing, and a full 12-parameter Adjust stack on EVERY child video segment (`video con`).**

[Tiếng Việt](README.md) · [AI Colorist Guide](skills/capcut-ai-colorist/SKILL.md) · [Draft Tech Engine](skills/capcut-draft-tech/SKILL.md) · [MIT](LICENSE)

## Architecture (2 Separated Skills)

1. **[`capcut-ai-colorist`](skills/capcut-ai-colorist/SKILL.md)** *(AI Colorist Strategy & Decision Guide)*:
   - **Mandatory Pre-Grading User Intake**: Measures actual timeline segments first, then asks the user to confirm (1) AI model source (`Seedance 2.5`, `Google Omni 1.1`, `Grok`, or mixed) and (2) Emotional grading direction (`narrative-arc`, `warm-human`, `dramatic-contrast`, `luxury-crisp`, `vintage-35mm`).
   - **Ground-Truth Scene Clustering**: Auto-discovers `*BANGTRA*.txt`, `*PROMPT*.txt`, or `sf-board.json` to map exact scene locations and merges reverse camera angles (`BG1a` + `BG1b` → `BG1`).
   - **Angle-Adaptive Sibling LUT Counter-Balancing**: Respects CapCut's shader order (`AmazingFilter_lut` at `zorder: 2000` runs before `AmazingFeature_adjustColor` at `zorder: 8004`) so camera angles that jump in exposure/color temperature switch to a counter-balance sibling LUT and converge to the exact same Scene Target.
   - **Full Per-Child-Video (`video con`) Adjust Stack**: Every child video inside every Compound Clip (`subdraft`) and on the root timeline receives its own LUT + 11–12 tailored Adjust sliders.
2. **[`capcut-draft-tech`](skills/capcut-draft-tech/SKILL.md)** *(Technical Execution & CapCut JSON Sync Engine)*:
   - Bundles the 12 core 35mm `.cube` LUTs in `skills/capcut-draft-tech/resources/luts/` and auto-registers them in `LutImportCfg.json`.
   - Extracts 16 `ffmpeg` + `numpy` metrics per timeline segment (`source_timerange`).
   - Safely syncs `draft_info.json`, `draft_info.json.bak`, `template-2.tmp`, `Timelines/*`, and `subdraft/*/draft_content.json`.

## Install

```sh
git clone https://github.com/Ductoii/capcut-color-grading.git
cd capcut-color-grading
python3 install.py
```
