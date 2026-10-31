---
name: capcut-ai-colorist
description: >-
  AI Colorist guide and empirical color science reference for grading AI-generated
  films in CapCut. Enforces a mandatory pre-grading user intake (asking model source
  and emotional direction), uses ground-truth BANGTRA/script scene mapping + per-segment
  source_timerange measurements, and counter-balances camera-angle color jumps within
  scenes using Sibling LUTs + 12 Adjust sliders to achieve a seamless Seedance 2.5 35mm look.
  Pair with capcut-draft-tech for technical execution.
---

# AI Colorist Guide: Scene Continuity, Angle-Jump Counter-Balancing & Seedance 2.5 35mm Emulation

This skill defines **how the AI Colorist thinks, interacts with the user, maps scenes from ground-truth scripts (`BANGTRA.txt` / `sf-board.json`), handles camera-angle color jumps within a scene, and tunes LUT + 12 Adjust parameters per timeline segment**. For technical JSON execution, use **`capcut-draft-tech`**.

---

## 1. MANDATORY STEP 0: Pre-Grading User Intake & AI Visual Inspection (STOP, VIEW FRAMES & ASK FIRST)

> **CRITICAL RULE — NEVER GRADE BLINDLY & NEVER RELY ON CODE NUMBERS ALONE**:
> Even within the same AI model (Seedance 2.5, Google Omni 1.1, or Grok), every video and every cut segment (`source_timerange`) varies by prompt, camera angle, and generation seed.
> Furthermore, **the AI MUST visually inspect the actual video frames with its own eyes (`read_file` / `view_file`) both BEFORE and AFTER grading** — never trust metrics alone!

Follow this exact 3-step workflow:
1. **Run Read-Only Segment-Level Analysis & Visually Inspect `/tmp/<project>_raw_scenes.jpg` First**:
    - Run `capcut-draft-tech/scripts/analyze_project.py` on the requested project.
    - It searches the video folder for ground-truth script files (`*BANGTRA*.txt`, `*PROMPT*.txt`, `sf-board.json`) to map every clip to its exact location (`BG1a/b`, `BG2a/b`, etc.), measures **every cut segment (`Seg[idx]` at its exact `source_timerange`)**, AND renders the **Raw Scene Contact Sheet (`/tmp/<project>_raw_scenes.jpg`)**.
    - **MANDATORY AI VISUAL CHECK #1 (Before Grading)**: Call `read_file` / `view_file` on `/tmp/<project>_raw_scenes.jpg` to **visually see the raw colors, skin tones, lighting, and atmosphere of each scene with your own eyes**!
2. **Ask the User Before Grading** (use `ask_question` or concise bullet questions):
    - **Question 1 — Nguồn tạo video (Model Source & Outliers)**:
      - Ask: *"Project này được tạo bằng model nào (**Seedance 2.5**, **Google Omni 1.1**, **Grok**, hay **Hỗn hợp nhiều model**)? Nếu là hỗn hợp, bạn có nhớ các clip nào thuộc model khác không?"*
      - Share your visual observations from `/tmp/<project>_raw_scenes.jpg`, report whether a ground-truth `*BANGTRA*.txt` / `sf-board.json` was found, and list any segments with strong intra-scene angle jumps.
    - **Question 2 — Định hướng Cảm xúc & Mạch phim (Emotional Direction)**:
      - Ask the user what emotional tone they want for the film:
        - **Ấm áp – Nhân văn – Chữa lành (Warm Human Drama & Golden Reunion)**
        - **Kịch tính – Gai góc – Tương phản sâu (Dramatic High-Contrast Cinema)**
        - **Sang trọng – Điện ảnh Hiện đại (Luxury & Crisp 35mm)**
        - **Chia theo Mạch Cảm Xúc Kịch Bản (Dynamic Narrative Arc)**: Tự động chuyển tone mượt mà theo từng hồi của kịch bản (Mở đầu -> Xung đột -> Cao trào -> Kết đoàn viên).
3. **Apply Grade & Perform MANDATORY AI VISUAL QA #2 on `/tmp/<project>_graded_comparison.jpg`**:
    - After receiving the user's answers, run `apply_grade.py`. It automatically renders the side-by-side **`RAW vs. GRADED` Visual Comparison Sheet (`/tmp/<project>_graded_comparison.jpg`)** with the actual `.cube` 3D LUT + Skin Protection + 12 Adjust sliders applied to each scene's real pixels.
    - **MANDATORY AI VISUAL CHECK #2 (After Grading)**: Call `read_file` / `view_file` on `/tmp/<project>_graded_comparison.jpg` to **visually verify every scene's graded result with your own eyes**! If any scene/shot looks visually off (unnatural skin, crushed shadows, or scene-to-scene color mismatch), override that scene/segment via `--plan` and re-run `apply_grade.py` until the visual comparison sheet looks 100% cinema-grade!

---

## 2. SOLVING THE TWO BIGGEST CONTINUITY TRAPS IN AI FILMS

### Trap 1: Why Code-Only Scene Detection Can Fail -> Always Prefer Ground-Truth `BANGTRA.txt` / `sf-board.json`
- **The Problem**: Pure heuristic code that guesses scene cuts from brightness/color jumps **will make mistakes**:
  1. When a single scene cuts from a bright wide shot to a dark close-up on a black suit (`Mid` drops from `0.27` to `0.08`), naive code falsely splits it into a new scene.
  2. When two different locations happen to have similar average brightness (`Mid ≈ 0.26`), naive code falsely merges them into one scene.
- **The Solution (3-Tier Scene Mapping Hierarchy)**:
  1. **Tier 1 (100% Ground-Truth)**: Parse `*BANGTRA*.txt`, `*PROMPT*.txt`, or `sf-board.json` in the project's video directory. Map each clip number (`001`, `002`, `V-001`) to its **Base Location Code** — crucially merging reverse camera angles of the same room (`BG1a` + `BG1b` = Location `BG1`; `BG2a` + `BG2b` = Location `BG2`) within the same time block into **one continuous Scene Cluster**.
  2. **Tier 2 (Per-Segment `source_timerange` Measurement)**: A 30s Seedance clip (`2.mp4`) is often cut into 8–10 segments (`Seg[1..9]`) on the CapCut timeline, where each segment is a different shot (`1.5s..5.0s`, `5.0s..6.4s`, `12.0s..14.5s`). **Always measure each `Seg[idx]` at its exact `source_timerange`**, never just once per `.mp4` file!
  3. **Tier 3 (Fallback Heuristic + User Confirmation)**: Only use moving-window visual clustering when no `*BANGTRA*.txt` / `sf-board.json` exists on disk, and verify with the user during Step 0.

---

### Trap 2: What Happens When Camera Angles Jump Color Inside the Same Scene (`BG1a` ↔ `BG1b` or Wide ↔ Close-up)?
- **The Problem**:
  - Suppose Scene 1 is in the Grand Lobby (`BG1a` looking toward the desk, `BG1b` looking toward the glass doors, or Shot 1 vs Shot 5 inside `2.mp4`).
  - To the viewer, it is **one continuous room**, so the final graded color must look 100% unified.
  - However, in the raw AI video, Angle A (`Seg[1]`) measures `Mid = 0.27, R-B = +0.094` (cooler glass doors), while Angle B (`Seg[5]`) jumps to `Mid = 0.28, R-B = +0.198` (heavy orange chandelier), and Angle C (`Seg[3]`) drops to `Mid = 0.083` (dark close-up)!
  - **Crucial CapCut Shader Pipeline Fact (`config.json`)**:
    - In CapCut, **`AmazingFilter_lut` runs FIRST (`zorder: 2000`)** on the raw video frame, and **`AmazingFeature_adjustColor` (`temperature`, `tone`, `brightness`, `shadow`, `highlight`, `contrast`, `saturation`) runs SECOND (`zorder: 8004`)** after the LUT!
    - Therefore, if you blindly lock a warm LUT (like `LUT 3 Luxury` or `LUT 1 Life`) onto Angle B (`R-B = +0.198`), the LUT at `zorder 2000` will **bake and clip the orange channel first**, and pulling `temperature` down afterwards at `zorder 8004` cannot un-bake the clipped orange!
- **The Solution: Lock the SCENE TARGET (`Mid_target, RB_target, GM_target`), and Use Counter-Balance Sibling LUTs + Post-LUT Adjust When an Angle Jumps!**
  - Within a continuous Scene Cluster $S$ (with Scene Anchor $\bar{M}_S$ and Scene Target $T_S$):
    1. **Standard Shots in Scene $S$** (`|RB - Anchor_RB| <= 0.040` and `|Mid - Anchor_Mid| <= 0.060`):
       - All receive the **Scene Primary LUT** (e.g. `LUT 3 Luxury - rec709.cube`) + fine Adjust sliders (`brightness`, `temperature`, `tone`, `shadow`, `highlight`) to converge onto $T_S$.
    2. **Angle-Jump Shots Inside Scene $S$** (when a reverse angle `BG1b` or close-up `Seg[idx]` jumps strongly away from the Scene Anchor):
       - **Case A — Angle Jumps Too Warm / Heavy Amber (`RB - Anchor_RB > +0.040` and `RB >= 0.160`)**:
         - Switch that segment's LUT at `zorder 2000` to the **Amber-Cooling Sibling LUT** `CINEMATIC  (27).cube` (`shdRB = -0.054, midRB = -0.023`) + apply negative `temperature` at `zorder 8004` so its final color matches the Scene Target $T_S$!
       - **Case B — Angle Jumps Too Dark / Shadow Close-up (`Mid - Anchor_Mid < -0.060` and `Mid < 0.21`)**:
         - Switch that segment's LUT at `zorder 2000` to the **Shadow-Lifting Sibling LUT** `DRAMATIC (14).cube` (`mid = 0.570`, if warm `RB >= 0.12`) or `FILM LOOK (19).cube` (`mid = 0.600`, if cool `RB < 0.12`) + positive `shadow` and `brightness` at `zorder 8004` so the dark angle doesn't crush to black!
       - **Case C — Angle Jumps Too Cool / Washed-Out (`RB - Anchor_RB < -0.040` or `HSV_S < Anchor_HSV_S - 0.08`)**:
         - Switch that segment's LUT at `zorder 2000` to the **Warmth/Density-Restoring Sibling LUT** `SCC-rec709 (23).cube` (`satG = 1.40, midRB = +0.053`) or `LUT 1 Life - rec709.cube` (`midRB = +0.082`) + positive `temperature`/`saturation` at `zorder 8004`!
       - **Case D — Angle Jumps Too Bright / Milky (`Mid - Anchor_Mid > +0.065` or `P5 > Anchor_P5 + 0.045`)**:
         - Switch that segment's LUT at `zorder 2000` to the **Toe-Anchoring Sibling LUT** `CINEMATIC  (3).cube` (`blk = 0.045, mid = 0.442`) + negative `brightness`/`shadow`/`highlight` at `zorder 8004`!

> **Result**: Because the LUT (`zorder 2000`) and Adjust (`zorder 8004`) work together to pull every angle's raw deviation back to the **same Scene Target $T_S$**, the viewer sees **zero color jumping** when the camera cuts between `BG1a`, `BG1b`, wide shots, and close-ups!

---

## 3. Mandatory Per-Child-Video (`Từng Video Con`) 12-Parameter Adjust Engine (`zorder: 8004`)

**CRITICAL RULE**: Never grade only the outer Compound Clip wrapper or leave child videos with empty/sparse sliders! Every single child video segment (`từng video con` inside `materials.drafts[*]["draft"]["tracks"][*]["segments"]` AND any direct video segment on `root_draft`) MUST receive its own LUT (`enable_lut: true`) **PLUS** a complete, individualized **11–12 parameter Adjust stack** (`enable_adjust: true`) computed from that child video's exact `source_timerange` metrics (`m`) relative to the Scene Target (`scene_cfg`):

Target Seedance 2.5 35mm Film Standard: `P5 ≈ 0.048`, `Mid ≈ 0.25–0.34`, `P95 ≈ 0.635`, `G-M ≈ -0.015` (rosy Kodak skin), `HSV_S ≈ 0.48–0.55`, `Sharpness ≈ 0.046`, `HF_Noise ≈ 0.034`.

For every child video segment `Seg[idx]` (measured over its actual `source_timerange`), tune all 12 CapCut Adjust parameters:
1. **`shadow` (`v3`)**: Continuous 35mm black-point calibration toward `P5_target = 0.048` on every child video (`< 0` pulls down milky blacks; `> 0` opens crushed darks).
2. **`highlight` (`v3`)**: Continuous 35mm shoulder roll-off toward `P95_target = 0.635` on every child video.
3. **`brightness` (`v2`) & `light_sensation` (`""`)**: Pull each child video's `Mid` directly toward the Scene Target `Mid_target` + local neighbor smoothing (`nb_mid_diff`), eliminating shot-to-shot brightness jumps.
4. **`contrast` (`v3`)**: Continuous 35mm S-curve contrast toward `Contrast_target = 0.565` + black-point density compensation on every child video.
5. **`tone` (`v3`, Green ↔ Magenta Tint)**: Eliminate digital green casts (`GM > -0.006`) and lock every child video in the scene to Seedance 2.5's rosy Kodak skin tint (`GM_target ≈ -0.015`).
6. **`temperature` (`v3`, Blue ↔ Amber)**: Pull every child video's `RB` toward the Scene Target `RB_target` + neighbor smoothing so all camera angles in the room converge to the exact same warmth.
7. **`saturation` (`v1`)**: Continuous chroma density lock to the Scene Target `Sat_target` + HSV compensation on every child video.
8. **`sharpen` (`v1`), `clear` (`""`), `particle` (`v2`), `vignetting` (`v1`)**: Guaranteed per-child-video 35mm optical acutance (`sharpen`, `clear`) and Kodak film grain (`particle`) inversely calibrated to each child video's `Sharpness` and `HF_Noise`.

---

## 4. Video-First Pre-Flight `Video × 1,636 LUT` Simulation & CapCut `tt_skin_seg` Skin Protection

**NEVER hard-lock or reject LUTs blindly without checking the video's actual color first!** Instead, `apply_grade.py` evaluates all **1,636 LUTs** in `lut_optical_db.json` by **checking the video's measured color state first (`diagnose_video_state`) and mathematically simulating `Video × LUT` (`simulate_video_through_lut`)**:

1. **Step 1 — Diagnose Actual Video Optical State First (`diagnose_video_state(m)`)**:
   - Inspects `(P5, Mid, P95, Contrast, RB, GM, Sat)` of the scene/segment before touching any LUT:
     - `FLAT_LOG_OR_FOGGY` (`Contrast < 0.40, P5 > 0.105, P95 < 0.64`): Video itself is flat/washed-out like Log or heavy fog -> **automatically unlocks steeper S-curve & Log-recovery LUTs** (`camera_log` / `stylized_rec709`) that would clip normal footage!
     - `HIGH_KEY_MILKY` (`P5 > 0.115, Mid > 0.38`): Bright milky curve -> unlocks toe-anchoring LUTs.
     - `LOW_KEY_DARK` (`Mid < 0.215`): Dark shadow curve -> unlocks shadow-lifting LUTs without crushing blacks.
     - `HARSH_CONTRAST` (`Contrast > 0.66, P5 < 0.040`): Already steep contrast -> only allows gentle/low-contrast LUTs so shadows/highlights never clip.
     - `HEAVY_WARM_CAST` (`RB >= 0.165`) / `COLD_CYAN_CAST` (`RB <= 0.080`): Strong color cast -> selects counter-balancing LUTs (`midRB < 0` or `midRB > 0`).
2. **Step 2 — Simulate `Video × LUT` Output with CapCut `tt_skin_seg` (`simulate_video_through_lut`)**:
   - For every candidate LUT in the 1,636 database and candidate intensity `alpha ∈ {0.18 .. 0.32}`, computes the exact post-LUT video metrics `(sim_p5, sim_mid, sim_hi, sim_contrast, sim_rb, sim_gm, sim_sat)` while modeling CapCut's neural skin mask `tt_skin_seg` (`"enable_skin_tone_correction": true` at `zorder: 2000`).
   - **Dynamic Post-LUT Quality Gate**: Rejects any `(LUT, alpha)` pair *only if its simulated output on THIS specific video* crushes blacks (`sim_p5 < 0.014`), clips highlights (`sim_hi > 0.85`), inverts tonal order (`sim_p5 >= sim_mid` or `sim_mid >= sim_hi`), or turns a color video monochrome (`sim_sat < 0.085` unless input is monochrome or `--allow-bw` is set).
   - Picks the `(LUT, alpha)` pair whose simulated output lands closest to the 35mm Scene Target `(target_p5=0.048, target_mid, target_hi=0.635, target_rb, target_gm, target_sat)`.

---

## 5. Reference Baseline: Typical Model Tendencies (Context Only — Always Measure)

| Characteristic | **Seedance 2.5** (Target Reference) | **Google Omni 1.1** (Typical Tendency) | **Grok Video** (Typical Tendency) |
| :--- | :--- | :--- | :--- |
| **Black Floor (`P5`)** | Deep anchored (`0.040 – 0.079`) | Lifted, milky grey (`0.110 – 0.180`), even at night | Variable; often hard toe (`0.035`) or bright midtones (`0.29`) |
| **Midtones (`Mid`)** | Low-key cinema (`0.24 – 0.33`) | High-key bright (`0.41 – 0.53`), night looks like twilight | Fluctuates clip-to-clip (`0.19 – 0.30`) |
| **Highlights (`P95`)** | Soft film shoulder (`0.55 – 0.68`) | Hot, near-clipping (`0.75 – 0.89`) | Crisp/steep S-curve (`0.50 – 0.66`) |
| **Color & Tint (`R-B / G-M`)** | Warm amber (`+0.15`), rosy Kodak tint (`G-M = -0.016`) | Cooler (`+0.10`), slight green digital tint (`G-M = +0.003`) | Often overly orange (`R-B > +0.18`) or overly cold (`+0.00`) |
| **Texture (`Sharpness`)** | Crisp pore/fabric detail (`0.040 – 0.050`) | ~35% softer/smoother (`0.026 – 0.034`), low grain | High digital edge contrast (`0.040 – 0.047`), synthetic grain |
