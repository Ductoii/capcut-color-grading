#!/usr/bin/env python3
"""Scene-Continuous + Angle-Jump Counter-Balanced 14-Parameter CapCut Grading Engine:
- Uses Ground-Truth BANGTRA/sf-board.json scene clusters (merging reverse angles BG1a/BG1b -> BG1).
- Measures every timeline segment at its exact source_timerange (catching internal angle cuts inside multi-shot clips).
- Locks the Scene Target (Mid, R-B, G-M, Sat) across the scene: normal shots use the Scene Primary LUT, while shots that jump strongly in angle/lighting automatically switch to a Counter-Balance Sibling LUT (at zorder 2000) + Post-LUT Adjust (at zorder 8004) so all shots exit with the exact same final color.
"""
import argparse
import copy
import glob
import json
import os
import shutil
import subprocess
import uuid
from collections import defaultdict
import numpy as np

from analyze_project import (
    cluster_timeline_scenes,
    collect_all_video_segments,
    discover_ground_truth_scene_map,
    extract_segment_metrics,
    resolve_project_dir,
)

LUT_CFG_PATH = "/Users/daho100/Movies/CapCut/User Data/Resources/Lut/LutImportCfg.json"
DEFAULT_EFFECT_CACHE = "/Users/daho100/Library/Containers/com.lemon.lvoverseas/Data/Movies/CapCut/User Data/Cache/effect/7501974767453474064/26e574bcfa176c3c1a1f7fd5df0a4996"
OPTICAL_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "lut_optical_db.json")


def load_optical_db(lut_map: dict) -> list[dict]:
    if not os.path.exists(OPTICAL_DB_PATH):
        return []
    with open(OPTICAL_DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)
    return [entry for entry in db if entry.get("title") in lut_map]

VERSION_MAP = {
    "lut": "",
    "brightness": "v2",
    "contrast": "v3",
    "highlight": "v3",
    "shadow": "v3",
    "light_sensation": "",
    "saturation": "v1",
    "temperature": "v3",
    "tone": "v3",
    "fade": "",
    "sharpen": "v1",
    "clear": "",
    "particle": "v2",
    "vignetting": "v1",
}


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def non_zero_clamp(val: float, lo: float, hi: float, min_abs: float = 0.01, default_sign: float = 1.0) -> float:
    clamped = clamp(val, lo, hi)
    if abs(clamped) < min_abs:
        sign = 1.0 if clamped > 0 else (-1.0 if clamped < 0 else default_sign)
        clamped = sign * min_abs
    return round(clamped, 3)


def load_lut_map() -> dict:
    lut_dir = os.path.dirname(LUT_CFG_PATH)
    os.makedirs(lut_dir, exist_ok=True)
    cfg = []
    if os.path.exists(LUT_CFG_PATH):
        with open(LUT_CFG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    lut_map = {item["title"]: item["path"] for item in cfg if os.path.exists(item.get("path", ""))}

    # Auto-install bundled core LUTs if missing on this machine
    bundled_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "luts")
    if os.path.isdir(bundled_dir):
        changed = False
        for fn in sorted(os.listdir(bundled_dir)):
            if not fn.lower().endswith((".cube", ".3dl")):
                continue
            if fn not in lut_map:
                src = os.path.join(bundled_dir, fn)
                dst = os.path.join(lut_dir, fn)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                cfg = [it for it in cfg if it.get("title") != fn]
                cfg.append({"path": dst, "title": fn})
                lut_map[fn] = dst
                changed = True
        if changed:
            with open(LUT_CFG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
    return lut_map


def make_effect(eff_type: str, value: float, lut_map: dict, effect_cache: str, lut_title: str = "", version: str = None, skin_corr: bool = False) -> dict:
    if eff_type == "lut":
        assert lut_title in lut_map, f"LUT not found in LutImportCfg.json: {lut_title}"
        path = f"{lut_map[lut_title]};{effect_cache}"
    else:
        path = effect_cache
    ver = VERSION_MAP.get(eff_type, "") if version is None else version
    return {
        "id": str(uuid.uuid4()).upper(),
        "effect_id": "7501974767453474064",
        "resource_id": "7501974767453474064",
        "third_resource_id": "",
        "name": lut_title if eff_type == "lut" else "",
        "report_name": "",
        "type": eff_type,
        "sub_type": "none",
        "path": path,
        "value": round(float(value), 4),
        "visible": True,
        "item_effect_type": 0,
        "category_id": "",
        "category_name": "",
        "category_key": "",
        "sub_category_id": "",
        "sub_category_name": "",
        "platform": "all",
        "apply_target_type": 0,
        "source_platform": 1,
        "version": ver,
        "adjust_params": [],
        "time_range": None,
        "formula_id": "",
        "enable_skin_tone_correction": skin_corr,
        "algorithm_artifact_path": "",
        "intensity_key": "",
        "face_adjust_params": [],
        "exclusion_group": [],
        "panel_id": "",
        "bloom_params": None,
        "request_id": "",
        "color_match_info": {
            "target_feature_path": "",
            "source_feature_path": "",
            "target_image_path": ""
        },
        "multi_language_current": "",
        "lumi_hub_path": effect_cache + "/lumi_hub_path",
        "covering_relation_change": 0,
        "beauty_face_auto_preset_id": "",
        "beauty_body_auto_preset_id": "",
        "beauty_face_auto_retouch_info": {
            "face_id": [],
            "beauty_face_auto_retouch_id": ""
        },
        "smart_color_mode": 0,
        "is_from_intelligent_quality": False
    }


def diagnose_video_state(m: dict) -> str:
    """
    Inspect the actual measured video metrics BEFORE choosing any LUT:
    identifies whether the footage is flat/log-like, high-key milky, dark low-key,
    steep-contrast, heavy warm/amber, cold/cyan, or balanced Rec.709.
    """
    p5 = m.get("p5", 0.05)
    mid = m.get("mid", 0.28)
    p95 = m.get("p95", 0.62)
    c = m.get("contrast", p95 - p5)
    rb = m.get("rb", 0.13)
    sat = m.get("sat", 0.15)
    if sat < 0.035:
        return "MONOCHROME_INPUT"
    if c < 0.40 and p5 > 0.105 and p95 < 0.64:
        return "FLAT_LOG_OR_FOGGY"
    if p5 > 0.115 and mid > 0.38:
        return "HIGH_KEY_MILKY"
    if mid < 0.215 or (p5 < 0.028 and mid < 0.24):
        return "LOW_KEY_DARK"
    if c > 0.66 and p5 < 0.040:
        return "HARSH_CONTRAST"
    if rb >= 0.165:
        return "HEAVY_WARM_CAST"
    if rb <= 0.080:
        return "COLD_CYAN_CAST"
    return "BALANCED_REC709"


def simulate_video_through_lut(video_m: dict, lut_entry: dict, alpha: float) -> dict:
    """
    Mathematically simulate applying `lut_entry` (at intensity `alpha` with CapCut's
    `enable_skin_tone_correction: true` / `tt_skin_seg` neural skin mask active)
    onto the actual measured video `video_m`.
    """
    p5 = max(0.005, video_m.get("p5", 0.05))
    mid = max(0.02, video_m.get("mid", 0.28))
    p95 = max(0.10, video_m.get("p95", 0.62))
    rb = video_m.get("rb", 0.13)
    gm = video_m.get("gm", -0.012)
    sat = video_m.get("sat", 0.15)

    # Non-linear 3D LUT transfer ratios anchored at reference points (0.05, 0.28, 0.75)
    lut_blk_mapped = clamp(lut_entry["blk"] * (p5 / 0.05), 0.0, 1.0)
    lut_mid_mapped = clamp(lut_entry["mid"] * (mid / 0.28), 0.0, 1.0)
    lut_hi_mapped = clamp(lut_entry["hi"] * (p95 / 0.75), 0.0, 1.0)

    sim_p5 = (1.0 - alpha) * p5 + alpha * lut_blk_mapped
    sim_mid = (1.0 - alpha) * mid + alpha * lut_mid_mapped
    sim_hi = (1.0 - alpha) * p95 + alpha * lut_hi_mapped

    # Environment RB/GM shift + CapCut `tt_skin_seg` skin protection (~72% skin hue preservation)
    sim_rb = rb + alpha * lut_entry["midRB"]
    sim_gm = gm + alpha * (0.28 * (lut_entry["sknGM"] - (-0.025)) + 0.72 * lut_entry["midGM"])
    sim_sat = sat * ((1.0 - alpha) + alpha * lut_entry["satGain"])

    return {
        "sim_p5": sim_p5,
        "sim_mid": sim_mid,
        "sim_hi": sim_hi,
        "sim_contrast": sim_hi - sim_p5,
        "sim_rb": sim_rb,
        "sim_gm": sim_gm,
        "sim_sat": sim_sat,
    }


def match_optimal_lut_by_video_simulation(
    optical_db: list[dict],
    video_m: dict,
    target_mid: float,
    target_rb: float,
    target_gm: float,
    target_sat: float,
    preferred_families: tuple[str, ...],
    fallback_lut: str,
    allow_bw: bool = False,
) -> tuple[str, float, str]:
    """
    Instead of blindly hard-locking LUT tiers without checking the video:
    1. Diagnoses the actual video's color & dynamic range state (`video_m`).
    2. Simulates `Video × LUT` across all 1,636 LUTs in `lut_optical_db.json` over candidate intensities `alpha`.
    3. Rejects any (LUT, alpha) pair that causes clipping, shadow crushing, tonal inversion, or unwanted B&W on THIS specific video.
    4. Selects the (LUT, alpha) pair whose simulated post-LUT output lands closest to the 35mm Scene Target.
    """
    v_state = diagnose_video_state(video_m)
    if not optical_db:
        return fallback_lut, 0.28, v_state

    target_p5 = 0.048
    target_hi = 0.635
    is_mono_video = (v_state == "MONOCHROME_INPUT") or allow_bw

    best_title = fallback_lut
    best_alpha = 0.28
    best_score = 1e9

    candidate_alphas = (0.20, 0.24, 0.28, 0.32) if v_state in ("FLAT_LOG_OR_FOGGY", "HIGH_KEY_MILKY") else (0.18, 0.23, 0.27, 0.30)

    for entry in optical_db:
        # Only skip monochrome LUTs when the video is color and B&W wasn't requested
        if not is_mono_video and (entry.get("tier") == "bw_mono" or entry.get("satGain", 1.0) < 0.28):
            continue
        # Only allow camera_log LUTs if the video itself measured as FLAT_LOG_OR_FOGGY or HIGH_KEY_MILKY
        if entry.get("tier") == "camera_log" and v_state not in ("FLAT_LOG_OR_FOGGY", "HIGH_KEY_MILKY"):
            continue

        t = entry["title"]
        t_up = t.upper()
        fam_bonus = -0.028 if any(k in t_up for k in preferred_families) else 0.0

        for alpha in candidate_alphas:
            sim = simulate_video_through_lut(video_m, entry, alpha)
            # Dynamic Post-LUT Optical Safety Gates on THIS specific video:
            if not (sim["sim_p5"] < sim["sim_mid"] < sim["sim_hi"]):
                continue
            if sim["sim_p5"] < 0.014 or sim["sim_p5"] > 0.105:
                continue
            if sim["sim_hi"] > 0.85 or sim["sim_hi"] < 0.45:
                continue
            if not is_mono_video and (sim["sim_sat"] < 0.085 or sim["sim_sat"] > 0.235):
                continue

            # Distance between simulated Post-LUT video output and 35mm Scene Target
            err_mid = abs(sim["sim_mid"] - target_mid) * 2.6
            err_rb = abs(sim["sim_rb"] - target_rb) * 3.2
            err_p5 = abs(sim["sim_p5"] - target_p5) * 2.4
            err_hi = abs(sim["sim_hi"] - target_hi) * 1.4
            err_gm = abs(sim["sim_gm"] - target_gm) * 1.8
            err_sat = abs(sim["sim_sat"] - target_sat) * 1.6

            score = err_mid + err_rb + err_p5 + err_hi + err_gm + err_sat + fam_bonus
            if score < best_score:
                best_score = score
                best_title = t
                best_alpha = alpha

    return best_title, best_alpha, v_state


def select_scene_primary_lut_and_targets(
    anchor: dict,
    emotion: str,
    progress_ratio: float,
    optical_db: list[dict] = None,
    allow_bw: bool = False,
) -> dict:
    p5 = anchor.get("p5", 0.05)
    mid = anchor.get("mid", 0.28)
    p95 = anchor.get("p95", 0.62)
    rb = anchor.get("rb", 0.13)
    sat = anchor.get("sat", 0.15)
    hsv_s = anchor.get("hsv_s", 0.45)

    if emotion == "narrative-arc":
        if progress_ratio < 0.15:
            act_mood = "luxury-crisp"
        elif progress_ratio < 0.62:
            act_mood = "dramatic-contrast"
        elif progress_ratio < 0.85:
            act_mood = "climax-punch"
        else:
            act_mood = "warm-human"
    else:
        act_mood = emotion

    if rb < 0.085 and mid > 0.35:
        target_mid = 0.28 if act_mood == "dramatic-contrast" else 0.31
    elif mid > 0.38:
        target_mid = 0.34 if p95 > 0.78 else 0.31
    elif mid < 0.22:
        target_mid = 0.25
    else:
        target_mid = clamp(mid, 0.24, 0.35)

    if rb > 0.165 and act_mood != "vintage-35mm":
        target_rb = 0.145
    elif rb < 0.095 and act_mood in ("warm-human", "vintage-35mm"):
        target_rb = 0.120
    else:
        target_rb = clamp(rb, 0.085, 0.155)

    target_gm = -0.016 if act_mood in ("warm-human", "vintage-35mm") else -0.013
    target_sat = clamp(sat, 0.140, 0.185)
    use_vignette = (act_mood in ("dramatic-contrast", "climax-punch") and target_mid < 0.32)

    if act_mood == "warm-human":
        fb_lut, fams = "Mood 9 Skin rec709.cube", ("MOOD", "SKINTONE", "LUT ", "WARM", "THE ROMANCE")
    elif act_mood == "climax-punch":
        fb_lut, fams = "SCC-rec709 (23).cube", ("SCC-REC709", "FILM LOOK", "HOLLYWOOD", "DRAMATIC")
    elif act_mood == "dramatic-contrast":
        fb_lut, fams = "FILM LOOK (4).cube", ("FILM LOOK", "CINEMATIC", "DRAMATIC", "ATMOSPHERIC")
    elif act_mood == "vintage-35mm":
        fb_lut, fams = "LUT 1 Life - rec709.cube", ("VINTAGE", "FILM LOOK", "LUT ", "MOOD")
    else:
        fb_lut, fams = "LUT 3 Luxury - rec709.cube", ("SKINTONE", "LUT ", "COMMERCIAL", "CINEMATIC")

    lut, base_val, v_state = match_optimal_lut_by_video_simulation(
        optical_db, anchor, target_mid, target_rb, target_gm, target_sat, fams, fb_lut, allow_bw=allow_bw
    )
    prof = f"[{act_mood} | {v_state}] Simulated Video×LUT Match"

    return {
        "act_mood": act_mood,
        "video_state": v_state,
        "lut": lut,
        "base_val": base_val,
        "profile": prof,
        "target_mid": target_mid,
        "target_rb": target_rb,
        "target_gm": target_gm,
        "target_sat": target_sat,
        "use_vignette": use_vignette,
    }


def compute_shot_lut_and_adjustments(
    m: dict,
    scene_cfg: dict,
    scene_anchor: dict,
    local_neighbor_m: dict,
    force_neighbor_match: bool = False,
    optical_db: list[dict] = None,
) -> tuple[str, float, list, str]:
    """
    Solves intra-scene camera-angle color jumps!
    Because CapCut executes LUT FIRST (zorder 2000) and Adjust SECOND (zorder 8004):
    - Normal shots in the scene use `scene_cfg["lut"]`.
    - When a camera angle inside the scene jumps strongly in raw color/exposure (`dRB` or `dMid`),
      it searches the 508 Cinema-Safe Rec.709 LUT database for the Counter-Balance Sibling LUT at zorder 2000
      + Post-LUT Adjust at zorder 8004 so its final rendered pixels converge to the exact same Scene Target.
    """
    p5 = m.get("p5", 0.05)
    p25 = m.get("p25", 0.15)
    mid = m.get("mid", 0.28)
    p95 = m.get("p95", 0.62)
    p99 = m.get("p99", 0.76)
    c = m.get("contrast", 0.55)
    rb = m.get("rb", 0.13)
    gm = m.get("gm", -0.012)
    sat = m.get("sat", 0.15)
    hsv_s = m.get("hsv_s", 0.45)
    sharp = m.get("sharp", 0.042)
    hf_noise = m.get("hf_noise", 0.030)

    target_mid = scene_cfg["target_mid"]
    target_rb = scene_cfg["target_rb"]
    target_gm = scene_cfg["target_gm"]
    target_sat = scene_cfg["target_sat"]
    act_mood = scene_cfg["act_mood"]

    d_rb_scene = rb - scene_anchor.get("rb", rb)
    d_mid_scene = mid - scene_anchor.get("mid", mid)

    lut_title = scene_cfg["lut"]
    base_val = scene_cfg["base_val"]
    prof_note = scene_cfg["profile"]

    if abs(d_mid_scene) > 0.058 or abs(d_rb_scene) > 0.040:
        fb = "DRAMATIC (14).cube" if d_mid_scene < -0.058 else ("CINEMATIC  (27).cube" if d_rb_scene > 0.040 else "SCC-rec709 (23).cube")
        lut_title, base_val, shot_state = match_optimal_lut_by_video_simulation(
            optical_db, m, target_mid, target_rb, target_gm, target_sat,
            ("CINEMATIC", "DRAMATIC", "FILM LOOK", "SCC-REC709", "SKINTONE", "MOOD"), fb,
        )
        prof_note = f"{scene_cfg['profile']} [Angle-Jump Simulated: {shot_state}]"

    nb_mid_diff = mid - local_neighbor_m.get("mid", mid)
    nb_rb_diff = rb - local_neighbor_m.get("rb", rb)
    nb_sat_diff = sat - local_neighbor_m.get("sat", sat)
    nb_weight = 0.65 if force_neighbor_match else 0.35

    adjust = []

    # GUARANTEED PER-CHILD-VIDEO (`video con`) ADJUST STACK:
    # Every child video segment receives its own complete, tailored Adjust sliders
    # calibrating its exact source_timerange metrics (`m`) to the Scene Target (`scene_cfg`).

    # 1. SHADOW (`shadow`, v3) — Continuous 35mm black-point calibration toward target_p5 = 0.048
    if p5 > 0.048:
        raw_shd = -(p5 - 0.048) * 1.15
    else:
        raw_shd = (0.048 - p5) * 1.25 + max(0.0, 0.135 - p25) * 0.35
    shadow_val = non_zero_clamp(raw_shd, -0.15, 0.14, min_abs=0.01, default_sign=-1.0)
    adjust.append(("shadow", shadow_val))

    # 2. HIGHLIGHT (`highlight`, v3) — Continuous 35mm highlight roll-off toward target_p95 = 0.635
    raw_hl = -(p95 - 0.635) * 0.62 - max(0.0, p99 - 0.820) * 0.35
    highlight_val = non_zero_clamp(raw_hl, -0.15, 0.08, min_abs=0.01, default_sign=-1.0)
    adjust.append(("highlight", highlight_val))

    # 3. BRIGHTNESS (`brightness`, v2) & LIGHT SENSATION (`light_sensation`, "") — Per-shot exposure lock to Scene Target
    raw_br = -(mid - target_mid) * 0.60 - nb_mid_diff * nb_weight
    bright_val = non_zero_clamp(raw_br, -0.10, 0.09, min_abs=0.01, default_sign=1.0 if target_mid >= mid else -1.0)
    adjust.append(("brightness", bright_val))

    raw_ls = -(mid - target_mid) * 0.25 - nb_mid_diff * 0.15
    light_sens_val = non_zero_clamp(raw_ls, -0.05, 0.04, min_abs=0.01, default_sign=1.0 if target_mid >= mid else -1.0)
    adjust.append(("light_sensation", light_sens_val))

    # 4. CONTRAST (`contrast`, v3) — Continuous 35mm S-curve contrast toward target_c = 0.565 + toe density
    raw_con = (0.565 - c) * 0.45 + (p5 - 0.050) * 0.45 - (0.025 if force_neighbor_match else 0.0)
    contrast_val = non_zero_clamp(raw_con, -0.08, 0.09, min_abs=0.01, default_sign=1.0)
    adjust.append(("contrast", contrast_val))

    # 5. TONE / TINT (`tone`, v3) — Continuous green/magenta skintone lock to Scene Target (`target_gm`)
    raw_tone = (gm - target_gm) * 2.8
    tone_val = non_zero_clamp(raw_tone, -0.06, 0.09, min_abs=0.01, default_sign=1.0)
    adjust.append(("tone", tone_val))

    # 6. TEMPERATURE (`temperature`, v3) — Continuous warmth lock to Scene Target (`target_rb`) + neighbor smoothing
    raw_temp = -(rb - target_rb) * 1.15 - nb_rb_diff * nb_weight
    temp_val = non_zero_clamp(raw_temp, -0.13, 0.11, min_abs=0.01, default_sign=1.0 if target_rb >= rb else -1.0)
    adjust.append(("temperature", temp_val))

    # 7. SATURATION (`saturation`, v1) — Continuous chroma density lock to Scene Target (`target_sat`)
    raw_sat = -(sat - target_sat) * 0.90 - nb_sat_diff * nb_weight + (0.45 - hsv_s) * 0.18
    sat_val = non_zero_clamp(raw_sat, -0.12, 0.09, min_abs=0.01, default_sign=1.0 if target_sat >= sat else -1.0)
    adjust.append(("saturation", sat_val))

    # 8. MICRO-TEXTURE (`sharpen`, `clear`, `particle`, `vignetting`) — Guaranteed per-child-video 35mm film texture
    sh_val = round(clamp(0.045 + max(0.0, 0.046 - sharp) * 5.8, 0.03, 0.15), 3)
    cl_val = round(clamp(0.020 + max(0.0, 0.044 - sharp) * 1.8, 0.015, 0.05), 3)
    adjust.append(("sharpen", sh_val))
    adjust.append(("clear", cl_val))

    pt_val = round(clamp(0.035 + max(0.0, 0.034 - hf_noise) * 3.2 + (0.015 if act_mood == "vintage-35mm" else 0.0), 0.025, 0.08), 3)
    adjust.append(("particle", pt_val))

    if scene_cfg["use_vignette"]:
        adjust.append(("vignetting", 0.045))

    lut_val = round(base_val + clamp((scene_anchor["mid"] - mid) * 0.08, -0.010, 0.010), 3)
    return lut_title, lut_val, adjust, prof_note


def main():
    parser = argparse.ArgumentParser(description="Apply scene-continuous & per-child-video 35mm grading to a CapCut project.")
    parser.add_argument("project", help="CapCut Project ID or path.")
    parser.add_argument("--metrics", default="", help="Optional precomputed diagnostics JSON path from analyze_project.py.")
    parser.add_argument(
        "--emotion",
        choices=["narrative-arc", "warm-human", "dramatic-contrast", "luxury-crisp", "vintage-35mm"],
        default="narrative-arc",
        help="Emotional grading direction chosen by the user.",
    )
    parser.add_argument("--plan", default="", help="Optional custom AI grading plan JSON path.")
    parser.add_argument("--match-clips", default="", help="Optional comma-separated clips to strongly match to immediate neighbors.")
    parser.add_argument("--auto-match-outliers", action="store_true", help="Automatically boost neighbor-matching on intra-scene outlier segments.")
    parser.add_argument("--allow-stylized", action="store_true", help="Unlock Tier 2 (stylized_rec709, 382 LUTs) with CapCut tt_skin_seg Skin Protection.")
    parser.add_argument("--allow-bw", action="store_true", help="Unlock Tier 3 (bw_mono, 45 LUTs) for flashback/noir sequences.")
    parser.add_argument("--input-color-space", choices=["rec709", "log"], default="rec709", help="Input color space (rec709 for all AI video; log only for real camera Log).")
    parser.add_argument("--relaunch", action="store_true", help="Quit CapCut before writing and relaunch after completion.")
    args = parser.parse_args()

    proj_dir = resolve_project_dir(args.project)
    lut_map = load_lut_map()

    if args.relaunch:
        subprocess.run(["osascript", "-e", 'tell application "CapCut" to quit'], check=False)

    backup_dir = os.path.join("/tmp", f"capcut_backup_{os.path.basename(proj_dir).replace(' ', '_')}")
    os.makedirs(backup_dir, exist_ok=True)

    root_path = os.path.join(proj_dir, "draft_info.json")
    shutil.copy2(root_path, os.path.join(backup_dir, "draft_info.json"))

    with open(root_path, "r", encoding="utf-8") as f:
        root_draft = json.load(f)

    sub_drafts = root_draft.get("materials", {}).get("drafts", [])
    has_compound = bool(sub_drafts)
    all_seg_tuples = collect_all_video_segments(root_draft)

    # Load or compute per-segment source_timerange diagnostics + ground-truth BG map
    seg_records = []
    clip_bg_map = {}
    if args.metrics and os.path.exists(args.metrics):
        loaded = json.load(open(args.metrics, "r", encoding="utf-8"))
        if isinstance(loaded, dict) and "segments" in loaded and len(loaded["segments"]) == len(all_seg_tuples):
            seg_records = loaded["segments"]
            clip_bg_map = {int(k): v for k, v in loaded.get("clip_bg_map", {}).items()}

    if not seg_records or len(seg_records) != len(all_seg_tuples):
        video_paths = list({v.get("path", "") for _, _, v, _ in all_seg_tuples if v.get("path")})
        clip_bg_map, _ = discover_ground_truth_scene_map(video_paths)
        cache_by_range = {}
        seg_records = []
        for idx, (_, seg, v, container_lbl) in enumerate(all_seg_tuples):
            vpath = v.get("path", "")
            mname = v.get("material_name") or os.path.basename(vpath)
            st_sec = seg.get("source_timerange", {}).get("start", 0) / 1e6
            dur_sec = seg.get("source_timerange", {}).get("duration", 5000000) / 1e6
            ck = (vpath, round(st_sec, 1), round(dur_sec, 1))
            if ck not in cache_by_range and os.path.exists(vpath):
                cache_by_range[ck] = extract_segment_metrics(vpath, st_sec, dur_sec)
            seg_records.append({
                "seg_idx": idx,
                "seg_id": seg.get("id", ""),
                "container": container_lbl,
                "material_name": mname,
                "start_sec": round(st_sec, 2),
                "dur_sec": round(dur_sec, 2),
                "metrics": cache_by_range.get(ck, {}),
            })

    custom_plan = json.load(open(args.plan, "r", encoding="utf-8")) if (args.plan and os.path.exists(args.plan)) else {}
    all_vals = [r["metrics"] for r in seg_records if r.get("metrics")]
    med_p5 = float(np.median([x.get("p5", 0.05) for x in all_vals])) if all_vals else 0.05
    med_shp = float(np.median([x.get("sharp", 0.042) for x in all_vals])) if all_vals else 0.042

    # 1. Clean old managed effects across root_draft and all sub_drafts
    managed_types = set(VERSION_MAP.keys())
    draft_containers = [root_draft] + [sd["draft"] for sd in sub_drafts if "draft" in sd]
    old_eff_ids = set()
    for d_obj in draft_containers:
        effs = d_obj.get("materials", {}).setdefault("effects", [])
        old_eff_ids.update(e["id"] for e in effs if e.get("type") in managed_types)
        effs[:] = [e for e in effs if e.get("type") not in managed_types]

    # Configure Master Compound Clip wrapper segments on root_draft
    if has_compound:
        root_effects = root_draft["materials"].setdefault("effects", [])
        compound_ids = {sd["id"] for sd in sub_drafts if "id" in sd}
        if "master" in custom_plan:
            m_cfg = custom_plan["master"]
            m_lut_title, m_lut_val = m_cfg.get("lut", ["SKINTONE (7).cube", 0.14])
            m_adjs = m_cfg.get("adjust", [("sharpen", 0.12), ("clear", 0.03), ("particle", 0.055)])
        else:
            m_lut_title = "CINEMATIC  (3).cube" if med_p5 >= 0.095 else "SKINTONE (7).cube"
            m_lut_val = 0.14
            m_sh = round(clamp(0.12 + max(0.0, 0.040 - med_shp) * 5.0, 0.10, 0.18), 3)
            m_cl = 0.04 if med_shp < 0.036 else 0.03
            m_pt = 0.065 if med_shp < 0.036 else 0.055
            m_adjs = [("sharpen", m_sh), ("clear", m_cl), ("particle", m_pt)]

        for trk in root_draft.get("tracks", []):
            for seg in trk.get("segments", []):
                seg_refs = [seg.get("material_id", "")] + list(seg.get("extra_material_refs", []))
                if any(r in compound_ids for r in seg_refs):
                    seg["extra_material_refs"] = [r for r in seg.get("extra_material_refs", []) if r not in old_eff_ids]
                    for adj_t, adj_v in m_adjs:
                        ae = make_effect(adj_t, adj_v, lut_map, DEFAULT_EFFECT_CACHE)
                        root_effects.append(ae)
                        seg["extra_material_refs"].append(ae["id"])
                    mle = make_effect("lut", m_lut_val, lut_map, DEFAULT_EFFECT_CACHE, lut_title=m_lut_title, skin_corr=True)
                    root_effects.append(mle)
                    seg["extra_material_refs"].append(mle["id"])
                    seg["enable_lut"] = True
                    seg["enable_adjust"] = True

    # 2. Ground-Truth Scene Clustering + Per-Child-Video (`video con`) LUT & Full Adjust Stack
    optical_db = load_optical_db(lut_map)
    clusters = cluster_timeline_scenes(seg_records, clip_bg_map)
    seg_to_cluster = {}
    total_segs = max(len(all_seg_tuples), 1)
    for c in clusters:
        mid_prog = 0.5 * (c["start_seg"] + c["end_seg"]) / total_segs
        c["scene_cfg"] = select_scene_primary_lut_and_targets(
            c["anchor"], args.emotion, mid_prog, optical_db=optical_db, allow_bw=args.allow_bw
        )
        for s_idx in c["seg_indices"]:
            seg_to_cluster[s_idx] = c

    explicit_match = {x.strip() for x in args.match_clips.split(",") if x.strip()}
    plan_segs = custom_plan.get("segments", {})
    default_m = {"p5": 0.05, "p25": 0.15, "mid": 0.28, "p95": 0.62, "p99": 0.76, "contrast": 0.55, "rb": 0.13, "gm": -0.012, "sat": 0.15, "hsv_s": 0.45, "sharp": 0.042, "hf_noise": 0.030}

    summary = []
    total_adjusts = 0
    angle_jump_fixes = 0
    for idx, (d_obj, seg, _, _) in enumerate(all_seg_tuples):
        effects = d_obj["materials"].setdefault("effects", [])
        seg["extra_material_refs"] = [r for r in seg.get("extra_material_refs", []) if r not in old_eff_ids]
        rec = seg_records[idx]
        mname = rec["material_name"]
        m = rec.get("metrics") or default_m
        c_info = seg_to_cluster[idx]
        scene_anchor = c_info["anchor"]
        scene_cfg = c_info["scene_cfg"]

        if str(idx) in plan_segs or mname in plan_segs:
            scfg = plan_segs.get(str(idx)) or plan_segs.get(mname)
            adj_list = scfg.get("adjust", [])
            lut_title, val = scfg["lut"]
            prof = scfg.get("profile", "Custom Plan")
        else:
            c_indices = c_info["seg_indices"]
            nb_ms = [seg_records[j].get("metrics") or default_m for j in c_indices if abs(j - idx) <= 2 and j != idx]
            if not nb_ms:
                nb_ms = [scene_anchor]
            local_nb_m = {k: float(np.mean([x.get(k, m[k]) for x in nb_ms])) for k in default_m}

            force_nb = (mname in explicit_match) or (
                args.auto_match_outliers
                and (abs(m["rb"] - scene_anchor["rb"]) > 0.045 or abs(m["mid"] - scene_anchor["mid"]) > 0.060)
            )
            lut_title, val, adj_list, prof = compute_shot_lut_and_adjustments(m, scene_cfg, scene_anchor, local_nb_m, force_nb, optical_db=optical_db)
            if "Angle-Jump" in prof:
                angle_jump_fixes += 1

        for adj_type, adj_val in adj_list:
            if abs(adj_val) >= 0.005:
                ae = make_effect(adj_type, adj_val, lut_map, DEFAULT_EFFECT_CACHE)
                effects.append(ae)
                seg["extra_material_refs"].append(ae["id"])
                total_adjusts += 1

        le = make_effect("lut", val, lut_map, DEFAULT_EFFECT_CACHE, lut_title=lut_title, skin_corr=True)
        effects.append(le)
        seg["extra_material_refs"].append(le["id"])
        seg["enable_lut"] = True
        seg["enable_adjust"] = True
        summary.append((idx, c_info["cluster_id"], mname, lut_title, val, prof, len(adj_list)))

    # 3. Write synced files (Root + Timelines + ALL Subdraft Compound Clips)
    root_json_str = json.dumps(root_draft, ensure_ascii=False, separators=(",", ":"))
    sync_targets = ["draft_info.json", "draft_info.json.bak", "template-2.tmp"]
    for tl_dir in glob.glob(os.path.join(proj_dir, "Timelines", "*")):
        if os.path.isdir(tl_dir):
            for fname in ("draft_info.json", "draft_info.json.bak", "template-2.tmp"):
                rel = os.path.relpath(os.path.join(tl_dir, fname), proj_dir)
                sync_targets.append(rel)
            patch_file = os.path.join(tl_dir, "attachment", "patch", "patch.json")
            if os.path.exists(patch_file):
                with open(patch_file, "w", encoding="utf-8") as pf:
                    pf.write('{"patch_data":[]}\n')

    for rel in sync_targets:
        full_p = os.path.join(proj_dir, rel)
        if os.path.exists(os.path.dirname(full_p)):
            with open(full_p, "w", encoding="utf-8") as f:
                f.write(root_json_str)

    if has_compound:
        draft_by_id = {sd["id"]: sd["draft"] for sd in sub_drafts if "id" in sd and "draft" in sd}
        for sub_file in glob.glob(os.path.join(proj_dir, "subdraft", "*", "draft_content.json")):
            with open(sub_file, "r", encoding="utf-8") as sf:
                sub_d = json.load(sf)
            changed_sub = False
            for sd_entry in sub_d.get("materials", {}).get("drafts", []):
                sid = sd_entry.get("id")
                if sid in draft_by_id:
                    shutil.copy2(sub_file, os.path.join(backup_dir, f"subdraft_{sid}_content.json"))
                    sd_entry["draft"] = copy.deepcopy(draft_by_id[sid])
                    changed_sub = True
            if changed_sub:
                with open(sub_file, "w", encoding="utf-8") as sf:
                    json.dump(sub_d, sf, ensure_ascii=False, separators=(",", ":"))

    mode_str = clusters[0]["source_mode"] if clusters else "NONE"
    print(
        f"Graded {len(summary)} child video segments (video con) across {len(clusters)} Scene Clusters ({mode_str}) in '{os.path.basename(proj_dir)}' "
        f"[Emotion: {args.emotion} | {angle_jump_fixes} Angle-Jump Sibling LUT fixes | {total_adjusts} per-video Adjust sliders ({total_adjusts/max(len(summary),1):.1f} sliders/video)]"
    )
    for c in clusters:
        cfg = c["scene_cfg"]
        print(
            f"  - Scene[{c['cluster_id']:2d}] ({c['bg_code']:8s}) Segs {c['start_seg']:3d}..{c['end_seg']:3d} ({len(c['seg_indices']):2d} segs) | "
            f"Primary LUT: {cfg['lut']:26s} ({cfg['base_val']*100:.0f}%)"
        )

    if args.relaunch:
        subprocess.run(["open", "-a", "/Applications/CapCut.app"], check=False)


if __name__ == "__main__":
    main()
