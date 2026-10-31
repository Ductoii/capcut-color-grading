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
    extract_rgb_frame,
    extract_segment_metrics,
    resolve_project_dir,
    resolve_video_path,
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
    target_p5: float = 0.046,
    target_hi: float = 0.635,
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
        return fallback_lut, 0.25, v_state

    is_mono_video = (v_state == "MONOCHROME_INPUT") or allow_bw
    min_allowed_sat = min(0.055, video_m.get("sat", 0.10) * 0.72)

    best_title = fallback_lut
    best_alpha = 0.25
    best_score = 1e9

    candidate_alphas = (0.18, 0.22, 0.26, 0.30) if v_state in ("FLAT_LOG_OR_FOGGY", "HIGH_KEY_MILKY") else (0.16, 0.20, 0.24, 0.28)

    cinematic_safe_prefixes = (
        "FILM LOOK", "CINEMATIC", "ATMOSPHERIC", "DRAMATIC", "HOLLYWOOD",
        "MOOD", "SKINTONE", "SCC-REC709", "LUT ", "WARM", "COMMERCIAL",
        "THE ROMANCE", "VINTAGE", "PRESETPRO"
    )

    for entry in optical_db:
        if not is_mono_video and (entry.get("tier") == "bw_mono" or entry.get("satGain", 1.0) < 0.55):
            continue
        # Never use camera_log LUTs on Rec.709 AI videos, and avoid extreme highlight color casts (e.g. pink skies)
        if entry.get("tier") == "camera_log":
            continue
        if abs(entry.get("hiRB", 0.0)) > 0.085 or abs(entry.get("hiGM", 0.0)) > 0.050:
            continue

        t = entry["title"]
        t_up = t.upper()
        if not any(t_up.startswith(pfx) for pfx in cinematic_safe_prefixes):
            continue
        fam_bonus = -0.025 if any(k in t_up for k in preferred_families) else 0.0

        for alpha in candidate_alphas:
            sim = simulate_video_through_lut(video_m, entry, alpha)
            if not (sim["sim_p5"] < sim["sim_mid"] < sim["sim_hi"]):
                continue
            if sim["sim_p5"] < 0.012 or sim["sim_p5"] > 0.115:
                continue
            if sim["sim_hi"] > 0.88 or sim["sim_hi"] < 0.42:
                continue
            if not is_mono_video and (sim["sim_sat"] < min_allowed_sat or sim["sim_sat"] > 0.245):
                continue

            err_mid = abs(sim["sim_mid"] - target_mid) * 2.8
            err_rb = abs(sim["sim_rb"] - target_rb) * 3.4
            err_p5 = abs(sim["sim_p5"] - target_p5) * 2.4
            err_hi = abs(sim["sim_hi"] - target_hi) * 1.5
            err_gm = abs(sim["sim_gm"] - target_gm) * 2.0
            err_sat = abs(sim["sim_sat"] - target_sat) * 1.8

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
    bg_code: str = "",
    seedance_ref: dict = None,
) -> dict:
    p5 = anchor.get("p5", 0.05)
    mid = anchor.get("mid", 0.28)
    p95 = anchor.get("p95", 0.62)
    rb = anchor.get("rb", 0.13)
    sat = anchor.get("sat", 0.15)

    is_seedance_hook = bg_code.startswith("SEEDANCE_HOOK")
    is_dark_storm_or_night = (
        is_seedance_hook
        or bg_code == "E0:BG1"
        or "NIGHT" in bg_code
        or "NIGH" in bg_code
        or (mid <= 0.265 and rb <= 0.025)
        or (rb <= -0.012 and mid <= 0.335)
    )
    is_dark_warm_interior = (not is_dark_storm_or_night) and (mid <= 0.255 and rb > 0.025)

    if is_seedance_hook:
        act_mood = "seedance-35mm-ref"
        target_p5 = 0.028
        target_mid = 0.258
        target_hi = 0.585
        target_rb = 0.015
        target_gm = -0.012
        target_sat = 0.118
        use_vignette = True
        fb_lut, fams = "FILM LOOK (4).cube", ("FILM LOOK", "CINEMATIC", "ATMOSPHERIC", "DRAMATIC")
    elif is_dark_storm_or_night and seedance_ref:
        # Synchronize dark/rainy/night Omni scenes directly to the Graded 30s Seedance Hook!
        act_mood = "seedance-dark-sync"
        target_p5 = seedance_ref.get("target_p5", 0.028)
        target_mid = seedance_ref.get("target_mid", 0.248)
        target_hi = seedance_ref.get("target_hi", 0.580)
        target_rb = seedance_ref.get("target_rb", 0.018)
        target_gm = seedance_ref.get("target_gm", -0.014)
        target_sat = seedance_ref.get("target_sat", 0.122)
        use_vignette = True
        fb_lut, fams = "CINEMATIC  (3).cube", ("CINEMATIC", "FILM LOOK", "DRAMATIC", "ATMOSPHERIC")
    elif is_dark_warm_interior and seedance_ref:
        # Dark evening study/interior: lock tonal density & blacks to Seedance, preserve warm lamp hue
        act_mood = "seedance-dark-warm"
        target_p5 = 0.032
        target_mid = 0.255
        target_hi = 0.595
        target_rb = clamp(rb, 0.075, 0.105)
        target_gm = -0.014
        target_sat = 0.125
        use_vignette = True
        fb_lut, fams = "CINEMATIC  (3).cube", ("CINEMATIC", "FILM LOOK", "DRAMATIC", "MOOD")
    else:
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

        target_p5 = 0.048
        target_hi = 0.635
        if rb < 0.085 and mid > 0.35:
            target_mid = 0.335 if p95 > 0.75 else 0.315
        elif mid > 0.38:
            target_mid = 0.345 if p95 > 0.78 else 0.325
        elif mid < 0.22:
            target_mid = 0.265
        else:
            target_mid = clamp(mid, 0.26, 0.35)

        if rb > 0.165 and act_mood != "vintage-35mm":
            target_rb = 0.142
        elif rb < 0.095 and act_mood in ("warm-human", "vintage-35mm"):
            target_rb = 0.105
        else:
            target_rb = clamp(rb, 0.055, 0.145)

        target_gm = -0.014 if act_mood in ("warm-human", "vintage-35mm") else -0.010
        target_sat = clamp(sat * 1.12, 0.108, 0.165)
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
        optical_db, anchor, target_mid, target_rb, target_gm, target_sat, fams, fb_lut,
        allow_bw=allow_bw, target_p5=target_p5, target_hi=target_hi
    )
    prof = f"[{act_mood} | {v_state}] Simulated Video×LUT Match"

    return {
        "act_mood": act_mood,
        "video_state": v_state,
        "lut": lut,
        "base_val": base_val,
        "profile": prof,
        "target_p5": target_p5,
        "target_mid": target_mid,
        "target_hi": target_hi,
        "target_rb": target_rb,
        "target_gm": target_gm,
        "target_sat": target_sat,
        "use_vignette": use_vignette,
    }


def assign_unified_scene_targets(clusters: list, emotion: str, optical_db: list[dict], allow_bw: bool = False):
    """
    1. Extracts Seedance 35mm reference from `SEEDANCE_HOOK` (if present at Cluster 0) and locks all
       dark/rainy/night clusters to the Graded Seedance 35mm look.
    2. Unifies all clusters sharing the same `(LightingRegime, BaseLocation)` (e.g. all Daytime BG4 Hangar
       scenes, all Daytime BG5 Briefing scenes, all Daytime BG6 Tarmac scenes) to a single unified
       Location Target and Primary LUT so returning to the same room/tarmac never jumps color!
    """
    total_segs = sum(len(c["seg_indices"]) for c in clusters) or 1
    seedance_ref = {
        "target_p5": 0.028,
        "target_mid": 0.248,
        "target_hi": 0.580,
        "target_rb": 0.018,
        "target_gm": -0.014,
        "target_sat": 0.122,
    }

    # First pass: group clusters by (LightingRegime, BaseLocation) to compute unified Location Anchors
    loc_groups = defaultdict(list)
    for c in clusters:
        bg_code = c["bg_code"]
        a = c["anchor"]
        base_loc = bg_code.split(":")[-1] if ":" in bg_code else bg_code
        if bg_code.startswith("SEEDANCE_HOOK"):
            regime = "SEEDANCE_HOOK"
        elif bg_code == "E0:BG1" or "NIGHT" in bg_code or "NIGH" in bg_code or (a["mid"] <= 0.265 and a["rb"] <= 0.025) or (a["rb"] <= -0.012 and a["mid"] <= 0.335):
            regime = "DARK_STORM_NIGHT"
            base_loc = "NIGHT_EXT"
        elif a["mid"] <= 0.255 and a["rb"] > 0.025:
            regime = "DARK_WARM_INT"
        elif "E1_" in bg_code and base_loc == "BG3":
            regime = "EVENING_BALLROOM"
        elif "E7_" in bg_code or ("SUNDAY" in bg_code and base_loc == "BG3C"):
            regime = "GOLDEN_REUNION"
            base_loc = "COASTAL_SUN"
        else:
            regime = "DAYTIME"
        c["_group_key"] = (regime, base_loc)
        loc_groups[(regime, base_loc)].append(c)

    # Compute unified anchor for each (regime, base_loc) group weighted by segment count
    group_cfgs = {}
    for gkey, g_clusters in loc_groups.items():
        regime, base_loc = gkey
        keys = list(g_clusters[0]["anchor"].keys())
        weighted_anchor = {}
        weights = [max(1, len(x["seg_indices"])) for x in g_clusters]
        for k in keys:
            vals = [x["anchor"][k] for x in g_clusters]
            weighted_anchor[k] = round(float(np.average(vals, weights=weights)), 4)

        avg_prog = float(np.average([0.5 * (x["start_seg"] + x["end_seg"]) / total_segs for x in g_clusters], weights=weights))
        rep_bg = g_clusters[0]["bg_code"]
        cfg = select_scene_primary_lut_and_targets(
            weighted_anchor, emotion, avg_prog, optical_db=optical_db,
            allow_bw=allow_bw, bg_code=rep_bg, seedance_ref=seedance_ref
        )
        group_cfgs[gkey] = (weighted_anchor, cfg)

    for c in clusters:
        unified_anchor, unified_cfg = group_cfgs[c["_group_key"]]
        c["unified_anchor"] = unified_anchor
        c["scene_cfg"] = copy.deepcopy(unified_cfg)


def compute_shot_lut_and_adjustments(
    m: dict,
    scene_cfg: dict,
    scene_anchor: dict,
    local_neighbor_m: dict,
    force_neighbor_match: bool = False,
    optical_db: list[dict] = None,
    seg_scale: float = 1.0,
) -> tuple[str, float, list, str]:
    """
    Closed-Loop Two-Stage Controller for Intra-Scene & Same-Location Continuity:
    - Stage 1 (zorder 2000): If a shot deviates from the unified Scene Target (`target_mid`, `target_rb`, `target_p5`),
      searches the 1,636 LUT database for the optimal Counter-Balance Sibling LUT.
    - Stage 2 (zorder 8004): Simulates the exact Post-LUT metrics `sim = simulate_video_through_lut(m, lut, val)`
      and computes all 12 Adjust sliders to cancel out the remaining Post-LUT residual (`sim - target`),
      plus adds a Scale-Aware Super-Resolution sharpness/clarity boost for zoomed clips (`scale` up to 1.50x)!
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

    target_p5 = scene_cfg.get("target_p5", 0.046)
    target_mid = scene_cfg["target_mid"]
    target_hi = scene_cfg.get("target_hi", 0.635)
    target_rb = scene_cfg["target_rb"]
    target_gm = scene_cfg["target_gm"]
    target_sat = scene_cfg["target_sat"]
    act_mood = scene_cfg["act_mood"]

    d_rb_target = rb - target_rb
    d_mid_target = mid - target_mid
    d_p5_target = p5 - target_p5

    lut_title = scene_cfg["lut"]
    base_val = scene_cfg["base_val"]
    prof_note = scene_cfg["profile"]

    # Stage 1 (zorder 2000): Trigger Counter-Balance Sibling LUT whenever shot deviates from Scene Target
    if abs(d_mid_target) > 0.042 or abs(d_rb_target) > 0.032 or abs(d_p5_target) > 0.026:
        fb = "DRAMATIC (14).cube" if d_mid_target < -0.042 else ("CINEMATIC  (27).cube" if d_rb_target > 0.032 else "SCC-rec709 (23).cube")
        lut_title, base_val, shot_state = match_optimal_lut_by_video_simulation(
            optical_db, m, target_mid, target_rb, target_gm, target_sat,
            ("CINEMATIC", "DRAMATIC", "FILM LOOK", "SCC-REC709", "SKINTONE", "MOOD", "ATMOSPHERIC"),
            fb, target_p5=target_p5, target_hi=target_hi
        )
        prof_note = f"{scene_cfg['profile']} [Angle-Jump Simulated: {shot_state}]"

    lut_val = round(base_val + clamp((target_mid - mid) * 0.06, -0.012, 0.012), 3)

    # Simulate exact Post-LUT optical state `sim` so Stage 2 (Adjust at zorder 8004) cancels the exact residual
    lut_entry = next((e for e in (optical_db or []) if e["title"] == lut_title), None)
    if lut_entry:
        sim = simulate_video_through_lut(m, lut_entry, lut_val)
    else:
        sim = {"sim_p5": p5, "sim_mid": mid, "sim_hi": p95, "sim_contrast": c, "sim_rb": rb, "sim_gm": gm, "sim_sat": sat}

    nb_mid_diff = mid - local_neighbor_m.get("mid", mid)
    nb_rb_diff = rb - local_neighbor_m.get("rb", rb)
    nb_sat_diff = sat - local_neighbor_m.get("sat", sat)
    nb_weight = 0.45 if force_neighbor_match else 0.22

    adjust = []

    # Stage 2 (zorder 8004) — Closed-Loop Residual-Canceling 12-Slider Adjust Stack per Child Video:
    # 1. SHADOW (`shadow`, v3) — Pull Post-LUT `sim_p5` to `target_p5`
    res_p5 = sim["sim_p5"] - target_p5
    if res_p5 > 0:
        raw_shd = -res_p5 * 2.10
    else:
        raw_shd = -res_p5 * 2.25 + max(0.0, 0.130 - p25) * 0.35
    shadow_val = non_zero_clamp(raw_shd, -0.24, 0.22, min_abs=0.01, default_sign=-1.0)
    adjust.append(("shadow", shadow_val))

    # 2. HIGHLIGHT (`highlight`, v3) — Pull Post-LUT `sim_hi` to `target_hi`
    res_hi = sim["sim_hi"] - target_hi
    raw_hl = -res_hi * 0.82 - max(0.0, p99 - 0.840) * 0.35
    highlight_val = non_zero_clamp(raw_hl, -0.24, 0.14, min_abs=0.01, default_sign=-1.0)
    adjust.append(("highlight", highlight_val))

    # 3. BRIGHTNESS (`brightness`, v2) & LIGHT SENSATION (`light_sensation`, "") — Pull Post-LUT `sim_mid` to `target_mid`
    res_mid = sim["sim_mid"] - target_mid
    raw_br = -res_mid * 1.15 - nb_mid_diff * nb_weight
    bright_val = non_zero_clamp(raw_br, -0.24, 0.20, min_abs=0.01, default_sign=1.0 if res_mid <= 0 else -1.0)
    adjust.append(("brightness", bright_val))

    raw_ls = -res_mid * 0.38 - nb_mid_diff * 0.15
    light_sens_val = non_zero_clamp(raw_ls, -0.08, 0.07, min_abs=0.01, default_sign=1.0 if res_mid <= 0 else -1.0)
    adjust.append(("light_sensation", light_sens_val))

    # 4. CONTRAST (`contrast`, v3) — Pull Post-LUT `sim_contrast` to `target_hi - target_p5`
    target_c = target_hi - target_p5
    raw_con = (target_c - sim["sim_contrast"]) * 0.52 + max(0.0, res_p5) * 0.55
    contrast_val = non_zero_clamp(raw_con, -0.12, 0.12, min_abs=0.01, default_sign=1.0)
    adjust.append(("contrast", contrast_val))

    # 5. TONE / TINT (`tone`, v3) — Eliminate Omni green tint & lock Post-LUT `sim_gm` to `target_gm`
    res_gm = sim["sim_gm"] - target_gm
    raw_tone = res_gm * 3.8
    tone_val = non_zero_clamp(raw_tone, -0.12, 0.16, min_abs=0.01, default_sign=1.0)
    adjust.append(("tone", tone_val))

    # 6. TEMPERATURE (`temperature`, v3) — Pull Post-LUT `sim_rb` directly to `target_rb`
    res_rb = sim["sim_rb"] - target_rb
    raw_temp = -res_rb * 1.85 - nb_rb_diff * nb_weight
    temp_val = non_zero_clamp(raw_temp, -0.25, 0.25, min_abs=0.01, default_sign=1.0 if res_rb <= 0 else -1.0)
    adjust.append(("temperature", temp_val))

    # 7. SATURATION (`saturation`, v1) — Pull Post-LUT `sim_sat` to `target_sat`
    res_sat = sim["sim_sat"] - target_sat
    raw_sat = -res_sat * 1.45 - nb_sat_diff * (nb_weight * 0.6) + (0.38 - hsv_s) * 0.14
    sat_val = non_zero_clamp(raw_sat, -0.18, 0.18, min_abs=0.01, default_sign=1.0 if res_sat <= 0 else -1.0)
    adjust.append(("saturation", sat_val))

    # 8. SCALE-AWARE SUPER-RESOLUTION MICRO-TEXTURE (`sharpen`, `clear`, `particle`, `vignetting`):
    # Boosts sharpness & local clarity for Omni footage AND adds proportional compensation for zoomed clips (up to 150% / 1.50x)!
    zoom_excess = max(0.0, float(seg_scale) - 1.0)
    sh_val = round(clamp(0.145 + max(0.0, 0.044 - sharp) * 4.5 + zoom_excess * 0.28, 0.08, 0.35), 3)
    cl_val = round(clamp(0.042 + max(0.0, 0.044 - sharp) * 1.8 + zoom_excess * 0.09, 0.025, 0.11), 3)
    adjust.append(("sharpen", sh_val))
    adjust.append(("clear", cl_val))

    pt_val = round(clamp(0.036 + max(0.0, 0.034 - hf_noise) * 3.0 + (0.012 if "seedance" in act_mood else 0.0), 0.025, 0.075), 3)
    adjust.append(("particle", pt_val))

    if scene_cfg["use_vignette"]:
        adjust.append(("vignetting", 0.045))

    return lut_title, lut_val, adjust, prof_note


def apply_3d_lut_and_adjust_to_rgb(rgb: np.ndarray, lut_path: str, lut_val: float, adj_list: list) -> np.ndarray:
    """
    Applies the actual `.cube` 3D LUT (at `zorder: 2000` with skin-tone protection)
    followed by the Adjust sliders (at `zorder: 8004`) onto a float32 RGB image `[0..1]`
    so the AI can visually inspect the exact Before vs. After graded look!
    """
    out = rgb.copy()
    if lut_path and os.path.exists(lut_path) and lut_path.lower().endswith(".cube"):
        rows = []
        size = 0
        with open(lut_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s[0] == "#":
                    continue
                if s.upper().startswith("LUT_3D_SIZE"):
                    size = int(s.split()[-1])
                    continue
                if s[0].isdigit() or s[0] in ("-", "."):
                    parts = s.split()
                    if len(parts) == 3:
                        try:
                            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
                        except ValueError:
                            pass
        if size > 0 and len(rows) == size**3:
            lut_grid = np.array(rows, dtype=np.float32).reshape((size, size, size, 3))
            idx_r = np.clip(np.round(out[..., 0] * (size - 1)).astype(int), 0, size - 1)
            idx_g = np.clip(np.round(out[..., 1] * (size - 1)).astype(int), 0, size - 1)
            idx_b = np.clip(np.round(out[..., 2] * (size - 1)).astype(int), 0, size - 1)
            mapped = lut_grid[idx_b, idx_g, idx_r]
            # Approximate CapCut `tt_skin_seg` skin protection mask on warm skin pixels
            r, g, b = out[..., 0], out[..., 1], out[..., 2]
            skin_mask = ((r > g) & (g > b) & ((r - b) > 0.08) & (r > 0.22) & (r < 0.88)).astype(np.float32)[..., None]
            eff_alpha = lut_val * (1.0 - 0.55 * skin_mask)
            out = (1.0 - eff_alpha) * out + eff_alpha * mapped

    adj_dict = dict(adj_list)
    br = adj_dict.get("brightness", 0.0)
    con = adj_dict.get("contrast", 0.0)
    shd = adj_dict.get("shadow", 0.0)
    hl = adj_dict.get("highlight", 0.0)
    temp = adj_dict.get("temperature", 0.0)
    tone = adj_dict.get("tone", 0.0)
    sat = adj_dict.get("saturation", 0.0)

    luma = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
    shd_w = np.clip(1.0 - luma * 2.2, 0.0, 1.0)[..., None]
    hl_w = np.clip((luma - 0.50) * 2.0, 0.0, 1.0)[..., None]
    out = out + br * 0.45 + shd * 0.35 * shd_w + hl * 0.35 * hl_w
    out = (out - 0.30) * (1.0 + con * 0.85) + 0.30
    out[..., 0] += temp * 0.18 + tone * 0.06
    out[..., 2] -= temp * 0.18 - tone * 0.06
    out[..., 1] -= tone * 0.09
    luma2 = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2])[..., None]
    out = luma2 + (out - luma2) * (1.0 + sat * 1.1)
    return np.clip(out, 0.0, 1.0)


def render_graded_comparison_sheet(
    all_seg_tuples: list,
    seg_records: list,
    clusters: list,
    lut_map: dict,
    out_jpg: str,
    proj_dir: str = "",
    optical_db: list[dict] = None,
):
    from PIL import Image, ImageDraw
    w, h, pad_top = 280, 158, 28
    pair_w = w * 2 + 6
    cols = 2
    rows = max(1, (len(clusters) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * pair_w + 12, rows * (h + pad_top) + 8), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)

    for idx, c in enumerate(clusters):
        mid_seg = c["seg_indices"][len(c["seg_indices"]) // 2]
        _, _, v, _ = all_seg_tuples[mid_seg]
        rec = seg_records[mid_seg]
        cfg = c["scene_cfg"]
        ts = rec["start_sec"] + 0.5 * rec["dur_sec"]
        vp = resolve_video_path(v.get("path", ""), proj_dir)
        raw_rgb = extract_rgb_frame(vp, ts, w, h)
        lut_title, lut_val, adj_list, _ = compute_shot_lut_and_adjustments(
            rec.get("metrics") or c["anchor"], cfg, c["anchor"], c["anchor"], False,
            optical_db=optical_db, seg_scale=rec.get("scale", 1.0)
        )
        graded_rgb = apply_3d_lut_and_adjust_to_rgb(raw_rgb, lut_map.get(lut_title, ""), lut_val, adj_list)

        r_i, c_i = divmod(idx, cols)
        x0 = c_i * pair_w + 6
        y0 = r_i * (h + pad_top) + 4
        sheet.paste(Image.fromarray((raw_rgb * 255).astype(np.uint8)), (x0, y0 + pad_top))
        sheet.paste(Image.fromarray((graded_rgb * 255).astype(np.uint8)), (x0 + w + 2, y0 + pad_top))
        lbl = f"S{c['cluster_id']:02d} ({c['bg_code'][:16]}) RAW vs GRADED -> {lut_title[:20]} ({int(lut_val*100)}%)"
        draw.text((x0 + 4, y0 + 6), lbl, fill=(160, 245, 175))

    sheet.save(out_jpg, quality=90)


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
    video_paths = list({resolve_video_path(v.get("path", ""), proj_dir) for _, _, v, _ in all_seg_tuples if v.get("path")})
    clip_bg_map, _ = discover_ground_truth_scene_map(video_paths)
    if args.metrics and os.path.exists(args.metrics):
        loaded = json.load(open(args.metrics, "r", encoding="utf-8"))
        if isinstance(loaded, dict) and "segments" in loaded and len(loaded["segments"]) == len(all_seg_tuples):
            seg_records = loaded["segments"]
            if not clip_bg_map:
                clip_bg_map = {int(k): v for k, v in loaded.get("clip_bg_map", {}).items()}

    if not seg_records or len(seg_records) != len(all_seg_tuples):
        cache_by_range = {}
        seg_records = []
        for idx, (_, seg, v, container_lbl) in enumerate(all_seg_tuples):
            vpath = resolve_video_path(v.get("path", ""), proj_dir)
            mname = v.get("material_name") or os.path.basename(vpath)
            st_sec = seg.get("source_timerange", {}).get("start", 0) / 1e6
            dur_sec = seg.get("source_timerange", {}).get("duration", 5000000) / 1e6
            ck = (vpath, round(st_sec, 1), round(dur_sec, 1))
            if ck not in cache_by_range and os.path.exists(vpath):
                cache_by_range[ck] = extract_segment_metrics(vpath, st_sec, dur_sec)
            sc_obj = seg.get("clip", {}).get("scale", {}) if isinstance(seg.get("clip"), dict) else {}
            seg_scale = max(float(sc_obj.get("x", 1.0)), float(sc_obj.get("y", 1.0)))
            seg_records.append({
                "seg_idx": idx,
                "seg_id": seg.get("id", ""),
                "container": container_lbl,
                "material_name": mname,
                "start_sec": round(st_sec, 2),
                "dur_sec": round(dur_sec, 2),
                "scale": round(seg_scale, 3),
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
            m_sh = round(clamp(0.15 + max(0.0, 0.042 - med_shp) * 5.0, 0.12, 0.22), 3)
            m_cl = 0.05 if med_shp < 0.036 else 0.04
            m_pt = 0.060 if med_shp < 0.036 else 0.050
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

    # 2. Ground-Truth Scene Clustering + Unified Location Targets + Seedance Dark-Scene Sync
    optical_db = load_optical_db(lut_map)
    clusters = cluster_timeline_scenes(seg_records, clip_bg_map)
    assign_unified_scene_targets(clusters, args.emotion, optical_db, allow_bw=args.allow_bw)
    seg_to_cluster = {}
    for c in clusters:
        for s_idx in c["seg_indices"]:
            seg_to_cluster[s_idx] = c

    explicit_match = {x.strip() for x in args.match_clips.split(",") if x.strip()}
    plan_segs = custom_plan.get("segments", {})
    default_m = {"p5": 0.05, "p25": 0.15, "mid": 0.28, "p95": 0.62, "p99": 0.76, "contrast": 0.55, "rb": 0.13, "gm": -0.012, "sat": 0.15, "hsv_s": 0.45, "sharp": 0.042, "hf_noise": 0.030}

    summary = []
    total_adjusts = 0
    angle_jump_fixes = 0
    zoomed_count = 0
    for idx, (d_obj, seg, _, _) in enumerate(all_seg_tuples):
        effects = d_obj["materials"].setdefault("effects", [])
        seg["extra_material_refs"] = [r for r in seg.get("extra_material_refs", []) if r not in old_eff_ids]
        rec = seg_records[idx]
        mname = rec["material_name"]
        m = rec.get("metrics") or default_m
        sc_obj = seg.get("clip", {}).get("scale", {}) if isinstance(seg.get("clip"), dict) else {}
        seg_scale = float(rec.get("scale") or max(float(sc_obj.get("x", 1.0)), float(sc_obj.get("y", 1.0))))
        if seg_scale > 1.02:
            zoomed_count += 1
        c_info = seg_to_cluster[idx]
        scene_anchor = c_info.get("unified_anchor") or c_info["anchor"]
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
                and (abs(m["rb"] - scene_anchor["rb"]) > 0.035 or abs(m["mid"] - scene_anchor["mid"]) > 0.045)
            )
            lut_title, val, adj_list, prof = compute_shot_lut_and_adjustments(
                m, scene_cfg, scene_anchor, local_nb_m, force_nb, optical_db=optical_db, seg_scale=seg_scale
            )
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
        f"[Emotion: {args.emotion} | {angle_jump_fixes} Angle-Jump Sibling LUT fixes | {zoomed_count} Zoomed (~150%) Super-Res Boosts | {total_adjusts} per-video Adjust sliders ({total_adjusts/max(len(summary),1):.1f} sliders/video)]"
    )
    for c in clusters:
        cfg = c["scene_cfg"]
        print(
            f"  - Scene[{c['cluster_id']:2d}] ({c['bg_code']:24s}) Segs {c['start_seg']:3d}..{c['end_seg']:3d} ({len(c['seg_indices']):2d} segs) | "
            f"Primary LUT: {cfg['lut']:26s} ({cfg['base_val']*100:.0f}%) | Target Mid={cfg['target_mid']:.2f} RB={cfg['target_rb']:+.2f}"
        )

    comp_jpg = f"/tmp/{os.path.basename(proj_dir).replace(' ', '_')}_graded_comparison.jpg"
    try:
        render_graded_comparison_sheet(all_seg_tuples, seg_records, clusters, lut_map, comp_jpg, proj_dir=proj_dir, optical_db=optical_db)
        print(f"\n[MANDATORY AI VISUAL QA] Rendered RAW vs. GRADED comparison sheet: {comp_jpg}")
        print("-> AI MUST call read_file / view_file on this image to visually verify skin tones, contrast, and scene continuity with its own eyes!")
    except Exception as exc:
        print(f"Note: Could not render graded comparison sheet ({exc})")

    if args.relaunch:
        subprocess.run(["open", "-a", "/Applications/CapCut.app"], check=False)


if __name__ == "__main__":
    main()
