#!/usr/bin/env python3
"""Model-Aware, Script-Adaptive & Scene-Continuous 14-Parameter CapCut Grading Engine:
- Auto-detects AI Video Model (`seedance`, `grok`, `omni`, `hybrid`) from un-cut source video durations (`video suộc gốc`):
  * No source video > 10s (<= 10.5s) -> `omni`
  * Source videos span 6s / 10s / 15s (none > 15.5s) -> `grok`
  * Has source video(s) > 15.5s (~25s-30s) -> `seedance` (or `hybrid` when >= 65% of sources are <= 10.5s Omni/Grok clips)
- Uses Ground-Truth BANGTRA / PROMPT_ANH_REF / sf-board.json scene clusters and script lighting regimes
  (`DAWN_DUSK_GOLD`, `DAY_CRISP`, `WARM_INT`, `NIGHT_WARM`, `NIGHT_COOL`, `COOL_TECH`).
- Separates Stage 1 (Creative 35mm Primary LUT + Angle-Jump Counter-Balance Sibling LUT at zorder 2000)
  from Stage 2 (Scene Defect Conditioning + Non-Canceling Intra-Scene Shot Continuity Lock at zorder 8004)
  so healthy scenes retain 100% of their 35mm LUT character while intra-scene camera cuts lock to identical color.
"""
import argparse
import copy
import glob
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from analyze_project import (
    cluster_timeline_scenes,
    collect_all_video_segments,
    detect_source_models,
    discover_ground_truth_scene_map,
    extract_clip_number,
    extract_rgb_frame,
    extract_segment_metrics,
    get_default_metrics_cache_path,
    resolve_project_dir,
    resolve_video_path,
)

LUT_CFG_PATH = "/Users/daho100/Movies/CapCut/User Data/Resources/Lut/LutImportCfg.json"
DEFAULT_EFFECT_CACHE = "/Users/daho100/Library/Containers/com.lemon.lvoverseas/Data/Movies/CapCut/User Data/Cache/effect/7501974767453474064/26e574bcfa176c3c1a1f7fd5df0a4996"
OPTICAL_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "lut_optical_db.json")

_CUBE_GRID_CACHE = {}


def load_optical_db(lut_map: dict) -> list[dict]:
    if not os.path.exists(OPTICAL_DB_PATH):
        return []
    with open(OPTICAL_DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)
    seen_bnames = set()
    seen_sigs = set()
    deduped = []
    for entry in db:
        t = entry.get("title", "")
        if t not in lut_map or not t.lower().endswith(".cube"):
            continue
        bname = os.path.basename(lut_map[t])
        sig = (
            round(entry.get("blk", 0.05), 3),
            round(entry.get("mid", 0.28), 3),
            round(entry.get("hi", 0.75), 3),
            round(entry.get("midRB", 0.0), 3),
            round(entry.get("satGain", 1.0), 2),
        )
        if bname in seen_bnames or sig in seen_sigs:
            continue
        seen_bnames.add(bname)
        seen_sigs.add(sig)
        deduped.append(entry)
    return deduped


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


def promote_active_luts_in_cfg(active_lut_titles: list[str], proj_dir: str) -> list[int]:
    """
    Promotes active project LUTs into `LutImportCfg.json[0:100]` (`index <= 99`) so CapCut's C++
    ComboBox (`ImportLUTDataManager::loadImportedLutList` & `VELutSpinComboBoxViewModel`) never shows "None".
    Uses fast regex scanning across other drafts instead of full JSON parsing.
    """
    if not os.path.exists(LUT_CFG_PATH):
        return []
    with open(LUT_CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, list) or not cfg:
        return []

    by_title = {}
    for item in cfg:
        t = item.get("title")
        if t and t not in by_title and os.path.exists(item.get("path", "")):
            by_title[t] = item

    other_used = set()
    drafts_root = os.path.dirname(os.path.abspath(proj_dir))
    lut_name_re = re.compile(r'"name"\s*:\s*"([^"]+\.(?:cube|3dl))"', re.IGNORECASE)
    for d_path in sorted(glob.glob(os.path.join(drafts_root, "*", "draft_info.json"))):
        if os.path.abspath(os.path.dirname(d_path)) == os.path.abspath(proj_dir):
            continue
        try:
            with open(d_path, "r", encoding="utf-8", errors="ignore") as df:
                raw_txt = df.read()
            for match_name in lut_name_re.findall(raw_txt):
                if match_name in by_title:
                    other_used.add(match_name)
        except Exception:
            pass

    top_items = []
    used_titles = set()
    used_basenames = set()

    def try_add_top(title: str):
        if title in used_titles or title not in by_title:
            return
        item = by_title[title]
        bname = os.path.basename(item.get("path", ""))
        if len(top_items) < 100 and bname in used_basenames:
            return
        top_items.append(item)
        used_titles.add(title)
        used_basenames.add(bname)

    for t in sorted(set(active_lut_titles)):
        if t in by_title:
            item = by_title[t]
            top_items.append(item)
            used_titles.add(t)
            used_basenames.add(os.path.basename(item.get("path", "")))

    for t in sorted(other_used):
        try_add_top(t)

    for item in cfg:
        if len(top_items) >= 100:
            break
        t = item.get("title")
        if t:
            try_add_top(t)

    tail_items = [item for item in cfg if item.get("title") not in used_titles]
    new_cfg = top_items + tail_items

    with open(LUT_CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, ensure_ascii=False, indent=4)

    title_to_idx = {item.get("title"): idx for idx, item in enumerate(new_cfg)}
    return [title_to_idx.get(t, -1) for t in sorted(set(active_lut_titles))]


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
    if mid < 0.225 or (p5 < 0.028 and mid < 0.245):
        return "LOW_KEY_DARK"
    if c > 0.65 and p95 > 0.72:
        return "HARSH_CONTRAST"
    if rb >= 0.165:
        return "HEAVY_WARM_CAST"
    if rb <= 0.045:
        return "COLD_CYAN_CAST"
    return "BALANCED_REC709"


def simulate_video_through_lut(video_m: dict, lut_entry: dict, alpha: float) -> dict:
    """
    Mathematically simulate applying `lut_entry` (at intensity `alpha` with CapCut's
    `enable_skin_tone_correction: true` / `tt_skin_seg` neural skin mask active)
    onto the measured video `video_m`.
    """
    p5 = max(0.005, video_m.get("p5", 0.05))
    mid = max(0.02, video_m.get("mid", 0.28))
    p95 = max(0.10, video_m.get("p95", 0.62))
    rb = video_m.get("rb", 0.13)
    gm = video_m.get("gm", -0.012)
    sat = video_m.get("sat", 0.15)

    lut_blk_mapped = clamp(lut_entry["blk"] * (p5 / 0.05), 0.0, 1.0)
    lut_mid_mapped = clamp(lut_entry["mid"] * (mid / 0.28), 0.0, 1.0)
    lut_hi_mapped = clamp(lut_entry["hi"] * (p95 / 0.75), 0.0, 1.0)

    sim_p5 = (1.0 - alpha) * p5 + alpha * lut_blk_mapped
    sim_mid = (1.0 - alpha) * mid + alpha * lut_mid_mapped
    sim_hi = (1.0 - alpha) * p95 + alpha * lut_hi_mapped

    sim_rb = rb + alpha * lut_entry["midRB"]
    sim_gm = gm + alpha * (0.28 * (lut_entry.get("sknGM", -0.025) - (-0.025)) + 0.72 * lut_entry["midGM"])
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
    allow_stylized: bool = False,
    target_p5: float = 0.046,
    target_hi: float = 0.650,
    used_lut_counts: dict = None,
    diversity_weight: float = 0.012,
    is_primary: bool = True,
    regime: str = "DAY_CRISP",
) -> tuple[str, float, str]:
    """
    Selects the optimal (LUT, alpha) pair from the 3D-trilinear optical LUT database:
    - For Primary Scene LUTs (`is_primary=True`), evaluates `alpha in (0.18, 0.20, 0.22, 0.24)` and rewards
      authentic 35mm contrast separation (`hi - blk >= 0.68`) so the LUT imparts a real cinema look.
    - For Intra-Scene Counter-Balance Sibling LUTs (`is_primary=False`), evaluates `alpha in (0.15, 0.18, 0.21, 0.24)`
      to pull an angle-jump shot's Post-LUT state directly toward the Scene Anchor's Post-LUT state.
    - Enforces regime-aware color guardrails so heavy cyan/teal LUTs (`midRB < -0.055` or `midGM > +0.018`)
      are never applied to warm or neutral-daylight scenes.
    """
    v_state = diagnose_video_state(video_m)
    if not optical_db:
        return fallback_lut, 0.20, v_state

    is_mono_video = (v_state == "MONOCHROME_INPUT") or allow_bw
    min_allowed_sat = min(0.050, video_m.get("sat", 0.10) * 0.72)

    best_title = fallback_lut
    best_alpha = 0.20
    best_score = 1e9

    candidate_alphas = (0.18, 0.20, 0.22, 0.24) if is_primary else (0.15, 0.18, 0.21, 0.24)

    cinematic_safe_prefixes = (
        "FILM LOOK", "CINEMATIC", "ATMOSPHERIC", "HOLLYWOOD",
        "MOOD", "SKINTONE", "SCC-REC709", "LUT ", "COMMERCIAL",
        "THE ROMANCE", "DRAMATIC", "CW_WEDDING"
    )

    allowed_tiers = {"core_rec709", "cinema_rec709"}
    if allow_stylized:
        allowed_tiers.add("stylized_rec709")
    if is_mono_video:
        allowed_tiers.add("bw_mono")

    allow_heavy_teal = regime in ("COOL_TECH", "NIGHT_COOL")
    is_warm_regime = regime in ("DAWN_DUSK_GOLD", "WARM_INT")

    for entry in optical_db:
        tier = entry.get("tier", "")
        if tier not in allowed_tiers:
            continue
        if not is_mono_video and (entry.get("satGain", 1.0) < 0.74 or entry.get("satGain", 1.0) > 1.32):
            continue
        if entry.get("blk", 0.05) < 0.022 or entry.get("blk", 0.05) > 0.095:
            continue
        if entry.get("mid", 0.28) < 0.220 or entry.get("mid", 0.28) > 0.355:
            continue
        if abs(entry.get("hiRB", 0.0)) > 0.085 or abs(entry.get("hiGM", 0.0)) > 0.055:
            continue

        # Regime-aware LUT tint guardrails: prevent cyan/teal LUTs from washing out warm/daylight scenes
        mid_rb_lut = entry.get("midRB", 0.0)
        mid_gm_lut = entry.get("midGM", 0.0)
        if not allow_heavy_teal and (mid_rb_lut < -0.055 or mid_gm_lut > 0.018):
            continue
        if is_warm_regime and mid_rb_lut < -0.038:
            continue

        t = entry["title"]
        t_up = t.upper()
        if not is_mono_video:
            if not any(t_up.startswith(pfx) for pfx in cinematic_safe_prefixes):
                continue
            if any(bad in t_up for bad in ("SEPIA", "BW", "B&W", "MONO")):
                continue

        fam_bonus = -0.024 if any(k in t_up for k in preferred_families) else 0.0
        lut_span = entry.get("hi", 0.75) - entry.get("blk", 0.05)
        film_curve_bonus = -0.012 if (0.68 <= lut_span <= 0.80 and 0.032 <= entry.get("blk", 0.05) <= 0.068) else 0.0
        div_pen = min(0.060, used_lut_counts.get(t, 0) * diversity_weight) if used_lut_counts else 0.0

        for alpha in candidate_alphas:
            sim = simulate_video_through_lut(video_m, entry, alpha)
            if not (sim["sim_p5"] < sim["sim_mid"] < sim["sim_hi"]):
                continue
            if sim["sim_p5"] < 0.010 or sim["sim_p5"] > max(0.140, video_m.get("p5", 0.05) + 0.018):
                continue
            if sim["sim_hi"] > max(0.90, video_m.get("p95", 0.62) + 0.025) or sim["sim_hi"] < 0.38:
                continue
            if not is_mono_video and (sim["sim_sat"] < min_allowed_sat or sim["sim_sat"] > max(0.265, video_m.get("sat", 0.15) * 1.12)):
                continue

            err_mid = abs(sim["sim_mid"] - target_mid) * 2.6
            err_rb = abs(sim["sim_rb"] - target_rb) * 3.4
            err_p5 = abs(sim["sim_p5"] - target_p5) * 2.2
            err_hi = abs(sim["sim_hi"] - target_hi) * 1.4
            err_gm = abs(sim["sim_gm"] - target_gm) * 2.0
            err_sat = abs(sim["sim_sat"] - target_sat) * 1.8

            score = err_mid + err_rb + err_p5 + err_hi + err_gm + err_sat + fam_bonus + film_curve_bonus + div_pen
            if score < best_score:
                best_score = score
                best_title = t
                best_alpha = alpha

    if used_lut_counts is not None:
        used_lut_counts[best_title] = used_lut_counts.get(best_title, 0) + 1

    return best_title, best_alpha, v_state


def select_scene_primary_lut_and_targets(
    anchor: dict,
    regime: str,
    project_model: str,
    emotion: str,
    progress_ratio: float,
    optical_db: list[dict] = None,
    allow_bw: bool = False,
    allow_stylized: bool = False,
    is_flashback: bool = False,
    bg_code: str = "",
    seedance_ref: dict = None,
    used_primary_luts: dict = None,
) -> dict:
    """
    Model-Aware & Script-Adaptive Scene Target + Primary LUT Selector:
    Adapts dynamically to:
      1. The existing measured color of the scene (`anchor`),
      2. The screenplay's lighting regime (`DAWN_DUSK_GOLD`, `DAY_CRISP`, `WARM_INT`, `NIGHT_WARM`, `NIGHT_COOL`, `COOL_TECH`),
      3. The detected AI Video Model (`seedance`, `grok`, `omni`, `hybrid`).
    """
    p5 = anchor.get("p5", 0.05)
    mid = anchor.get("mid", 0.30)
    p95 = anchor.get("p95", 0.65)
    rb = anchor.get("rb", 0.12)
    gm = anchor.get("gm", -0.010)
    sat = anchor.get("sat", 0.14)
    skin_rb = anchor.get("skin_rb", 0.20)
    skin_sat = anchor.get("skin_sat", 0.42)

    is_seedance_hook = (regime == "SEEDANCE_HOOK") or bg_code.startswith("SEEDANCE_HOOK")
    use_bw = allow_bw and is_flashback

    # 1. Special Hybrid Seedance Hook & Omni Night/Storm Synchronization
    if is_seedance_hook:
        act_mood = "seedance-35mm-ref"
        target_p5 = clamp(p5 * 0.50, 0.038, 0.052)
        target_mid = clamp(mid, 0.305, 0.355)
        target_hi = clamp(p95, 0.635, 0.710)
        target_rb = clamp(rb, 0.010, 0.035)
        target_gm = -0.008
        target_sat = clamp(sat * 1.05, 0.095, 0.125)
        desired_lut_rb = -0.005
        use_vignette = False
        fb_lut, fams = "SKINTONE (4).cube", ("SKINTONE", "FILM LOOK", "CINEMATIC", "MOOD", "LUT ")
    elif seedance_ref and regime in ("NIGHT_WARM", "NIGHT_COOL") and (rb <= 0.045 or mid <= 0.265):
        # Synchronize dark/rainy/evening Omni scenes in a Hybrid project to the 30s Seedance Hook reference
        act_mood = "seedance-dark-sync" if rb <= 0.045 else "seedance-dark-warm"
        target_p5 = seedance_ref.get("target_p5", 0.045)
        target_mid = clamp(max(mid + 0.045, seedance_ref.get("target_mid", 0.310)), 0.300, 0.340)
        target_hi = seedance_ref.get("target_hi", 0.645)
        target_rb = seedance_ref.get("target_rb", 0.020) if rb <= 0.045 else clamp(rb, 0.065, 0.105)
        target_gm = -0.008
        target_sat = clamp(sat, 0.100, 0.125)
        desired_lut_rb = -0.010 if rb <= 0.045 else +0.005
        use_vignette = False
        fb_lut = "SKINTONE (4).cube" if rb <= 0.045 else "Mood 9 Skin rec709.cube"
        fams = ("SKINTONE", "MOOD", "FILM LOOK", "CINEMATIC", "LUT ")
    else:
        # 2. Determine narrative / emotional mood label
        if emotion == "narrative-arc":
            if progress_ratio < 0.18:
                act_mood = "luxury-crisp"
            elif progress_ratio < 0.65:
                act_mood = "dramatic-contrast"
            elif progress_ratio < 0.85:
                act_mood = "climax-punch"
            else:
                act_mood = "warm-human"
        else:
            act_mood = emotion

        # 3. Dynamic Scene Exposure Targets (adapting to existing film brightness + model characteristics)
        if p5 > 0.095:
            target_p5 = clamp(p5 - 0.035, 0.044, 0.082)
        elif p5 < 0.032:
            target_p5 = 0.042
        else:
            target_p5 = clamp(p5, 0.042, 0.075)

        if mid < 0.230:
            # Very dark night/server scene: lift cleanly so facial expressions read clearly
            target_mid = clamp(mid + 0.060, 0.235, 0.280)
        elif mid < 0.275:
            # Low-key interior/evening: open shadows & midtones gently
            target_mid = clamp(mid + 0.028, 0.268, 0.298)
        elif mid > 0.455:
            # High-key outdoor glare / window blow-out: pull down harder on Grok/Omni than on sunny Seedance
            pull = 0.050 if project_model in ("grok", "omni", "hybrid") else 0.022
            target_mid = clamp(mid - pull, 0.395, 0.460)
        else:
            # Healthy cinematic exposure (0.275 .. 0.455): preserve the scene's natural exposure!
            if project_model == "grok" and mid > 0.385:
                target_mid = round(mid - 0.015, 4)
            else:
                target_mid = mid

        if project_model == "grok" and p95 > 0.680:
            target_hi = clamp(p95 - 0.045, 0.610, 0.705)
        elif p95 > 0.785:
            target_hi = clamp(p95 - 0.040, 0.680, 0.755)
        elif p95 < 0.560:
            target_hi = clamp(p95 + 0.045, 0.580, 0.650)
        else:
            target_hi = clamp(p95, 0.615, 0.750)

        # 4. Dynamic Color & Saturation Targets by Script Lighting Regime + Model Defect Profile
        # Use achromatic white-balance (`neutral_rb`) alongside `skin_rb` so colorful props/hats/brick walls
        # (`neutral_rb <= 0.082`) are never misclassified as an ambient orange lighting cast.
        neutral_rb = anchor.get("neutral_rb", rb * 0.45)
        is_hot_orange_defect = (
            neutral_rb > 0.082
            and (
                (project_model == "grok" and (rb > 0.172 or skin_rb > 0.238 or skin_sat > 0.495))
                or (project_model != "grok" and (rb > 0.205 or skin_rb > 0.265))
            )
        )

        if regime == "DAWN_DUSK_GOLD":
            if is_hot_orange_defect:
                target_rb = clamp(rb - 0.032, 0.130, 0.170) if project_model == "grok" else clamp(rb - 0.025, 0.145, 0.180)
                desired_lut_rb = -0.010
            else:
                target_rb = clamp(rb + 0.008, 0.115, 0.195)
                desired_lut_rb = +0.014
            target_gm = -0.012
            target_sat = clamp(sat * (0.95 if is_hot_orange_defect else 1.03), 0.110, 0.190)
            fb_lut = "THE ROMANCE 05.cube" if desired_lut_rb > 0 else "SKINTONE (10).cube"
            fams = ("THE ROMANCE", "MOOD", "SKINTONE", "LUT ", "FILM LOOK")

        elif regime == "WARM_INT":
            if is_hot_orange_defect:
                target_rb = clamp(rb - 0.035, 0.115, 0.155) if project_model == "grok" else clamp(rb - 0.028, 0.125, 0.165)
                desired_lut_rb = -0.012
            else:
                target_rb = clamp(rb + 0.005, 0.085, 0.180)
                desired_lut_rb = +0.008
            target_gm = -0.010
            target_sat = clamp(sat * (0.94 if is_hot_orange_defect else 1.02), 0.100, 0.175)
            fb_lut = "SKINTONE (10).cube"
            fams = ("SKINTONE", "MOOD", "LUT ", "CINEMATIC", "FILM LOOK")

        elif regime == "NIGHT_WARM":
            if rb <= 0.045:
                # Cool rainy night sub-scene inside an evening sequence
                target_rb = clamp(rb + 0.008, 0.008, 0.050)
                desired_lut_rb = -0.010
                fb_lut = "ATMOSPHERIC (12).cube"
                fams = ("ATMOSPHERIC", "CINEMATIC", "FILM LOOK", "SKINTONE")
            elif is_hot_orange_defect:
                target_rb = clamp(rb - 0.036, 0.115, 0.158) if project_model == "grok" else clamp(rb - 0.028, 0.130, 0.170)
                desired_lut_rb = -0.014
                fb_lut = "CINEMATIC  (27).cube"
                fams = ("CINEMATIC", "SKINTONE", "FILM LOOK", "ATMOSPHERIC", "MOOD")
            else:
                target_rb = clamp(rb, 0.075, 0.185)
                desired_lut_rb = -0.005
                fb_lut = "ATMOSPHERIC (12).cube"
                fams = ("ATMOSPHERIC", "CINEMATIC", "SKINTONE", "FILM LOOK", "MOOD")
            target_gm = -0.010
            target_sat = clamp(sat * (0.94 if is_hot_orange_defect else 1.01), 0.095, 0.175)

        elif regime in ("NIGHT_COOL", "COOL_TECH"):
            target_rb = clamp(rb, -0.015, 0.075)
            desired_lut_rb = -0.018
            target_gm = -0.006
            target_sat = clamp(sat * 0.98, 0.080, 0.145)
            fb_lut = "DRAMATIC (14).cube"
            fams = ("DRAMATIC", "CINEMATIC", "FILM LOOK", "ATMOSPHERIC", "SCC-REC709")

        else:
            # DAY_CRISP (Daylight Office, Street, Hangar, Courthouse, Beach Day, Outdoor Event)
            if is_hot_orange_defect:
                target_rb = clamp(rb - 0.032, 0.105, 0.145) if project_model == "grok" else clamp(rb - 0.024, 0.115, 0.155)
                desired_lut_rb = -0.014
            elif rb < 0.035:
                # Cold/cyan Omni daylight -> warm subtly toward clean neutral daylight
                target_rb = clamp(rb + 0.022, 0.035, 0.075)
                desired_lut_rb = +0.010
            else:
                target_rb = rb
                desired_lut_rb = -0.006
            target_gm = -0.008
            target_sat = clamp(sat * (0.94 if is_hot_orange_defect else 1.03), 0.090, 0.165)
            fb_lut = "FILM LOOK (4).cube" if desired_lut_rb < 0 else "SKINTONE (10).cube"
            fams = ("FILM LOOK", "SCC-REC709", "SKINTONE", "LUT ", "COMMERCIAL", "CINEMATIC")

        use_vignette = bool(is_flashback or act_mood == "vintage-35mm")

    # Stage 1 Primary LUT Selection: match toward the creative 35mm target (with film contrast & desired_lut_rb)
    lut_target_p5 = clamp(target_p5 - 0.005, 0.032, 0.072)
    lut_target_hi = clamp(target_hi + 0.012, 0.620, 0.780)
    lut_target_rb = target_rb + desired_lut_rb

    lut, base_val, v_state = match_optimal_lut_by_video_simulation(
        optical_db, anchor, target_mid, lut_target_rb, target_gm, target_sat, fams, fb_lut,
        allow_bw=use_bw, allow_stylized=allow_stylized,
        target_p5=lut_target_p5, target_hi=lut_target_hi,
        used_lut_counts=used_primary_luts, diversity_weight=0.002, is_primary=True,
        regime=regime,
    )

    lut_entry = next((e for e in (optical_db or []) if e["title"] == lut), None)
    if lut_entry:
        sim_anchor = simulate_video_through_lut(anchor, lut_entry, base_val)
    else:
        sim_anchor = {
            "sim_p5": p5, "sim_mid": mid, "sim_hi": p95,
            "sim_contrast": p95 - p5, "sim_rb": rb, "sim_gm": gm, "sim_sat": sat,
        }

    prof = f"[{project_model.upper()} | {regime} | {v_state}]"

    return {
        "act_mood": act_mood,
        "regime": regime,
        "project_model": project_model,
        "video_state": v_state,
        "lut": lut,
        "base_val": base_val,
        "profile": prof,
        "target_p5": round(target_p5, 4),
        "target_mid": round(target_mid, 4),
        "target_hi": round(target_hi, 4),
        "target_rb": round(target_rb, 4),
        "target_gm": round(target_gm, 4),
        "target_sat": round(target_sat, 4),
        "sim_anchor": sim_anchor,
        "use_vignette": use_vignette,
        "preserve_warm_props": bool(neutral_rb <= 0.082 and rb >= 0.130),
        "temp_bias": 0.0,
        "bright_bias": 0.0,
        "contrast_bias": 0.0,
        "sat_bias": 0.0,
        "shadow_bias": 0.0,
        "highlight_bias": 0.0,
        "tone_bias": 0.0,
        "ai_note": "",
    }


def _apply_ai_scene_plan_override(cfg: dict, scene_anchor: dict, sc_plan: dict, global_plan: dict, optical_db: list[dict]):
    """Applies the AI Colorist's explicit scene-level direction (`sc_plan` + `global_plan`) onto `cfg`."""
    if not sc_plan and not global_plan:
        return
    sc_plan = sc_plan or {}
    global_plan = global_plan or {}

    if "regime" in sc_plan and sc_plan["regime"]:
        cfg["regime"] = sc_plan["regime"]
    if "target_p5" in sc_plan:
        cfg["target_p5"] = round(float(sc_plan["target_p5"]), 4)
    if "target_mid" in sc_plan:
        cfg["target_mid"] = round(float(sc_plan["target_mid"]), 4)
    elif "mid_shift" in sc_plan:
        cfg["target_mid"] = round(clamp(cfg["target_mid"] + float(sc_plan["mid_shift"]), 0.18, 0.52), 4)
    if "target_hi" in sc_plan:
        cfg["target_hi"] = round(float(sc_plan["target_hi"]), 4)
    if "target_rb" in sc_plan:
        cfg["target_rb"] = round(float(sc_plan["target_rb"]), 4)
    elif "rb_shift" in sc_plan:
        cfg["target_rb"] = round(clamp(cfg["target_rb"] + float(sc_plan["rb_shift"]), -0.05, 0.25), 4)
    if "target_gm" in sc_plan:
        cfg["target_gm"] = round(float(sc_plan["target_gm"]), 4)
    if "target_sat" in sc_plan:
        cfg["target_sat"] = round(float(sc_plan["target_sat"]), 4)
    elif "sat_shift" in sc_plan:
        cfg["target_sat"] = round(clamp(cfg["target_sat"] + float(sc_plan["sat_shift"]), 0.05, 0.25), 4)

    if "preserve_warm_props" in sc_plan:
        cfg["preserve_warm_props"] = bool(sc_plan["preserve_warm_props"])
    if "use_vignette" in sc_plan:
        cfg["use_vignette"] = bool(sc_plan["use_vignette"])

    # If the AI Colorist specified preferred_families (without hardcoding a LUT), re-run optical LUT simulation
    if "preferred_families" in sc_plan and "lut" not in sc_plan:
        fams = tuple(str(x).upper() for x in sc_plan["preferred_families"])
        lut, base_val, _ = match_optimal_lut_by_video_simulation(
            optical_db, scene_anchor, cfg["target_mid"], cfg["target_rb"], cfg["target_gm"], cfg["target_sat"],
            fams, cfg["lut"], target_p5=cfg["target_p5"], target_hi=cfg["target_hi"],
            is_primary=True, regime=cfg["regime"],
        )
        cfg["lut"] = lut
        cfg["base_val"] = base_val

    if "lut" in sc_plan and sc_plan["lut"]:
        if isinstance(sc_plan["lut"], (list, tuple)) and len(sc_plan["lut"]) == 2:
            cfg["lut"] = str(sc_plan["lut"][0])
            cfg["base_val"] = round(float(sc_plan["lut"][1]), 3)
        else:
            cfg["lut"] = str(sc_plan["lut"])
    if "lut_val" in sc_plan:
        cfg["base_val"] = round(float(sc_plan["lut_val"]), 3)
    elif "base_val" in sc_plan:
        cfg["base_val"] = round(float(sc_plan["base_val"]), 3)

    if "lut_val_bias" in global_plan:
        cfg["base_val"] = round(clamp(cfg["base_val"] + float(global_plan["lut_val_bias"]), 0.10, 0.32), 3)

    for bias_key in ("temp_bias", "bright_bias", "contrast_bias", "sat_bias", "shadow_bias", "highlight_bias", "tone_bias"):
        g_b = float(global_plan.get(bias_key, 0.0))
        s_b = float(sc_plan.get(bias_key, 0.0))
        cfg[bias_key] = round(g_b + s_b, 4)

    if sc_plan.get("note"):
        cfg["ai_note"] = str(sc_plan["note"])
        cfg["profile"] = f"[{cfg['project_model'].upper()} | {cfg['regime']} | AI-Plan: {cfg['ai_note'][:28]}]"

    lut_entry = next((e for e in (optical_db or []) if e["title"] == cfg["lut"]), None)
    if lut_entry:
        cfg["sim_anchor"] = simulate_video_through_lut(scene_anchor, lut_entry, cfg["base_val"])


def assign_unified_scene_targets(
    clusters: list,
    project_model: str,
    emotion: str,
    optical_db: list[dict],
    allow_bw: bool = False,
    allow_stylized: bool = False,
    custom_plan: dict = None,
):
    """
    1. Honors any AI Colorist scene-level regime overrides (`custom_plan["scenes"]`) BEFORE location grouping.
    2. Groups scenes sharing the same `(regime, base_location)` so returning to the same room/location
       under the same lighting regime shares the same Primary 35mm LUT family/stock.
    3. Blends each cluster's own anchor (`c["anchor"]`) with the location-group anchor (`loc_anchor`)
       so intra-scene shots lock tightly to the scene's actual lighting while cross-scene returns to the
       same location maintain consistent color identity.
    4. Applies the AI Colorist's bespoke scene targets, Primary LUT choices, and post-LUT biases.
    """
    custom_plan = custom_plan or {}
    plan_scenes = custom_plan.get("scenes", {}) or {}
    global_plan = custom_plan.get("global", {}) or {}

    def _get_sc_plan(c: dict) -> dict:
        cid_str = str(c["cluster_id"])
        bg = c["bg_code"]
        base_loc = bg.split(":")[-1] if ":" in bg else bg
        return plan_scenes.get(cid_str) or plan_scenes.get(f"S{c['cluster_id']:02d}") or plan_scenes.get(bg) or plan_scenes.get(base_loc) or {}

    # Pre-pass: apply AI Colorist regime overrides before grouping locations
    for c in clusters:
        sc_p = _get_sc_plan(c)
        if sc_p.get("regime"):
            c["regime"] = str(sc_p["regime"])

    total_segs = sum(len(c["seg_indices"]) for c in clusters) or 1
    has_seedance_hook = any(c["bg_code"].startswith("SEEDANCE_HOOK") or c.get("regime") == "SEEDANCE_HOOK" for c in clusters)
    seedance_ref = None
    if has_seedance_hook:
        hook_c = next(c for c in clusters if c["bg_code"].startswith("SEEDANCE_HOOK") or c.get("regime") == "SEEDANCE_HOOK")
        ha = hook_c["anchor"]
        seedance_ref = {
            "target_p5": round(clamp(ha["p5"] * 0.50, 0.038, 0.052), 4),
            "target_mid": round(clamp(ha["mid"], 0.305, 0.345), 4),
            "target_hi": round(clamp(ha["p95"], 0.635, 0.690), 4),
            "target_rb": round(clamp(ha["rb"], 0.012, 0.032), 4),
            "target_gm": -0.008,
            "target_sat": round(clamp(ha["sat"] * 1.05, 0.098, 0.120), 4),
        }

    loc_groups = defaultdict(list)
    for c in clusters:
        bg_code = c["bg_code"]
        regime = c.get("regime", "DAY_CRISP")
        base_loc = bg_code.split(":")[-1] if ":" in bg_code else bg_code
        if bg_code.startswith("SEEDANCE_HOOK") or regime == "SEEDANCE_HOOK":
            gkey = ("SEEDANCE_HOOK", "SEEDANCE_HOOK")
        else:
            gkey = (regime, base_loc)
        c["_group_key"] = gkey
        loc_groups[gkey].append(c)

    group_primary = {}
    used_primary_luts = {}
    for gkey, g_clusters in loc_groups.items():
        regime, _ = gkey
        keys = list(g_clusters[0]["anchor"].keys())
        loc_anchor = {}
        weights = [max(1, len(x["seg_indices"])) for x in g_clusters]
        for k in keys:
            vals = [x["anchor"][k] for x in g_clusters]
            loc_anchor[k] = round(float(np.average(vals, weights=weights)), 4)

        avg_prog = float(np.average([0.5 * (x["start_seg"] + x["end_seg"]) / total_segs for x in g_clusters], weights=weights))
        rep_c = g_clusters[0]
        group_cfg = select_scene_primary_lut_and_targets(
            loc_anchor, regime, project_model, emotion, avg_prog,
            optical_db=optical_db, allow_bw=allow_bw, allow_stylized=allow_stylized,
            is_flashback=rep_c.get("is_flashback", False), bg_code=rep_c["bg_code"],
            seedance_ref=seedance_ref, used_primary_luts=used_primary_luts,
        )
        # Check if any cluster in this location group has an explicit AI LUT in custom_plan so the whole location group can share it
        for gc in g_clusters:
            scp = _get_sc_plan(gc)
            if scp.get("lut"):
                if isinstance(scp["lut"], (list, tuple)) and len(scp["lut"]) == 2:
                    group_cfg["lut"] = str(scp["lut"][0])
                    group_cfg["base_val"] = round(float(scp["lut"][1]), 3)
                else:
                    group_cfg["lut"] = str(scp["lut"])
                    if "lut_val" in scp or "base_val" in scp:
                        group_cfg["base_val"] = round(float(scp.get("lut_val", scp.get("base_val", group_cfg["base_val"]))), 3)
                break
        group_primary[gkey] = (loc_anchor, group_cfg)

    # Second pass: compute per-cluster scene_anchor & scene_cfg (sharing the location group's Primary LUT)
    loc_blend_mid = 0.30 if project_model in ("grok", "omni", "hybrid") else 0.15
    loc_blend_rb = 0.35 if project_model in ("grok", "omni", "hybrid") else 0.22

    for c in clusters:
        loc_anchor, group_cfg = group_primary[c["_group_key"]]
        own = c["anchor"]
        scene_anchor = dict(own)
        # Only blend mid/rb toward loc_anchor if they are in the same general lighting ballpark (not a deliberate light switch)
        if abs(own["mid"] - loc_anchor["mid"]) <= 0.085:
            scene_anchor["mid"] = round((1.0 - loc_blend_mid) * own["mid"] + loc_blend_mid * loc_anchor["mid"], 4)
        if abs(own["rb"] - loc_anchor["rb"]) <= 0.065:
            scene_anchor["rb"] = round((1.0 - loc_blend_rb) * own["rb"] + loc_blend_rb * loc_anchor["rb"], 4)

        prog = 0.5 * (c["start_seg"] + c["end_seg"]) / total_segs
        cfg = select_scene_primary_lut_and_targets(
            scene_anchor, c.get("regime", "DAY_CRISP"), project_model, emotion, prog,
            optical_db=optical_db, allow_bw=allow_bw, allow_stylized=allow_stylized,
            is_flashback=c.get("is_flashback", False), bg_code=c["bg_code"],
            seedance_ref=seedance_ref, used_primary_luts=None,
        )
        # Lock Primary LUT to the location group's Primary LUT so same-location scenes share identical film stock
        cfg["lut"] = group_cfg["lut"]
        cfg["base_val"] = group_cfg["base_val"]
        lut_entry = next((e for e in (optical_db or []) if e["title"] == cfg["lut"]), None)
        if lut_entry:
            cfg["sim_anchor"] = simulate_video_through_lut(scene_anchor, lut_entry, cfg["base_val"])

        # Apply AI Colorist scene-specific plan overrides (if any)
        sc_p = _get_sc_plan(c)
        _apply_ai_scene_plan_override(cfg, scene_anchor, sc_p, global_plan, optical_db)

        c["unified_anchor"] = scene_anchor
        c["scene_cfg"] = cfg


def _remaining_scene_defect(raw_val: float, sim_anchor_val: float, target_val: float, deadband: float = 0.010) -> float:
    """
    Returns the un-corrected scene defect `sim_anchor_val - target_val` ONLY when the raw scene anchor
    deviated from `target_val` beyond `deadband` in the same direction.
    When `abs(raw_val - target_val) <= deadband`, the scene anchor was already healthy and any shift in
    `sim_anchor_val` is the Primary 35mm LUT's intentional creative look — so returns 0.0 (never cancels the LUT!).
    """
    raw_err = raw_val - target_val
    if abs(raw_err) <= deadband:
        return 0.0
    sim_err = sim_anchor_val - target_val
    if raw_err * sim_err <= 0:
        return 0.0
    return sim_err if abs(sim_err) < abs(raw_err) else raw_err


def _deadband(val: float, db: float) -> float:
    if abs(val) <= db:
        return 0.0
    return val - (db if val > 0 else -db)


def compute_shot_lut_and_adjustments(
    m: dict,
    scene_cfg: dict,
    scene_anchor: dict,
    local_neighbor_m: dict,
    force_neighbor_match: bool = False,
    optical_db: list[dict] = None,
    seg_scale: float = 1.0,
    used_shot_luts: dict = None,
    clip_model: str = "",
    seg_plan: dict = None,
) -> tuple[str, float, list, str]:
    """
    Scene-Locked Two-Stage Controller:
    - Stage 1 (zorder 2000): Locks 100% of segments in a Scene Cluster to the exact same Scene Primary LUT
      (`scene_cfg["lut"]`) so camera-angle cuts inside a scene never swap `.cube` 3D color matrices.
    - Stage 2 (zorder 8004): Computes Adjust = `SceneBaseline(model, anchor, target) + AchromaticShotContinuity`.
      Drives intra-scene `temperature` & `tone` strictly from achromatic `neutral_rb` / `neutral_gm` (with a 0.015
      deadband) so background/costume color differences (e.g. red brick wall vs. white suit) never cause white-balance
      swings between adjacent cuts, while smoothing genuine exposure/lighting jumps by ~65%.
    """
    seg_plan = seg_plan or {}
    p5 = m.get("p5", 0.05)
    p25 = m.get("p25", 0.15)
    mid = m.get("mid", 0.28)
    p95 = m.get("p95", 0.62)
    p99 = m.get("p99", 0.76)
    c = m.get("contrast", 0.55)
    rb = m.get("rb", 0.13)
    gm = m.get("gm", -0.012)
    sat = m.get("sat", 0.15)
    neutral_rb = m.get("neutral_rb", rb * 0.45)
    neutral_gm = m.get("neutral_gm", gm)
    skin_rb = m.get("skin_rb", rb + 0.06)
    skin_sat = m.get("skin_sat", 0.42)
    sharp = m.get("sharp", 0.035)
    hf_noise = m.get("hf_noise", 0.030)

    eff_model = clip_model or scene_cfg.get("project_model", "seedance")
    regime = scene_cfg.get("regime", "DAY_CRISP")
    target_p5 = scene_cfg.get("target_p5", 0.048)
    target_mid = scene_cfg["target_mid"]
    target_hi = scene_cfg.get("target_hi", 0.650)
    target_rb = scene_cfg["target_rb"]
    target_gm = scene_cfg["target_gm"]
    target_sat = scene_cfg["target_sat"]
    sim_anchor = scene_cfg.get("sim_anchor") or {
        "sim_p5": scene_anchor.get("p5", 0.05),
        "sim_mid": scene_anchor.get("mid", 0.30),
        "sim_hi": scene_anchor.get("p95", 0.65),
        "sim_contrast": scene_anchor.get("contrast", 0.58),
        "sim_rb": scene_anchor.get("rb", 0.12),
        "sim_gm": scene_anchor.get("gm", -0.010),
        "sim_sat": scene_anchor.get("sat", 0.14),
    }

    # Measure intra-scene shot jump relative to scene_anchor
    d_mid_anchor = mid - scene_anchor.get("mid", mid)
    anc_nrb = scene_anchor.get("neutral_rb", scene_anchor.get("rb", rb) * 0.45)
    anc_ngm = scene_anchor.get("neutral_gm", scene_anchor.get("gm", gm))
    d_nrb_anchor = neutral_rb - anc_nrb
    d_ngm_anchor = neutral_gm - anc_ngm

    # Stage 1 (zorder 2000): Lock 100% of shots in the scene to Scene Primary LUT (unless overridden in seg_plan)
    lut_title = scene_cfg["lut"]
    base_val = scene_cfg["base_val"]
    prof_note = scene_cfg["profile"]

    if seg_plan.get("lut"):
        if isinstance(seg_plan["lut"], (list, tuple)) and len(seg_plan["lut"]) == 2:
            lut_title, base_val = str(seg_plan["lut"][0]), float(seg_plan["lut"][1])
        else:
            lut_title = str(seg_plan["lut"])
            if "lut_val" in seg_plan:
                base_val = float(seg_plan["lut_val"])
        prof_note = f"{scene_cfg['profile']} [AI Shot Override]"
    elif abs(d_nrb_anchor) > 0.020 or abs(d_mid_anchor) > 0.045:
        prof_note = f"{scene_cfg['profile']} [Angle-Jump Continuity Dampened]"

    if used_shot_luts is not None:
        used_shot_luts[lut_title] = used_shot_luts.get(lut_title, 0) + 1

    lut_val = round(base_val + clamp(-d_mid_anchor * 0.04, -0.010, 0.010), 3)
    if "lut_val" in seg_plan and not isinstance(seg_plan.get("lut"), (list, tuple)):
        lut_val = round(float(seg_plan["lut_val"]), 3)

    lut_entry = next((e for e in (optical_db or []) if e["title"] == lut_title), None)
    if lut_entry:
        sim = simulate_video_through_lut(m, lut_entry, lut_val)
    else:
        sim = {"sim_p5": p5, "sim_mid": mid, "sim_hi": p95, "sim_contrast": c, "sim_rb": rb, "sim_gm": gm, "sim_sat": sat}

    # Decompose Stage 2 error into:
    #   (A) Un-corrected Scene Defect (0 when anchor was already healthy -> preserves LUT look!)
    #   (B) Intra-Scene Shot Continuity Error (with deadbands so composition changes never cause color/exposure swings!)
    preserve_warm = bool(scene_cfg.get("preserve_warm_props", False)) or (anc_nrb <= 0.082)
    scene_def_p5 = _remaining_scene_defect(scene_anchor.get("p5", p5), sim_anchor["sim_p5"], target_p5, deadband=0.008)
    scene_def_mid = _remaining_scene_defect(scene_anchor.get("mid", mid), sim_anchor["sim_mid"], target_mid, deadband=0.010)
    scene_def_hi = _remaining_scene_defect(scene_anchor.get("p95", p95), sim_anchor["sim_hi"], target_hi, deadband=0.015)
    scene_def_rb = _remaining_scene_defect(scene_anchor.get("rb", rb), sim_anchor["sim_rb"], target_rb, deadband=0.010)
    if preserve_warm:
        scene_def_rb *= 0.30
    scene_def_gm = _remaining_scene_defect(scene_anchor.get("gm", gm), sim_anchor["sim_gm"], target_gm, deadband=0.005)
    scene_def_sat = _remaining_scene_defect(scene_anchor.get("sat", sat), sim_anchor["sim_sat"], target_sat, deadband=0.008)

    shot_gain = 1.15 if (force_neighbor_match or seg_plan.get("force_anchor_match")) else 1.0
    shot_err_p5 = _deadband(sim["sim_p5"] - sim_anchor["sim_p5"], 0.010) * shot_gain
    shot_err_mid = _deadband(sim["sim_mid"] - sim_anchor["sim_mid"], 0.015) * shot_gain
    shot_err_hi = _deadband(sim["sim_hi"] - sim_anchor["sim_hi"], 0.018) * shot_gain
    shot_err_nrb = _deadband(d_nrb_anchor, 0.015) * shot_gain
    shot_err_ngm = _deadband(d_ngm_anchor, 0.008) * shot_gain
    shot_err_sat = _deadband(sim["sim_sat"] - sim_anchor["sim_sat"], 0.012) * shot_gain

    nb_mid_diff = _deadband(mid - local_neighbor_m.get("mid", mid), 0.018)
    nb_nrb_diff = _deadband(neutral_rb - local_neighbor_m.get("neutral_rb", anc_nrb), 0.015)
    nb_sat_diff = _deadband(sat - local_neighbor_m.get("sat", sat), 0.015)
    nb_weight = 0.22 if force_neighbor_match else 0.15

    adjust = []

    # 1. SHADOW (`shadow`, v3) — Open dark night shadows (`+0.02..+0.15`), anchor milky blacks (`-0.09..-0.01`)
    if scene_def_p5 > 0:
        raw_shd = -scene_def_p5 * 1.25 - shot_err_p5 * 0.65
    else:
        raw_shd = -scene_def_p5 * 2.10 - shot_err_p5 * 0.65 + max(0.0, 0.145 - p25) * 0.35
    raw_shd += float(scene_cfg.get("shadow_bias", 0.0)) + float(seg_plan.get("shadow_offset", 0.0))
    shadow_val = non_zero_clamp(raw_shd, -0.09, 0.15, min_abs=0.01, default_sign=-1.0 if p5 > 0.085 else 1.0)
    adjust.append(("shadow", shadow_val))

    # 2. HIGHLIGHT (`highlight`, v3) — Tame Grok specular glare (`-0.04..-0.12`), keep Seedance highlights luminous
    grok_glare_pull = -0.035 if (eff_model == "grok" and p95 > 0.66) else 0.0
    raw_hl = -scene_def_hi * 0.65 - shot_err_hi * 0.45 - max(0.0, p99 - 0.850) * 0.25 + grok_glare_pull
    raw_hl += float(scene_cfg.get("highlight_bias", 0.0)) + float(seg_plan.get("highlight_offset", 0.0))
    highlight_val = non_zero_clamp(raw_hl, -0.12, 0.08, min_abs=0.01, default_sign=-1.0 if p95 > 0.66 else 1.0)
    adjust.append(("highlight", highlight_val))

    # 3. BRIGHTNESS (`brightness`, v2) & LIGHT SENSATION (`light_sensation`, "")
    raw_br = (
        -scene_def_mid * 0.85
        - shot_err_mid * 0.50
        - nb_mid_diff * nb_weight
        + float(scene_cfg.get("bright_bias", 0.0))
        + float(seg_plan.get("bright_offset", 0.0))
    )
    bright_val = non_zero_clamp(raw_br, -0.11, 0.16, min_abs=0.01, default_sign=1.0 if mid < 0.34 else -1.0)
    adjust.append(("brightness", bright_val))

    base_ls = -0.015 if (eff_model == "grok" and c > 0.58) else (+0.012 if eff_model == "seedance" and mid < 0.42 else 0.0)
    raw_ls = base_ls - scene_def_mid * 0.25 - shot_err_mid * 0.15
    light_sens_val = non_zero_clamp(raw_ls, -0.04, 0.05, min_abs=0.01, default_sign=1.0 if base_ls >= 0 else -1.0)
    adjust.append(("light_sensation", light_sens_val))

    # 4. CONTRAST (`contrast`, v3) — Soften harsh Grok glare (`-0.02..-0.05`), add crisp 35mm depth to Seedance/Omni (`+0.015..+0.04`)
    if eff_model == "grok":
        base_con = -0.022 if c > 0.57 else +0.012
    else:
        base_con = +0.022 if c < 0.62 else -0.015
    target_c = target_hi - target_p5
    raw_con = base_con + (target_c - sim_anchor["sim_contrast"]) * 0.18 + float(scene_cfg.get("contrast_bias", 0.0)) + float(seg_plan.get("contrast_offset", 0.0))
    contrast_val = non_zero_clamp(raw_con, -0.06, 0.05, min_abs=0.01, default_sign=1.0 if base_con >= 0 else -1.0)
    adjust.append(("contrast", contrast_val))

    # 5. TONE / TINT (`tone`, v3) — Positive shifts toward magenta (canceling Omni/Grok green cast `gm > 0`)
    raw_tone = (scene_def_gm + shot_err_ngm) * 1.85 + float(scene_cfg.get("tone_bias", 0.0)) + float(seg_plan.get("tone_offset", 0.0))
    tone_val = non_zero_clamp(raw_tone, -0.05, 0.06, min_abs=0.01, default_sign=1.0 if gm >= -0.008 else -1.0)
    adjust.append(("tone", tone_val))

    # 6. TEMPERATURE (`temperature`, v3) — Driven by achromatic `neutral_rb` so red brick walls / blue shirts never trigger false temperature swings!
    plastic_thresh = (0.248 if preserve_warm else 0.232) if eff_model == "grok" else 0.262
    shot_plastic_excess = max(0.0, (skin_rb - plastic_thresh) * 0.30) if neutral_rb > 0.075 else 0.0
    raw_temp = (
        -scene_def_rb * 1.15
        - shot_err_nrb * 1.10
        - nb_nrb_diff * nb_weight
        - shot_plastic_excess
        + float(scene_cfg.get("temp_bias", 0.0))
        + float(seg_plan.get("temp_offset", 0.0))
    )
    default_temp_sign = 1.0 if regime in ("DAWN_DUSK_GOLD", "WARM_INT") and scene_def_rb <= 0 else -1.0
    temp_val = non_zero_clamp(raw_temp, -0.14, 0.14, min_abs=0.01, default_sign=default_temp_sign)
    adjust.append(("temperature", temp_val))

    # 7. SATURATION (`saturation`, v1) — Clean 35mm color density; tames Grok oversaturated orange skin (`skin_sat > 0.48`)
    shot_sat_excess = max(0.0, (skin_sat - 0.485) * 0.22) if eff_model == "grok" else max(0.0, (skin_sat - 0.540) * 0.20)
    raw_sat = (
        -scene_def_sat * 1.05
        - shot_err_sat * 0.55
        - nb_sat_diff * nb_weight
        - shot_sat_excess
        + float(scene_cfg.get("sat_bias", 0.0))
        + float(seg_plan.get("sat_offset", 0.0))
    )
    default_sat_sign = -1.0 if (eff_model == "grok" or scene_def_sat > 0) else 1.0
    sat_val = non_zero_clamp(raw_sat, -0.09, 0.08, min_abs=0.01, default_sign=default_sat_sign)
    adjust.append(("saturation", sat_val))

    # 8. MODEL-AWARE 35MM OPTICAL SHARPNESS (`sharpen`, `clear`, `particle`) + ZOOMED SUPER-RES BOOST
    zoom_excess = max(0.0, float(seg_scale) - 1.0)
    if eff_model == "grok" and sharp > 0.042:
        # Grok footage already has crisp digital edges; keep sharpen gentle to avoid harsh digital haloes
        sh_val = round(clamp(0.045 + zoom_excess * 0.14, 0.040, 0.140), 3)
        cl_val = round(clamp(0.012 + zoom_excess * 0.030, 0.010, 0.035), 3)
        pt_val = round(clamp(0.028 + max(0.0, 0.035 - hf_noise) * 1.4, 0.022, 0.040), 3)
    else:
        # Seedance & Omni footage benefits from crisp 35mm optical micro-contrast + Super-Res on 150% zoomed cuts
        sh_val = round(clamp(0.072 + max(0.0, 0.038 - sharp) * 2.2 + zoom_excess * 0.16, 0.055, 0.180), 3)
        cl_val = round(clamp(0.018 + max(0.0, 0.038 - sharp) * 0.6 + zoom_excess * 0.035, 0.014, 0.042), 3)
        pt_val = round(clamp(0.022 + max(0.0, 0.030 - hf_noise) * 1.2, 0.018, 0.036), 3)

    adjust.append(("sharpen", sh_val))
    adjust.append(("clear", cl_val))
    adjust.append(("particle", pt_val))

    if scene_cfg.get("use_vignette"):
        adjust.append(("vignetting", 0.025))

    return lut_title, lut_val, adjust, prof_note


def _load_cube_grid(lut_path: str):
    if lut_path in _CUBE_GRID_CACHE:
        return _CUBE_GRID_CACHE[lut_path]
    if not lut_path or not os.path.exists(lut_path) or not lut_path.lower().endswith(".cube"):
        _CUBE_GRID_CACHE[lut_path] = None
        return None
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
        grid = np.array(rows, dtype=np.float32).reshape((size, size, size, 3))
        _CUBE_GRID_CACHE[lut_path] = grid
        return grid
    _CUBE_GRID_CACHE[lut_path] = None
    return None


def apply_3d_lut_and_adjust_to_rgb(rgb: np.ndarray, lut_path: str, lut_val: float, adj_list: list) -> np.ndarray:
    """
    Applies the `.cube` 3D LUT using vectorized 3D trilinear interpolation (with `tt_skin_seg` skin protection)
    followed by the 12 Adjust sliders onto a float32 RGB image `[0..1]`.
    """
    out = np.clip(rgb.copy(), 0.0, 1.0)
    lut_grid = _load_cube_grid(lut_path)
    if lut_grid is not None:
        size = lut_grid.shape[0]
        scaled = out * (size - 1)
        r0 = np.clip(np.floor(scaled[..., 0]).astype(np.int32), 0, size - 2)
        g0 = np.clip(np.floor(scaled[..., 1]).astype(np.int32), 0, size - 2)
        b0 = np.clip(np.floor(scaled[..., 2]).astype(np.int32), 0, size - 2)
        dr = (scaled[..., 0] - r0)[..., None]
        dg = (scaled[..., 1] - g0)[..., None]
        db = (scaled[..., 2] - b0)[..., None]

        c000 = lut_grid[b0, g0, r0]
        c100 = lut_grid[b0, g0, r0 + 1]
        c010 = lut_grid[b0, g0 + 1, r0]
        c110 = lut_grid[b0, g0 + 1, r0 + 1]
        c001 = lut_grid[b0 + 1, g0, r0]
        c101 = lut_grid[b0 + 1, g0, r0 + 1]
        c011 = lut_grid[b0 + 1, g0 + 1, r0]
        c111 = lut_grid[b0 + 1, g0 + 1, r0 + 1]

        c00 = c000 * (1.0 - dr) + c100 * dr
        c01 = c001 * (1.0 - dr) + c101 * dr
        c10 = c010 * (1.0 - dr) + c110 * dr
        c11 = c011 * (1.0 - dr) + c111 * dr
        c0 = c00 * (1.0 - dg) + c10 * dg
        c1 = c01 * (1.0 - dg) + c11 * dg
        mapped = c0 * (1.0 - db) + c1 * db

        r, g, b = out[..., 0], out[..., 1], out[..., 2]
        skin_mask = ((r > g) & (g > b) & ((r - b) > 0.06) & (r > 0.18) & (r < 0.90)).astype(np.float32)[..., None]
        eff_alpha = lut_val * (1.0 - 0.50 * skin_mask)
        out = (1.0 - eff_alpha) * out + eff_alpha * mapped

    adj_dict = dict(adj_list)
    br = adj_dict.get("brightness", 0.0)
    ls = adj_dict.get("light_sensation", 0.0)
    con = adj_dict.get("contrast", 0.0)
    shd = adj_dict.get("shadow", 0.0)
    hl = adj_dict.get("highlight", 0.0)
    temp = adj_dict.get("temperature", 0.0)
    tone = adj_dict.get("tone", 0.0)
    sat = adj_dict.get("saturation", 0.0)

    luma = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
    shd_w = np.clip(1.0 - luma * 2.1, 0.0, 1.0)[..., None]
    mid_w = (4.0 * luma * (1.0 - luma))[..., None]
    hl_w = np.clip((luma - 0.45) * 1.9, 0.0, 1.0)[..., None]

    out = out + (br * 0.48 + ls * 0.35) * (0.35 + 0.65 * mid_w) + shd * 0.36 * shd_w + hl * 0.36 * hl_w
    out = (out - 0.33) * (1.0 + con * 0.95) + 0.33
    out[..., 0] += temp * 0.18 + tone * 0.06
    out[..., 2] -= temp * 0.18 - tone * 0.06
    out[..., 1] -= tone * 0.09
    luma2 = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2])[..., None]
    out = luma2 + (out - luma2) * (1.0 + sat * 1.15)
    return np.clip(out, 0.0, 1.0)


def _measure_rgb_mid_rb(rgb: np.ndarray) -> tuple[float, float, float]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    rb_map = r - b
    rb = float(np.mean(rb_map))
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    valid = (luma >= 0.06) & (luma <= 0.90)
    th = max(0.065, float(np.percentile(chroma[valid], 30)) if np.any(valid) else 0.08)
    nmask = valid & (chroma <= th)
    nrb = float(np.mean(rb_map[nmask])) if np.sum(nmask) >= 16 else rb * 0.45
    return float(np.median(luma)), rb, nrb


def render_graded_comparison_sheet(
    all_seg_tuples: list,
    seg_records: list,
    clusters: list,
    lut_map: dict,
    out_jpg: str,
    proj_dir: str = "",
    optical_db: list[dict] = None,
    seg_applied: dict = None,
):
    """
    Renders a Multi-Shot Before/After Contact Sheet where each Scene Cluster is a 4-tile strip:
      [1. Anchor RAW] -> [2. Anchor GRADED]  |  [3. Outlier RAW] -> [4. Outlier GRADED]
    allowing the AI Colorist to visually inspect both the primary 35mm scene grade and the
    intra-scene outlier shot convergence.
    """
    from PIL import Image, ImageDraw
    seg_applied = seg_applied or {}
    w, h, pad_top, pad_bot = 216, 122, 26, 20
    strip_w = w * 4 + 14
    tile_h = h + pad_top + pad_bot
    cols = 1 if len(clusters) <= 14 else 2
    rows = max(1, (len(clusters) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * strip_w + 16, rows * tile_h + 12), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)

    def _grade_seg(s_idx: int, c: dict):
        _, _, v, _ = all_seg_tuples[s_idx]
        rec = seg_records[s_idx]
        cfg = c["scene_cfg"]
        scene_anchor = c.get("unified_anchor") or c["anchor"]
        ts = rec["start_sec"] + 0.5 * rec["dur_sec"]
        vp = resolve_video_path(v.get("path", ""), proj_dir)
        raw_rgb = extract_rgb_frame(vp, ts, w, h)
        if s_idx in seg_applied:
            lut_title, lut_val, adj_list, _ = seg_applied[s_idx]
        else:
            lut_title, lut_val, adj_list, _ = compute_shot_lut_and_adjustments(
                rec.get("metrics") or scene_anchor, cfg, scene_anchor, scene_anchor, False,
                optical_db=optical_db, seg_scale=rec.get("scale", 1.0),
                clip_model=rec.get("clip_model", cfg.get("project_model", "seedance")),
            )
        graded_rgb = apply_3d_lut_and_adjust_to_rgb(raw_rgb, lut_map.get(lut_title, ""), lut_val, adj_list)
        r_mid, r_rb, r_nrb = _measure_rgb_mid_rb(raw_rgb)
        g_mid, g_rb, g_nrb = _measure_rgb_mid_rb(graded_rgb)
        return s_idx, raw_rgb, graded_rgb, lut_title, lut_val, r_mid, r_rb, r_nrb, g_mid, g_rb, g_nrb

    def _render_strip(idx_c):
        idx, c = idx_c
        s_indices = c["seg_indices"]
        anc_seg = c.get("anchor_seg", s_indices[len(s_indices) // 2])
        out_seg = c.get("outlier_seg", s_indices[-1])
        anc_res = _grade_seg(anc_seg, c)
        out_res = _grade_seg(out_seg, c) if out_seg != anc_seg else anc_res
        return idx, c, anc_res, out_res

    with ThreadPoolExecutor(max_workers=12) as pool:
        rendered = list(pool.map(_render_strip, enumerate(clusters)))

    for idx, c, anc_res, out_res in rendered:
        r_i, c_i = divmod(idx, cols)
        x0 = c_i * strip_w + 8
        y0 = r_i * tile_h + 6
        draw.rectangle([x0 - 2, y0, x0 + strip_w - 6, y0 + tile_h - 4], fill=(24, 24, 28), outline=(56, 62, 70))

        a_idx, a_raw, a_grd, a_lut, a_val, ar_m, ar_rb, ar_nrb, ag_m, ag_rb, ag_nrb = anc_res
        o_idx, o_raw, o_grd, o_lut, o_val, or_m, or_rb, or_nrb, og_m, og_rb, og_nrb = out_res

        raw_d_mid = abs(or_m - ar_m)
        raw_d_nrb = abs(or_nrb - ar_nrb)
        grd_d_mid = abs(og_m - ag_m)
        grd_d_nrb = abs(og_nrb - ag_nrb)

        reg_lbl = c.get("regime", "")[:12]
        note_str = f" [{c['scene_cfg'].get('ai_note','')[:18]}]" if c["scene_cfg"].get("ai_note") else ""
        hdr = (
            f"S{c['cluster_id']:02d} ({c['bg_code'][:15]}|{reg_lbl}) -> {a_lut[:18]} ({int(a_val*100)}%){note_str} | "
            f"Spread dMid:{raw_d_mid:.2f}->{grd_d_mid:.2f} dNRB:{raw_d_nrb:.2f}->{grd_d_nrb:.2f}"
        )
        draw.text((x0 + 4, y0 + 5), hdr, fill=(160, 245, 175))

        tiles = [
            (a_raw, f"1.Anc RAW #{a_idx} M:{ar_m:.2f} NRB:{ar_nrb:+.2f}", (210, 210, 215)),
            (a_grd, f"2.Anc GRD #{a_idx} M:{ag_m:.2f} NRB:{ag_nrb:+.2f}", (155, 245, 175)),
            (o_raw, f"3.Out RAW #{o_idx} M:{or_m:.2f} NRB:{or_nrb:+.2f}", (245, 205, 155)),
            (o_grd, f"4.Out GRD #{o_idx} M:{og_m:.2f} NRB:{og_nrb:+.2f} ({o_lut[:10]})", (155, 235, 250)),
        ]
        for t_i, (arr, sub_lbl, col_txt) in enumerate(tiles):
            tx = x0 + t_i * (w + 3)
            ty = y0 + pad_top
            sheet.paste(Image.fromarray((arr * 255).astype(np.uint8)), (tx, ty))
            draw.text((tx + 3, ty + h + 3), sub_lbl, fill=col_txt)

    sheet.save(out_jpg, quality=88)
    page_paths = [out_jpg]
    if len(clusters) > 16:
        page_rows = 7
        page_h = page_rows * tile_h + 12
        total_pages = (rows + page_rows - 1) // page_rows
        base_no_ext = os.path.splitext(out_jpg)[0]
        page_paths = []
        for p_idx in range(total_pages):
            y_start = p_idx * page_rows * tile_h
            y_end = min(sheet.height, y_start + page_h)
            crop_img = sheet.crop((0, y_start, sheet.width, y_end))
            p_path = f"{base_no_ext}_p{p_idx + 1}.jpg"
            crop_img.save(p_path, quality=90)
            page_paths.append(p_path)
    return page_paths


MINI_DRAFT_ADJUST_SLOT_MAP = {
    "lut": "lut",
    "brightness": "brightness",
    "contrast": "contrast",
    "saturation": "saturation",
    "sharpen": "sharpening",
    "highlight": "highlight",
    "shadow": "shadow",
    "clear": "clear",
    "temperature": "color_temperature",
    "tone": "hue",
    "light_sensation": "light_sensation",
    "particle": "particle",
    "vignetting": "vignetting",
    "fade": "fade",
}


def make_mini_draft_effect(eff: dict) -> dict:
    return {
        "adjust_params": None,
        "algorithm_artifact_path": eff.get("algorithm_artifact_path", ""),
        "beauty_body_auto_preset_id": eff.get("beauty_body_auto_preset_id", ""),
        "beauty_face_auto_preset_id": eff.get("beauty_face_auto_preset_id", ""),
        "beauty_face_auto_retouch_info": {
            "beauty_face_auto_retouch_id": "",
            "face_id": [],
            "id": str(uuid.uuid4()).upper(),
        },
        "bloom_params": eff.get("bloom_params", None),
        "category_id": eff.get("category_id", ""),
        "category_key": eff.get("category_key", ""),
        "category_name": eff.get("category_name", ""),
        "color_match_info": {
            "id": str(uuid.uuid4()).upper(),
            "source_feature_path": "",
            "target_feature_path": "",
            "target_image_path": "",
        },
        "covering_relation_change": eff.get("covering_relation_change", 0),
        "effect_id": eff.get("effect_id", "7501974767453474064"),
        "enable_skin_tone_correction": bool(eff.get("enable_skin_tone_correction", False)),
        "exclusion_group": eff.get("exclusion_group", []),
        "face_adjust_params": None,
        "formula_id": eff.get("formula_id", ""),
        "id": eff["id"],
        "intensity_key": eff.get("intensity_key", ""),
        "is_from_intelligent_quality": bool(eff.get("is_from_intelligent_quality", False)),
        "item_effect_type": eff.get("item_effect_type", 0),
        "lumi_hub_path": eff.get("lumi_hub_path", ""),
        "name": eff.get("name", ""),
        "panel_id": eff.get("panel_id", ""),
        "path": eff.get("path", ""),
        "platform": eff.get("platform", "all"),
        "report_name": eff.get("report_name", ""),
        "request_id": eff.get("request_id", ""),
        "resource_id": eff.get("resource_id", "7501974767453474064"),
        "smart_color_mode": eff.get("smart_color_mode", 0),
        "smart_color_mode_cache": [],
        "source_platform": eff.get("source_platform", 1),
        "sub_category_id": eff.get("sub_category_id", ""),
        "sub_category_name": eff.get("sub_category_name", ""),
        "sub_type": 0,
        "third_resource_id": eff.get("third_resource_id", ""),
        "time_range": eff.get("time_range", None),
        "transparent_params": "",
        "type": eff["type"],
        "value": eff["value"],
        "version": eff.get("version", ""),
        "visible": bool(eff.get("visible", True)),
    }


def build_mini_draft_adjust(seg: dict, effects_by_id: dict, existing_adjust: dict = None) -> dict:
    base = {
        "adjust_mask": existing_adjust.get("adjust_mask") if isinstance(existing_adjust, dict) else None,
        "adjust_preset_resource_id": "",
        "backlight_correction": None,
        "black": None,
        "brightness": None,
        "clear": None,
        "color_adjust_pro": None,
        "color_correct": None,
        "color_correct_alg_result": "",
        "color_curves": existing_adjust.get("color_curves") if isinstance(existing_adjust, dict) else None,
        "color_dehaze": None,
        "color_levels": None,
        "color_match": None,
        "color_soften": None,
        "color_temperature": None,
        "contrast": None,
        "enable_adjust": bool(seg.get("enable_adjust", False)),
        "enable_adjust_mask": bool(existing_adjust.get("enable_adjust_mask", False)) if isinstance(existing_adjust, dict) else False,
        "enable_color_adjust_pro": False,
        "enable_color_correct_adjust": False,
        "enable_color_curves": True,
        "enable_color_match_adjust": False,
        "enable_color_wheels": True,
        "enable_hsl": bool(seg.get("enable_hsl", False)),
        "enable_hsl_curves": True,
        "enable_lut": bool(seg.get("enable_lut", False)),
        "enable_smart_color_adjust": False,
        "fade": None,
        "filter": None,
        "highlight": None,
        "hsl": existing_adjust.get("hsl") if isinstance(existing_adjust, dict) else None,
        "hsl_curves": existing_adjust.get("hsl_curves") if isinstance(existing_adjust, dict) else None,
        "hue": None,
        "id": existing_adjust.get("id") if (isinstance(existing_adjust, dict) and existing_adjust.get("id")) else str(uuid.uuid4()).upper(),
        "light_sensation": None,
        "log_color_wheels": existing_adjust.get("log_color_wheels") if isinstance(existing_adjust, dict) else None,
        "lut": None,
        "naturalSaturation": None,
        "particle": None,
        "primary_color_wheels": existing_adjust.get("primary_color_wheels") if isinstance(existing_adjust, dict) else None,
        "restore_detail": None,
        "saturation": None,
        "shadow": None,
        "sharpening": None,
        "skin_tone_correction": None,
        "smart_adjust_color": None,
        "type": "adjust",
        "vignetting": None,
        "white": None,
    }
    for ref_id in seg.get("extra_material_refs", []):
        eff = effects_by_id.get(ref_id)
        if not eff:
            continue
        slot = MINI_DRAFT_ADJUST_SLOT_MAP.get(eff.get("type", ""))
        if slot:
            base[slot] = make_mini_draft_effect(eff)
    return base


def sync_mini_draft_files(proj_dir: str, draft_containers: list, backup_dir: str):
    seg_info_map = {}
    for d_obj in draft_containers:
        eff_by_id = {e["id"]: e for e in d_obj.get("materials", {}).get("effects", []) if "id" in e}
        for trk in d_obj.get("tracks", []):
            if trk.get("type") == "video":
                for seg in trk.get("segments", []):
                    sid = seg.get("id")
                    if sid:
                        seg_info_map[sid] = (seg, eff_by_id)

    for tl_dir in glob.glob(os.path.join(proj_dir, "Timelines", "*")):
        if not os.path.isdir(tl_dir):
            continue
        patch_dir = os.path.join(tl_dir, "attachment", "patch")
        mini_path = os.path.join(patch_dir, "mini_draft.json")
        patch_path = os.path.join(patch_dir, "patch.json")
        if os.path.exists(mini_path):
            shutil.copy2(mini_path, os.path.join(backup_dir, f"mini_draft_{os.path.basename(tl_dir)}.json"))
            try:
                with open(mini_path, "r", encoding="utf-8") as mf:
                    md_obj = json.load(mf)
                mdd = md_obj.get("mini_draft_data", {})
                now_s = int(time.time())
                if isinstance(mdd.get("header"), dict):
                    mdd["header"]["timestamp"] = now_s
                    mdd["header"]["timestamp_ms"] = now_s * 1000
                for s in mdd.get("segments", []) or []:
                    sid = s.get("id")
                    if sid in seg_info_map and s.get("type") == "segment_video":
                        seg_d, eff_map = seg_info_map[sid]
                        s["adjust"] = build_mini_draft_adjust(seg_d, eff_map, s.get("adjust"))
                for md_entry in mdd.get("material_drafts", []) or []:
                    for s in md_entry.get("segments", []) or []:
                        sid = s.get("id")
                        if sid in seg_info_map and s.get("type") == "segment_video":
                            seg_d, eff_map = seg_info_map[sid]
                            s["adjust"] = build_mini_draft_adjust(seg_d, eff_map, s.get("adjust"))
                with open(mini_path, "w", encoding="utf-8") as mf:
                    json.dump(md_obj, mf, ensure_ascii=False, separators=(",", ":"))
            except Exception as exc:
                print(f"Warning: could not sync mini_draft.json ({exc})")
        if os.path.exists(patch_path):
            with open(patch_path, "w", encoding="utf-8") as pf:
                pf.write('{"patch_data":[]}\n')


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Apply model-aware, script-adaptive 35mm grading to a CapCut project.")
    parser.add_argument("project", help="CapCut Project ID or path.")
    parser.add_argument("--metrics", default="", help="Optional precomputed diagnostics JSON path from analyze_project.py.")
    parser.add_argument(
        "--model",
        choices=["auto", "seedance", "grok", "omni", "hybrid"],
        default="auto",
        help="AI Video Model override (defaults to deterministic source-duration auto-detection).",
    )
    parser.add_argument(
        "--emotion",
        choices=["narrative-arc", "warm-human", "dramatic-contrast", "luxury-crisp", "vintage-35mm"],
        default="narrative-arc",
        help="Emotional grading direction.",
    )
    parser.add_argument("--plan", default="", help="Optional custom AI grading plan JSON path.")
    parser.add_argument("--match-clips", default="", help="Optional comma-separated clips to strongly match to immediate neighbors.")
    parser.add_argument("--auto-match-outliers", action="store_true", help="Automatically boost neighbor-matching on intra-scene outlier segments.")
    parser.add_argument("--allow-stylized", action="store_true", help="Unlock Tier 2 (stylized_rec709) with CapCut tt_skin_seg Skin Protection.")
    parser.add_argument("--allow-bw", action="store_true", help="Unlock Tier 3 (bw_mono) for flashback/noir sequences.")
    parser.add_argument("--input-color-space", choices=["rec709", "log"], default="rec709", help="Input color space.")
    parser.add_argument("--relaunch", action="store_true", help="Quit CapCut before writing and relaunch after completion.")
    args = parser.parse_args()

    proj_dir = resolve_project_dir(args.project)
    lut_map = load_lut_map()

    if args.relaunch:
        subprocess.run(["killall", "-TERM", "CapCut"], stderr=subprocess.DEVNULL, check=False)
        time.sleep(0.8)
        subprocess.run(["killall", "-9", "CapCut"], stderr=subprocess.DEVNULL, check=False)
        time.sleep(0.5)

    backup_dir = os.path.join("/tmp", f"capcut_backup_{os.path.basename(proj_dir).replace(' ', '_')}")
    os.makedirs(backup_dir, exist_ok=True)

    root_path = os.path.join(proj_dir, "draft_info.json")
    shutil.copy2(root_path, os.path.join(backup_dir, "draft_info.json"))

    with open(root_path, "r", encoding="utf-8") as f:
        root_draft = json.load(f)

    sub_drafts = root_draft.get("materials", {}).get("drafts", [])
    has_compound = bool(sub_drafts)
    all_seg_tuples = collect_all_video_segments(root_draft)

    # 1. Auto-detect AI Video Model from source video durations (or honor --model override)
    detected_model, clip_model_by_path, model_stats = detect_source_models(all_seg_tuples, proj_dir)
    project_model = detected_model if args.model == "auto" else args.model

    video_paths = list({resolve_video_path(v.get("path", ""), proj_dir) for _, _, v, _ in all_seg_tuples if v.get("path")})
    clip_nums = [extract_clip_number(v.get("material_name") or os.path.basename(v.get("path", ""))) for _, _, v, _ in all_seg_tuples]
    expected_max_clip = max([c for c in clip_nums if c > 0] or [0])

    clip_bg_map, gt_path, bg_context = discover_ground_truth_scene_map(
        video_paths, expected_max_clip=expected_max_clip, project_model=project_model
    )

    # 2. Load cached metrics (defaulting to /tmp/capcut_metrics_<slug>.json) or compute in parallel
    metrics_path = args.metrics or get_default_metrics_cache_path(proj_dir)
    seg_records = []
    preloaded_cache = {}
    if metrics_path and os.path.exists(metrics_path):
        try:
            loaded = json.load(open(metrics_path, "r", encoding="utf-8"))
            if isinstance(loaded, dict):
                for r in loaded.get("segments", []):
                    if r.get("metrics") and "skin_sat" in r["metrics"] and "neutral_rb" in r["metrics"]:
                        preloaded_cache[(r.get("material_name", ""), round(float(r.get("start_sec", 0)), 1), round(float(r.get("dur_sec", 0)), 1))] = r["metrics"]
                if len(loaded.get("segments", [])) == len(all_seg_tuples) and all("neutral_rb" in (r.get("metrics") or {}) for r in loaded["segments"]):
                    seg_records = loaded["segments"]
                    for r in seg_records:
                        if "clip_model" not in r:
                            r["clip_model"] = project_model
        except Exception:
            pass

    if not seg_records or len(seg_records) != len(all_seg_tuples):
        tasks = []
        unique_jobs = {}
        for idx, (_, seg, v, container_lbl) in enumerate(all_seg_tuples):
            vpath = resolve_video_path(v.get("path", ""), proj_dir)
            mname = v.get("material_name") or os.path.basename(vpath)
            st_sec = seg.get("source_timerange", {}).get("start", 0) / 1e6
            dur_sec = seg.get("source_timerange", {}).get("duration", 5000000) / 1e6
            src_dur_sec = v.get("duration", 0) / 1e6
            c_model = clip_model_by_path.get(vpath, project_model) if args.model == "auto" else project_model
            sc_obj = seg.get("clip", {}).get("scale", {}) if isinstance(seg.get("clip"), dict) else {}
            seg_scale = max(float(sc_obj.get("x", 1.0)), float(sc_obj.get("y", 1.0)))
            pk = (mname, round(st_sec, 1), round(dur_sec, 1))
            ck = (vpath, round(st_sec, 1), round(dur_sec, 1))
            if pk not in preloaded_cache and ck not in unique_jobs and os.path.exists(vpath):
                unique_jobs[ck] = (vpath, st_sec, dur_sec)
            tasks.append((idx, seg.get("id", ""), container_lbl, mname, st_sec, dur_sec, src_dur_sec, c_model, seg_scale, pk, ck))

        cache_by_range = {}
        if unique_jobs:
            def _run_job(item):
                k, (vp, st, dur) = item
                return k, extract_segment_metrics(vp, st, dur)
            with ThreadPoolExecutor(max_workers=12) as pool:
                for k, res in pool.map(_run_job, unique_jobs.items()):
                    cache_by_range[k] = res

        seg_records = []
        for idx, sid, container_lbl, mname, st_sec, dur_sec, src_dur_sec, c_model, seg_scale, pk, ck in tasks:
            m_val = preloaded_cache.get(pk) or cache_by_range.get(ck, {})
            seg_records.append({
                "seg_idx": idx,
                "seg_id": sid,
                "container": container_lbl,
                "material_name": mname,
                "start_sec": round(st_sec, 2),
                "dur_sec": round(dur_sec, 2),
                "source_dur_sec": round(src_dur_sec, 2),
                "clip_model": c_model,
                "scale": round(seg_scale, 3),
                "metrics": m_val,
            })

    custom_plan = json.load(open(args.plan, "r", encoding="utf-8")) if (args.plan and os.path.exists(args.plan)) else {}

    # 3. Clean old managed effects across root_draft and all sub_drafts
    managed_types = set(VERSION_MAP.keys())
    draft_containers = [root_draft] + [sd["draft"] for sd in sub_drafts if "draft" in sd]
    old_eff_ids = set()
    for d_obj in draft_containers:
        effs = d_obj.get("materials", {}).setdefault("effects", [])
        old_eff_ids.update(e["id"] for e in effs if e.get("type") in managed_types)
        effs[:] = [e for e in effs if e.get("type") not in managed_types]

    compound_ids = {sd["id"] for sd in sub_drafts if "id" in sd}
    for d_obj in draft_containers:
        photo_ids = {v["id"] for v in d_obj.get("materials", {}).get("videos", []) if v.get("type") == "photo" and "id" in v}
        for trk in d_obj.get("tracks", []):
            if trk.get("type") == "video":
                for seg in trk.get("segments", []):
                    seg_refs = [seg.get("material_id", "")] + list(seg.get("extra_material_refs", []))
                    if any(r in compound_ids for r in seg_refs) or seg.get("material_id", "") in photo_ids:
                        seg["extra_material_refs"] = [r for r in seg.get("extra_material_refs", []) if r not in old_eff_ids]
                        seg["enable_lut"] = False
                        seg["enable_adjust"] = False

    # 4. Ground-Truth Scene Clustering + Model-Aware & Script-Adaptive Scene Targets
    optical_db = load_optical_db(lut_map)
    clusters = cluster_timeline_scenes(seg_records, clip_bg_map, bg_context=bg_context)
    assign_unified_scene_targets(
        clusters, project_model, args.emotion, optical_db,
        allow_bw=args.allow_bw, allow_stylized=args.allow_stylized,
        custom_plan=custom_plan,
    )
    seg_to_cluster = {}
    for c in clusters:
        for s_idx in c["seg_indices"]:
            seg_to_cluster[s_idx] = c

    explicit_match = {x.strip() for x in args.match_clips.split(",") if x.strip()}
    plan_segs = custom_plan.get("segments", {}) or {}
    default_m = {
        "p5": 0.05, "p25": 0.15, "mid": 0.28, "p95": 0.62, "p99": 0.76, "contrast": 0.55,
        "rb": 0.13, "gm": -0.012, "sat": 0.15, "hsv_s": 0.45, "neutral_rb": 0.045, "neutral_gm": -0.008,
        "skin_rb": 0.20, "skin_gm": -0.025, "skin_sat": 0.42, "sharp": 0.042, "hf_noise": 0.030,
    }

    summary = []
    seg_applied = {}
    total_adjusts = 0
    angle_jump_fixes = 0
    zoomed_count = 0
    used_shot_luts = {}
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

        scfg = plan_segs.get(str(idx)) or plan_segs.get(mname) or {}
        if scfg and "adjust" in scfg and "lut" in scfg and isinstance(scfg["lut"], (list, tuple)):
            adj_list = scfg.get("adjust", [])
            lut_title, val = scfg["lut"]
            prof = scfg.get("profile", "Custom Plan")
        else:
            c_indices = c_info["seg_indices"]
            nb_ms = [seg_records[j].get("metrics") or default_m for j in c_indices if abs(j - idx) <= 2 and j != idx]
            if not nb_ms:
                nb_ms = [scene_anchor]
            local_nb_m = {k: float(np.mean([x.get(k, m.get(k, default_m[k])) for x in nb_ms])) for k in default_m}

            force_nb = (mname in explicit_match) or bool(scfg.get("force_anchor_match")) or (
                args.auto_match_outliers
                and (
                    abs(m.get("neutral_rb", 0.0) - scene_anchor.get("neutral_rb", 0.0)) > 0.022
                    or abs(m["mid"] - scene_anchor["mid"]) > 0.045
                )
            )
            lut_title, val, adj_list, prof = compute_shot_lut_and_adjustments(
                m, scene_cfg, scene_anchor, local_nb_m, force_nb,
                optical_db=optical_db, seg_scale=seg_scale, used_shot_luts=used_shot_luts,
                clip_model=rec.get("clip_model", project_model),
                seg_plan=scfg,
            )
            if "Angle-Jump" in prof:
                angle_jump_fixes += 1

        seg_applied[idx] = (lut_title, val, adj_list, prof)

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

    # 5. Promote active project LUTs into LutImportCfg.json[0:100] (index <= 99)
    promoted_indices = promote_active_luts_in_cfg([s[3] for s in summary], proj_dir)

    # 6. Write synced files (Root + Timelines + mini_draft.json + ALL Subdraft Compound Clips)
    root_json_str = json.dumps(root_draft, ensure_ascii=False, separators=(",", ":"))
    sync_targets = ["draft_info.json", "draft_info.json.bak", "template-2.tmp"]
    for tl_dir in glob.glob(os.path.join(proj_dir, "Timelines", "*")):
        if os.path.isdir(tl_dir):
            for fname in ("draft_info.json", "draft_info.json.bak", "template-2.tmp"):
                rel = os.path.relpath(os.path.join(tl_dir, fname), proj_dir)
                sync_targets.append(rel)

    for rel in sync_targets:
        full_p = os.path.join(proj_dir, rel)
        if os.path.exists(os.path.dirname(full_p)):
            with open(full_p, "w", encoding="utf-8") as f:
                f.write(root_json_str)

    sync_mini_draft_files(proj_dir, draft_containers, backup_dir)

    if has_compound:
        draft_by_id = {sd["id"]: sd["draft"] for sd in sub_drafts if "id" in sd and "draft" in sd}
        draft_by_combo = {sd["combination_id"]: sd["draft"] for sd in sub_drafts if sd.get("combination_id") and "draft" in sd}
        for sub_file in glob.glob(os.path.join(proj_dir, "subdraft", "*", "draft_content.json")):
            with open(sub_file, "r", encoding="utf-8") as sf:
                sub_d = json.load(sf)
            changed_sub = False
            for sd_entry in sub_d.get("materials", {}).get("drafts", []):
                sid = sd_entry.get("id")
                cid = sd_entry.get("combination_id")
                matched_draft = draft_by_id.get(sid) or (draft_by_combo.get(cid) if cid else None)
                if matched_draft is not None:
                    shutil.copy2(sub_file, os.path.join(backup_dir, f"subdraft_{sid}_content.json"))
                    sd_entry["draft"] = copy.deepcopy(matched_draft)
                    changed_sub = True
            if changed_sub:
                with open(sub_file, "w", encoding="utf-8") as sf:
                    json.dump(sub_d, sf, ensure_ascii=False, separators=(",", ":"))

    mode_str = clusters[0]["source_mode"] if clusters else "NONE"
    print(
        f"Graded {len(summary)} child video segments across {len(clusters)} Scene Clusters ({mode_str}) in '{os.path.basename(proj_dir)}' "
        f"[Model: {project_model.upper()} (buckets={model_stats['duration_buckets_sec']}) | Emotion: {args.emotion} | "
        f"100% Scene-Locked LUTs ({angle_jump_fixes} Achromatic Shot-Continuity Locks) | {zoomed_count} Zoomed Super-Res Boosts | "
        f"{total_adjusts} Adjust sliders ({total_adjusts/max(len(summary),1):.1f}/video) | LutImportCfg indices: {promoted_indices}]"
    )
    for c in clusters:
        cfg = c["scene_cfg"]
        sa = c.get("unified_anchor") or c["anchor"]
        note_tag = f" | Note: {cfg['ai_note']}" if cfg.get("ai_note") else ""
        print(
            f"  - Scene[{c['cluster_id']:2d}] ({c['bg_code']:22s} | {c.get('regime',''):14s}) Segs {c['start_seg']:3d}..{c['end_seg']:3d} ({len(c['seg_indices']):2d} segs) | "
            f"Primary LUT: {cfg['lut']:26s} ({cfg['base_val']*100:.0f}%) | Raw Mid={sa['mid']:.2f} RB={sa['rb']:+.2f} NeutRB={sa.get('neutral_rb',0):+.2f} -> Target Mid={cfg['target_mid']:.2f} RB={cfg['target_rb']:+.2f}{note_tag}"
        )

    comp_jpg = f"/tmp/{os.path.basename(proj_dir).replace(' ', '_')}_graded_comparison.jpg"
    try:
        comp_pages = render_graded_comparison_sheet(
            all_seg_tuples, seg_records, clusters, lut_map, comp_jpg,
            proj_dir=proj_dir, optical_db=optical_db, seg_applied=seg_applied,
        )
        print(f"\n[MANDATORY AI VISUAL QA] Rendered RAW vs. GRADED comparison sheet in {time.time() - t0:.1f}s total: {comp_jpg}")
        if len(comp_pages) > 1:
            print(f"-> High-Res Paginated QA Sheets ({len(comp_pages)} pages): {', '.join(comp_pages)}")
    except Exception as exc:
        print(f"Note: Could not render graded comparison sheet ({exc})")

    if args.relaunch:
        subprocess.run(["open", "-a", "/Applications/CapCut.app"], check=False)


if __name__ == "__main__":
    main()
