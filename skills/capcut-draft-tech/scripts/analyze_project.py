#!/usr/bin/env python3
"""Segment-level (source_timerange) 16-metric video diagnostics, deterministic AI model detection
(Seedance / Grok / Omni / Hybrid from source video durations), and Ground-Truth Script/BANGTRA Scene Mapping."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import glob
import json
import os
import re
import subprocess
import time
import numpy as np

CAPCUT_PROJECTS_ROOT = "/Users/daho100/Movies/CapCut/User Data/Projects/com.lveditor.draft"


def resolve_project_dir(project_arg: str) -> str:
    if os.path.isdir(project_arg):
        return os.path.abspath(project_arg)
    candidate = os.path.join(CAPCUT_PROJECTS_ROOT, project_arg)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(f"CapCut project not found: {project_arg}")


def resolve_video_path(vpath: str, proj_dir: str) -> str:
    if not vpath:
        return ""
    if vpath.startswith("##_draftpath_placeholder_"):
        vpath = re.sub(r"^##_draftpath_placeholder_[^#]+_##", proj_dir, vpath)
    return vpath


def get_default_metrics_cache_path(proj_dir: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.basename(os.path.abspath(proj_dir))).strip("_")
    return f"/tmp/capcut_metrics_{slug}.json"


def get_default_frames_cache_path(proj_dir: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.basename(os.path.abspath(proj_dir))).strip("_")
    return f"/tmp/capcut_frames_{slug}.npy"


def collect_all_video_segments(root_draft: dict) -> list[tuple[dict, dict, dict, str]]:
    """
    Collect EVERY child video segment (`video con`) across all Compound Clips (`materials.drafts`,
    including multi-level nested Compound Clips) AND any direct video segments on the root timeline
    (`root_draft.tracks`), sorted in exact chronological timeline order.
    Returns list of tuples: (draft_obj, seg_obj, video_material_obj, container_label).
    """
    timed_items = []
    visited_drafts = set()
    root_nested = {
        nd["id"]: nd
        for nd in root_draft.get("materials", {}).get("drafts", [])
        if isinstance(nd, dict) and "draft" in nd
    }

    def _scan_draft(d_obj: dict, label: str, offset_us: int = 0):
        vids = {v["id"]: v for v in d_obj.get("materials", {}).get("videos", []) if v.get("type") != "photo"}
        local_nested = {
            nd["id"]: nd
            for nd in d_obj.get("materials", {}).get("drafts", [])
            if isinstance(nd, dict) and "draft" in nd
        }
        nested_drafts = {**root_nested, **local_nested}
        for trk_idx, trk in enumerate(d_obj.get("tracks", [])):
            if trk.get("type", "video") != "video":
                continue
            for seg in trk.get("segments", []):
                seg_start_us = offset_us + int(seg.get("target_timerange", {}).get("start", 0))
                mid = seg.get("material_id", "")
                refs = [mid] + list(seg.get("extra_material_refs", []))
                matched_nd_id = next((r for r in refs if r in nested_drafts), None)
                if matched_nd_id:
                    visited_drafts.add(matched_nd_id)
                    nd = nested_drafts[matched_nd_id]
                    _scan_draft(nd["draft"], f"{label}/Compound[{nd.get('name') or matched_nd_id[:6]}]", seg_start_us)
                elif mid in vids and vids[mid].get("path"):
                    timed_items.append((seg_start_us, trk_idx, d_obj, seg, vids[mid], f"{label}:T{trk_idx}"))

    _scan_draft(root_draft, "Root", 0)
    for i, sd in enumerate(root_draft.get("materials", {}).get("drafts", [])):
        if "draft" in sd and sd.get("id") not in visited_drafts:
            _scan_draft(sd["draft"], f"Compound[{i}]", 0)

    timed_items.sort(key=lambda x: (x[0], x[1]))
    return [(x[2], x[3], x[4], x[5]) for x in timed_items]


def detect_source_models(all_seg_tuples: list, proj_dir: str) -> tuple[str, dict, dict]:
    """
    Deterministic AI video generator detection from un-cut source video durations (`video suộc gốc`):
    - No source video > 10.5s (all <= 10s) -> 'omni' (Google Omni 1.1)
    - Source videos span 6s / 10s / 15s (none > 15.5s, has 10.5s..15.5s or < 8.5s) -> 'grok' (Grok Video)
    - Has source video(s) > 15.5s (~25s-30s):
      - If >= 35% of source videos are > 15.5s -> 'seedance' (any short <= 15.5s clips inside are flagged 'grok' replacements)
      - If < 35% of source videos are > 15.5s (e.g. 30s Seedance hook + 10s Omni body) -> 'hybrid'
    Returns (project_model, clip_model_by_path, stats_dict).
    """
    src_durs_by_path = {}
    for _, _, v, _ in all_seg_tuples:
        vp = resolve_video_path(v.get("path", ""), proj_dir)
        if not vp:
            continue
        dur_s = float(v.get("duration", 0)) / 1e6
        if dur_s > 0.1:
            src_durs_by_path[vp] = round(dur_s, 2)

    valid_durs = [d for d in src_durs_by_path.values() if 0.5 <= d <= 120.0]
    gt_15 = sum(1 for d in valid_durs if d > 15.5)
    b_10_15 = sum(1 for d in valid_durs if 10.5 < d <= 15.5)
    b_6 = sum(1 for d in valid_durs if d < 8.5)
    le_10 = sum(1 for d in valid_durs if d <= 10.5)
    total_valid = max(len(valid_durs), 1)

    if gt_15 > 0:
        if gt_15 / total_valid >= 0.35:
            project_model = "seedance"
            body_Short_model = "grok"
        else:
            project_model = "hybrid"
            body_Short_model = "grok" if (b_10_15 > 0 or (b_6 > 0 and le_10 > b_6)) else "omni"
    elif b_10_15 > 0 or (b_6 > 0 and le_10 >= b_6):
        project_model = "grok"
        body_Short_model = "grok"
    else:
        project_model = "omni"
        body_Short_model = "omni"

    clip_model_by_path = {}
    for vp, dur_s in src_durs_by_path.items():
        if dur_s > 15.5 and dur_s <= 120.0:
            clip_model_by_path[vp] = "seedance"
        elif project_model in ("seedance", "hybrid"):
            clip_model_by_path[vp] = body_Short_model
        else:
            clip_model_by_path[vp] = project_model

    buckets = dict(sorted(Counter(int(round(d)) for d in valid_durs).items()))
    stats = {
        "project_model": project_model,
        "total_unique_sources": len(valid_durs),
        "gt_15s_count": gt_15,
        "10_to_15s_count": b_10_15,
        "le_10s_count": le_10,
        "duration_buckets_sec": buckets,
    }
    return project_model, clip_model_by_path, stats


def _measure_single_frame(arr: np.ndarray) -> dict:
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    p1, p5, p25, p50, p75, p95, p99 = np.percentile(luma, [1, 5, 25, 50, 75, 95, 99])
    mid = float(np.mean(luma))
    contrast = float(p95 - p5)
    iqr_ratio = float((p75 - p25) / max(contrast, 1e-4))
    rb_map = r - b
    gm_map = g - 0.5 * (r + b)
    rb = float(np.mean(rb_map))
    shd_mask = luma <= p25
    mid_mask = (luma > p25) & (luma < p75)
    hi_mask = luma >= p75
    shd_rb = float(np.mean(rb_map[shd_mask])) if np.any(shd_mask) else rb
    mid_rb = float(np.mean(rb_map[mid_mask])) if np.any(mid_mask) else rb
    hi_rb = float(np.mean(rb_map[hi_mask])) if np.any(hi_mask) else rb
    gm = float(np.mean(gm_map))
    cmax = np.max(arr, axis=2)
    cmin = np.min(arr, axis=2)
    chroma = cmax - cmin
    sat = float(np.mean(chroma))
    hsv_s_map = chroma / np.maximum(cmax, 1e-4)
    hsv_s = float(np.mean(hsv_s_map))

    # Achromatic / near-neutral pixel white-balance probe (separates true ambient lighting cast from colorful props/hats/walls)
    valid_luma = (luma >= 0.06) & (luma <= 0.90)
    low_chroma_th = max(0.065, float(np.percentile(chroma[valid_luma], 30)) if np.any(valid_luma) else 0.08)
    neutral_mask = valid_luma & (chroma <= low_chroma_th)
    if np.sum(neutral_mask) >= 16:
        neutral_rb = float(np.mean(rb_map[neutral_mask]))
        neutral_gm = float(np.mean(gm_map[neutral_mask]))
    else:
        neutral_rb = float(min(abs(rb), abs(shd_rb), abs(hi_rb)) * (1.0 if rb >= 0 else -1.0))
        neutral_gm = gm

    # Central-frame human skin locus (excludes top sky / bottom floor and pure yellow/wood surfaces)
    rc, gc, bc = r[14:106, 22:194], g[14:106, 22:194], b[14:106, 22:194]
    lc = luma[14:106, 22:194]
    sc_hsv = hsv_s_map[14:106, 22:194]
    rb_c = rc - bc
    gb_ratio = (gc - bc) / np.maximum(rb_c, 1e-4)
    skin_mask = (rc > gc) & (gc > bc) & (rb_c > 0.060) & (gb_ratio >= 0.20) & (gb_ratio <= 0.68) & (lc > 0.16) & (lc < 0.88)
    if np.sum(skin_mask) >= 16:
        skin_rb = float(np.mean(rb_c[skin_mask]))
        skin_sat = float(np.mean(sc_hsv[skin_mask]))
        skin_hi = float(np.percentile(lc[skin_mask], 88))
    else:
        skin_rb = float(max(rb, mid_rb))
        skin_sat = float(hsv_s)
        skin_hi = float(p75)

    gx = np.abs(np.diff(luma, axis=1)).mean()
    gy = np.abs(np.diff(luma, axis=0)).mean()
    sharp = float(0.5 * (gx + gy))
    blur = 0.25 * (luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:])
    hf_noise = float(np.std(luma[1:-1, 1:-1] - blur))
    return {
        "p1": float(p1), "p5": float(p5), "p25": float(p25),
        "mid": float(mid), "p50": float(p50), "p75": float(p75),
        "p95": float(p95), "p99": float(p99), "contrast": float(contrast),
        "iqr_ratio": float(iqr_ratio), "rb": float(rb), "shd_rb": float(shd_rb),
        "mid_rb": float(mid_rb), "hi_rb": float(hi_rb),
        "neutral_rb": float(neutral_rb), "neutral_gm": float(neutral_gm),
        "gm": float(gm), "sat": float(sat), "hsv_s": float(hsv_s),
        "sharp": float(sharp), "hf_noise": float(hf_noise),
        "skin_rb": float(skin_rb), "skin_sat": float(skin_sat), "skin_hi": float(skin_hi),
    }


def extract_segment_metrics(video_path: str, start_sec: float = 0.0, dur_sec: float = 5.0, return_frame: bool = False):
    """
    Samples 3 temporal points (20% Entry, 50% Center, 80% Exit) inside [start_sec, start_sec + dur_sec]
    at 216x122 resolution in a SINGLE low-contention `-threads 1` FFmpeg pass:
    - Computes the temporal median across the 3 frames to reject transient flashes, blinks, or pillar occlusions.
    - Records `entry_mid`, `exit_mid`, `entry_nrb`, `exit_nrb`, and `intra_drift` so adjacent cut matching
      knows how `Seg_{i-1}` exits (80%) and how `Seg_i` enters (20%).
    - Returns the clean 50% center `uint8` frame for `/tmp/capcut_frames_<slug>.npy`.
    """
    W, H = 216, 122
    f_bytes = W * H * 3
    dur_safe = max(0.4, float(dur_sec))
    t20 = max(0.0, start_sec + 0.20 * dur_safe)
    span = max(0.20, 0.60 * dur_safe)
    fps_val = 3.0 / span
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "1", "-v", "error",
        "-ss", f"{t20:.2f}", "-t", f"{span + 0.15:.2f}", "-i", video_path,
        "-vf", f"fps={fps_val:.4f},scale={W}:{H}", "-frames:v", "3",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
    ]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    if len(raw) < f_bytes:
        mid_ts = max(0.0, start_sec + 0.50 * dur_safe)
        cmd_fb = [
            "ffmpeg", "-nostdin", "-threads", "1", "-v", "error", "-ss", f"{mid_ts:.2f}", "-i", video_path,
            "-vf", f"scale={W}:{H}", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
        ]
        raw = subprocess.run(cmd_fb, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

    if len(raw) < f_bytes:
        fallback_m = {
            "p1": 0.02, "p5": 0.05, "p25": 0.14, "mid": 0.28, "p50": 0.27, "p75": 0.40, "p95": 0.62, "p99": 0.75,
            "contrast": 0.57, "iqr_ratio": 0.45, "rb": 0.12, "shd_rb": 0.06, "mid_rb": 0.13, "hi_rb": 0.15,
            "neutral_rb": 0.04, "neutral_gm": -0.008,
            "gm": -0.012, "sat": 0.15, "hsv_s": 0.44, "sharp": 0.040, "hf_noise": 0.028,
            "skin_rb": 0.16, "skin_sat": 0.34, "skin_hi": 0.50,
            "entry_mid": 0.28, "exit_mid": 0.28, "entry_nrb": 0.04, "exit_nrb": 0.04, "intra_drift": 0.0,
        }
        if return_frame:
            return fallback_m, np.zeros((H, W, 3), dtype=np.uint8)
        return fallback_m

    n_frames = len(raw) // f_bytes
    u8_stack = np.frombuffer(raw[:n_frames * f_bytes], dtype=np.uint8).reshape((n_frames, H, W, 3)).copy()

    per_frame_ms = [_measure_single_frame(u8_stack[i].astype(np.float32) / 255.0) for i in range(n_frames)]
    res_m = {}
    for k in per_frame_ms[0].keys():
        res_m[k] = round(float(np.median([fm[k] for fm in per_frame_ms])), 4)

    # Pick the frame in u8_stack that best matches the temporal median metrics (avoids transient 50% cutaways/blinks)
    best_f_idx = min(
        range(n_frames),
        key=lambda i: abs(per_frame_ms[i]["mid"] - res_m["mid"]) * 1.5
        + abs(per_frame_ms[i]["neutral_rb"] - res_m["neutral_rb"]) * 2.0,
    )
    u8_frame = u8_stack[best_f_idx]

    res_m["entry_mid"] = round(float(per_frame_ms[0]["mid"]), 4)
    res_m["exit_mid"] = round(float(per_frame_ms[-1]["mid"]), 4)
    res_m["entry_nrb"] = round(float(per_frame_ms[0]["neutral_rb"]), 4)
    res_m["exit_nrb"] = round(float(per_frame_ms[-1]["neutral_rb"]), 4)
    res_m["intra_drift"] = round(abs(float(per_frame_ms[-1]["mid"]) - float(per_frame_ms[0]["mid"])), 4)

    if return_frame:
        return res_m, u8_frame
    return res_m


def extract_clip_number(mname: str) -> int:
    m_v = re.search(r"V-(\d+)", mname, re.IGNORECASE)
    if m_v:
        return int(m_v.group(1))
    m_sc = re.search(r"scene_(\d+)", mname, re.IGNORECASE)
    if m_sc:
        return int(m_sc.group(1))
    m_num = re.match(r"(\d+)", mname)
    if m_num:
        return int(m_num.group(1))
    return -1


def extract_clip_key(mname: str):
    """
    Extracts either an alphanumeric SF Board shot ID (e.g. 'V-S1-02-B1' from 'V-S1-02-B1_v1.mp4')
    or an integer clip number (e.g. 1 from 'V-001.mp4' / '001.mp4').
    """
    base = os.path.splitext(os.path.basename(mname))[0]
    base = re.sub(r"_v\d+$", "", base, flags=re.IGNORECASE)
    m_vs = re.match(r"^(V-S\d+-[0-9A-Z-]+)$", base, re.IGNORECASE)
    if m_vs:
        return m_vs.group(1).upper()
    return extract_clip_number(mname)


def classify_script_regime(
    text: str,
    bg_code: str = "",
    scene_title: str = "",
    bg_label: str = "",
) -> tuple[str, bool]:
    """
    Infers the lighting/scene regime and flashback flag from ground-truth script descriptions
    (location code suffixes, scene headers, BG labels, BG descriptions, and clip prompts).
    1. Prioritizes explicit time-of-day / location suffixes on `bg_code`.
    2. Prioritizes authoritative `scene_title` + `bg_label` over incidental dialogue words in `shot_prompt`.
    3. Never uses generic Vietnamese words like `"nhà"` (which appear in `"sân trước nhà"`, `"toà nhà"`,
       `"phòng nhà đất"`, `"nhà chờ xe buýt"`) as an indoor trigger before checking outdoor/daytime keywords.
    """
    t = (text or "").lower()
    hdr = f"{scene_title or ''} | {bg_label or ''}".lower().strip()
    if not hdr or hdr == "|":
        hdr = t[:160]
    bg_up = (bg_code or "").upper()
    base_loc = bg_up.split(":")[-1] if ":" in bg_up else bg_up
    is_flashback = any(
        k in t
        for k in ("mười năm trước", "10 năm trước", "years ago", "sinh nhật 15", "quá khứ", "flashback", "hồi tưởng")
    )

    # 1. Explicit time-of-day / location tokens in `base_loc` (e.g. `LANE_DAY`, `ICEDOCK_DAY`, `ROW_FINAL_NIGHT`)
    if any(k in base_loc for k in ("_DAWN", "_DUSK", "_SUNSET", "_EVENING", "_GOLD", "DAWN", "DUSK", "EVENING", "SUNSET")):
        return "DAWN_DUSK_GOLD", is_flashback
    if any(k in base_loc for k in ("_NIGHT", "_MIDNIGHT", "NIGHT", "MIDNIGHT")):
        if any(k in t for k in ("bão tuyết", "mưa đêm", "đêm mưa", "blizzard", "storm", "rainy night", "/coldtone:")):
            return "NIGHT_COOL", is_flashback
        if any(k in base_loc for k in ("SERVER", "CLINIC", "HOSPITAL", "MORGUE", "LAB", "ICEROOM")):
            return "COOL_TECH", is_flashback
        return "NIGHT_WARM", is_flashback
    if any(k in base_loc for k in ("_DAY", "_NOON", "_MORNING", "NOON", "MORNING")):
        if any(k in base_loc for k in ("SERVER", "CLINIC", "HOSPITAL", "MORGUE", "LAB", "ICEROOM")):
            return "COOL_TECH", is_flashback
        if any(k in base_loc for k in ("BACKROOM", "KITCHEN", "DINING", "LIVING", "HALL", "FOYER", "LOBBY", "PARLOR", "SALON", "APARTMENT")):
            return "WARM_INT", is_flashback
        return "DAY_CRISP", is_flashback

    # 2. Authoritative Scene Title + BG Label (`hdr`) & Full Description (`t`) checks
    if any(k in t for k in ("bão tuyết", "mưa đêm", "đêm mưa", "blizzard", "storm", "rainy night", "/coldtone:", "ngoài trời đêm", "3 giờ sáng", "3:00 am")):
        return "NIGHT_COOL", is_flashback
    if any(k in t for k in ("server", "máy chủ", "hành lang kỹ thuật", "phòng khám", "clinic", "hospital", "bệnh viện", "kho lạnh", "iceroom", "phòng thay đồ", "locker")):
        return "COOL_TECH", is_flashback

    has_day_in_hdr = any(k in hdr for k in ("sáng", "trưa", "ban ngày", "8:", "9:", "10:", "11:", "12:", "morning", "noon"))
    has_night_in_hdr = any(k in hdr for k in ("đêm", "tối", "khuya", "nửa đêm", "night", "midnight", "8 giờ tối", "9 giờ tối"))

    if has_night_in_hdr and not has_day_in_hdr:
        if any(k in t for k in ("moonlit", "ánh trăng", "mưa đêm", "blizzard", "/coldtone:")):
            return "NIGHT_COOL", is_flashback
        return "NIGHT_WARM", is_flashback

    # Golden Hour / Sunset / Dusk / Dawn check
    if any(k in hdr for k in ("hoàng hôn", "chiều tà", "nắng chiều", "golden hour", "sunset", "dusk", "bình minh", "sunrise")):
        return "DAWN_DUSK_GOLD", is_flashback
    if any(k in hdr for k in ("rạng sáng", "rạng đông", "dawn")):
        if "cold gray dawn" in t or "gray dawn" in t:
            return "DAY_CRISP", is_flashback
        return "DAWN_DUSK_GOLD", is_flashback
    if "chiều" in hdr and any(k in hdr for k in ("hiên", "sân", "ngoài", "vườn", "công viên", "bến", "đường", "phố", "porch", "yard")):
        return "DAWN_DUSK_GOLD", is_flashback
    if not has_day_in_hdr and any(k in t for k in ("warm golden afternoon", "golden hour", "hoàng hôn", "chiều tà", "nắng chiều", "sunset", "dusk")):
        return "DAWN_DUSK_GOLD", is_flashback

    # Explicit Outdoor / Street / Yard / Boardroom / Site Container in Scene Header -> DAY_CRISP
    outdoor_or_crisp_hdr = (
        "sân trước", "sân sau", "bậc thềm trước", "lề đường", "vỉa hè", "ngoài trời",
        "trạm xe buýt", "nhà chờ xe buýt", "lễ động thổ", "công trường", "container văn phòng",
        "phòng họp", "boardroom", "sân đỗ", "nhà chứa", "hangar", "tarmac",
        "công viên", "bãi biển", "bến cảng", "con phố", "phố ",
        "front yard", "backyard", "street", "sidewalk", "curb", "bus stop", "exterior", "outdoor",
    )
    if any(k in hdr for k in outdoor_or_crisp_hdr):
        return "DAY_CRISP", is_flashback

    if not has_day_in_hdr and any(k in t for k in ("đêm", "tối", "khuya", "nửa đêm", "night", "evening", "midnight", "/amberlook:", "tiệc tối", "8 giờ tối", "9 giờ tối")):
        return "NIGHT_WARM", is_flashback

    # Warm Interior Rooms (without generic "nhà")
    warm_int_keywords = (
        "bếp", "kitchen", "gác mái", "attic", "căn hộ", "apartment",
        "phòng khách", "living room", "phòng ăn", "dining", "thư phòng", "study",
        "bàn thanh tra", "phòng nhà đất", "sảnh", "lobby", "foyer", "salon", "backroom",
        "nhà thờ", "chapel",
    )
    if any(k in hdr for k in warm_int_keywords) or (not has_day_in_hdr and any(k in t for k in warm_int_keywords)):
        return "WARM_INT", is_flashback

    if has_day_in_hdr or any(k in t for k in ("văn phòng", "office", "phòng họp", "meeting", "boardroom", "điều trần", "hearing", "sân đỗ", "nhà chứa", "hangar", "tarmac", "ban ngày", "sáng", "trưa", "morning", "noon", "daylight")):
        return "DAY_CRISP", is_flashback

    if any(k in hdr for k in ("chiều", "afternoon")):
        return "WARM_INT", is_flashback

    if any(k in t for k in warm_int_keywords):
        return "WARM_INT", is_flashback

    return "AUTO", is_flashback


def discover_ground_truth_scene_map(
    video_paths: list,
    expected_max_clip: int = 0,
    project_model: str = "auto",
) -> tuple[dict, str, dict]:
    """
    Searches the video folders and parent folders for ground-truth script files:
    - sf-board.json (SF Board .project)
    - *BANGTRA*.txt + *PROMPT_ANH_REF*.txt + *PROMPT*.txt
    Returns (clip_key_to_bg_dict, source_file_path, bg_context_dict).
    """
    search_dirs = set()
    for vp in video_paths:
        if not vp:
            continue
        cur = os.path.dirname(vp)
        for _ in range(3):
            if os.path.isdir(cur) and cur not in ("/", "/Users", "/Users/daho100", "/Users/daho100/Documents", "/Users/daho100/Long", "/Users/daho100/Documents/SF Board"):
                search_dirs.add(cur)
            cur = os.path.dirname(cur)

    ordered_dirs = sorted(search_dirs, key=lambda x: (-len(x), x))

    # 1. Check sf-board.json
    for sdir in ordered_dirs:
        sfb = os.path.join(sdir, "sf-board.json")
        if os.path.exists(sfb):
            try:
                data = json.load(open(sfb, "r", encoding="utf-8"))
                clip_map = {}
                bg_context = {}
                distinct_variants = set()
                angle_variants = set()
                has_multi_variants = set()
                sf_lookup = {}
                bg_desc_by_id = {}
                bg_label_by_id = {}
                for sc in data.get("scenes", []):
                    for sf in sc.get("sfs", []):
                        rid = sf.get("id", "")
                        if rid:
                            sf_lookup[rid] = sf
                        rlabel = str(sf.get("label") or sf.get("name") or sf.get("title") or "").strip()
                        rdesc = " ".join(str(sf.get(k) or "") for k in ("label", "name", "title", "desc", "prompt")).strip()
                        if rid:
                            rid_up = rid.upper()
                            bg_desc_by_id[rid_up] = rdesc
                            bg_label_by_id[rid_up] = rlabel
                        m_var = re.match(r"^((?:BG|IMG)\d+)([a-z])$", rid, re.IGNORECASE)
                        if m_var:
                            base_v, suf_v = m_var.group(1).upper(), m_var.group(2).upper()
                            if suf_v != "A":
                                has_multi_variants.add(base_v)
                            if "góc" in rlabel.lower() or "angle" in rlabel.lower():
                                angle_variants.add(rid.upper())
                            else:
                                distinct_variants.add(rid.upper())

                def _resolve_sfb_bg(raw_ref: str, sc_id: str) -> str:
                    cur = raw_ref
                    for _ in range(6):
                        if not cur:
                            break
                        c_up = cur.upper()
                        if c_up.startswith(("REF_BG_", "BG", "IMG")):
                            if c_up.startswith("REF_BG_"):
                                return c_up.replace("REF_BG_", "")
                            base_only = re.sub(r"[A-Z]$", "", c_up)
                            if c_up in angle_variants:
                                return base_only
                            if base_only in has_multi_variants and c_up in distinct_variants:
                                return c_up
                            return base_only
                        node = sf_lookup.get(cur)
                        if not node:
                            break
                        refs = node.get("refs") or {}
                        nxt = refs.get("bg") or (refs.get("bgs") or [""])[0]
                        if not nxt or nxt == cur:
                            break
                        cur = nxt
                    return cur.upper() if cur else sc_id

                for sc in data.get("scenes", []):
                    sc_id = sc.get("id", "")
                    sc_title = sc.get("name") or sc.get("title") or ""
                    for sf in list(sc.get("sfs", [])) + list(sc.get("shots", [])):
                        cid = sf.get("id", "")
                        if not cid.upper().startswith("V-"):
                            continue
                        raw_bg = sf.get("sf") or (sf.get("refs") or {}).get("bg") or ((sf.get("refs") or {}).get("bgs") or [""])[0]
                        base_bg = _resolve_sfb_bg(raw_bg, sc_id)
                        ckey = extract_clip_key(cid)
                        bg_tag = f"{sc_id}:{base_bg}"
                        clip_map[ckey] = bg_tag
                        clip_map[str(ckey)] = bg_tag
                        shot_prompt = sf.get("prompt") or sf.get("desc") or sf.get("text") or ""
                        raw_bg_up = str(raw_bg).upper()
                        bg_desc = bg_desc_by_id.get(raw_bg_up, "") or bg_desc_by_id.get(base_bg, "")
                        bg_label = bg_label_by_id.get(raw_bg_up, "") or bg_label_by_id.get(base_bg, "")
                        combined_desc = f"{sc_title} | {bg_desc} | {shot_prompt[:180]}"
                        regime, is_fb = classify_script_regime(combined_desc, bg_tag, scene_title=sc_title, bg_label=bg_label)
                        if bg_tag not in bg_context or bg_context[bg_tag].get("regime") == "AUTO":
                            bg_context[bg_tag] = {
                                "scene_title": sc_title,
                                "bg_label": bg_label,
                                "bg_desc": bg_desc[:180],
                                "regime": regime,
                                "is_flashback": is_fb,
                            }
                if clip_map:
                    return clip_map, sfb, bg_context
            except Exception:
                pass

    # 2. Check *PROMPT_ANH_REF*.txt for BG descriptions & camera-angle vs distinct-room rules
    ref_bg_descs = {}
    angle_bg_bases = set()
    distinct_bg_codes = set()
    for sdir in ordered_dirs:
        for rfile in sorted(glob.glob(os.path.join(sdir, "*ANH_REF*.txt"))):
            try:
                for line in open(rfile, "r", encoding="utf-8", errors="ignore"):
                    m_ref = re.match(r"^\s*(BG\d+)([a-z])?\s*[—\-:]\s*([^\n]+)", line.strip(), re.IGNORECASE)
                    if m_ref:
                        base_b = m_ref.group(1).upper()
                        suf_b = (m_ref.group(2) or "").upper()
                        full_b = f"{base_b}{suf_b}"
                        desc_b = m_ref.group(3).strip()
                        ref_bg_descs[full_b] = desc_b
                        if base_b not in ref_bg_descs:
                            ref_bg_descs[base_b] = desc_b
                        d_low = desc_b.lower()
                        if "góc" in d_low or "angle" in d_low:
                            angle_bg_bases.add(base_b)
                        elif suf_b in ("B", "C", "D"):
                            distinct_bg_codes.add(full_b)
            except Exception:
                pass

    # 3. Check *BANGTRA*.txt or *PROMPT*.txt (score candidates to match project clip count & duration!)
    candidates = []
    for sdir in ordered_dirs:
        candidates.extend(sorted(glob.glob(os.path.join(sdir, "*BANGTRA*.txt"))))
        candidates.extend(sorted(glob.glob(os.path.join(sdir, "*PROMPT*.txt"))))

    best_result = ({}, "", {})
    best_score = -1e9

    for cfile in candidates:
        bname = os.path.basename(cfile)
        if "ANH_REF" in bname or "THIEU" in bname or "SUA_" in bname or "SKETCH" in bname:
            continue
        try:
            lines = open(cfile, "r", encoding="utf-8", errors="ignore").readlines()
            full_txt = "".join(lines)
            for m_hdr in re.finditer(r"----([^\n]+)----\s*\n(?:SCENE[^\n]*\n)?\s*\d{1,3}\s*\|[^|]+\|[^|]+\|\s*(BG\d+)[a-z]", full_txt, re.IGNORECASE):
                hdr_low = m_hdr.group(1).lower()
                if "góc" in hdr_low or "angle" in hdr_low:
                    angle_bg_bases.add(m_hdr.group(2).upper())

            clip_map = {}
            bg_context = {}
            row_durs = []
            era_idx = 0
            era_label = "E0"
            last_tod = ""
            current_hdr_text = ""

            def _norm_bg(base_b: str, suf_b: str) -> str:
                full_b = f"{base_b}{suf_b}"
                if full_b in distinct_bg_codes or suf_b == "C":
                    return full_b
                if suf_b and base_b in angle_bg_bases and suf_b in ("A", "B"):
                    return base_b
                if suf_b and not distinct_bg_codes:
                    return base_b
                return full_b

            for line in lines:
                line_s = line.strip()
                if line_s.startswith("----") and line_s.endswith("----"):
                    current_hdr_text = line_s.strip("- ").strip()
                    hdr_up = current_hdr_text.upper()
                    if any(k in hdr_up for k in ("ĐÊM", "TỐI", "KHUYA", "NIGHT", "EVENING")):
                        tod = "NIGHT"
                    elif any(k in hdr_up for k in ("HOÀNG HÔN", "CHIỀU", "DUSK", "SUNSET")):
                        tod = "DUSK"
                    elif any(k in hdr_up for k in ("BÌNH MINH", "DAWN", "SUNRISE")):
                        tod = "DAWN"
                    else:
                        tod = "DAY"
                    is_angle_switch = ("GÓC TỪ" in hdr_up or "GÓC NHÌN" in hdr_up)
                    if not is_angle_switch or tod != last_tod or era_idx == 0:
                        era_idx += 1
                    last_tod = tod
                    era_label = f"E{era_idx}_{tod}"
                    continue

                if "THẺ CHỮ" in line_s.upper() or "THE CHU" in line_s.upper():
                    era_idx += 1
                    m_tc = re.search(r"(?:THẺ CHỮ|THE CHU)\s*[:—\-]\s*([^|]+)", line_s, re.IGNORECASE)
                    tc_raw = m_tc.group(1).strip().strip('"“”') if m_tc else f"E{era_idx}"
                    current_hdr_text = tc_raw
                    tc_short = re.sub(r"[^A-Z0-9]+", "_", tc_raw.upper())[:14].strip("_") or f"E{era_idx}"
                    era_label = f"E{era_idx}_{tc_short}"
                    continue

                m_row = re.match(r"^(\d{1,3})\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*([^|]+)\|(.*)$", line_s)
                if m_row:
                    cnum = int(m_row.group(1))
                    row_durs.append(int(m_row.group(2)))
                    refs_str = m_row.group(3)
                    tail_str = m_row.group(4).strip()
                    # Check if tail_str has an explicit time-jump marker (like in THE_HIVE_10S_BANGTRA.txt)
                    if "CHUYỂN CẢNH" in tail_str.upper():
                        for kw, tod_tag in (("ĐÊM", "NIGHT"), ("TỐI", "NIGHT"), ("HOÀNG HÔN", "DUSK"), ("CHIỀU", "DUSK"), ("BÌNH MINH", "DAWN"), ("SÁNG", "DAY")):
                            if kw in tail_str.upper():
                                era_idx += 1
                                era_label = f"E{era_idx}_{tod_tag}"
                                current_hdr_text = tail_str
                                break
                    m_bg = re.search(r"(BG\d+)([a-z])?", refs_str, re.IGNORECASE)
                    if m_bg:
                        base_b, suf_b = m_bg.group(1).upper(), (m_bg.group(2) or "").upper()
                        bg_norm = _norm_bg(base_b, suf_b)
                        bg_tag = f"{era_label}:{bg_norm}"
                        clip_map[cnum] = bg_tag
                        bg_desc_val = ref_bg_descs.get(f"{base_b}{suf_b}", ref_bg_descs.get(base_b, ""))
                        raw_desc = f"{current_hdr_text} | {bg_desc_val} | {tail_str[:120]}"
                        regime, is_fb = classify_script_regime(
                            raw_desc, bg_tag, scene_title=current_hdr_text, bg_label=bg_desc_val[:80]
                        )
                        if bg_tag not in bg_context or bg_context[bg_tag].get("regime") == "AUTO":
                            bg_context[bg_tag] = {
                                "scene_title": current_hdr_text or tail_str[:60],
                                "bg_desc": bg_desc_val,
                                "regime": regime,
                                "is_flashback": is_fb,
                            }
                    else:
                        clip_map[cnum] = ""
                    continue

                m_pr = re.match(r"^Scene\s+(\d{1,3})\s*:(.*)$", line_s, re.IGNORECASE)
                if m_pr:
                    cnum = int(m_pr.group(1))
                    body = m_pr.group(2)
                    if "THẺ CHỮ" in body.upper() or "title card" in body.lower():
                        era_idx += 1
                        era_label = f"E{era_idx}"
                        continue
                    m_bg = re.search(r"(BG\d+)([a-z])?", body, re.IGNORECASE)
                    if m_bg:
                        base_b, suf_b = m_bg.group(1).upper(), (m_bg.group(2) or "").upper()
                        bg_norm = _norm_bg(base_b, suf_b)
                        bg_tag = f"{era_label}:{bg_norm}"
                        clip_map[cnum] = bg_tag
                        bg_desc_val = ref_bg_descs.get(f"{base_b}{suf_b}", "")
                        raw_desc = f"{current_hdr_text} | {bg_desc_val} | {body[:160]}"
                        regime, is_fb = classify_script_regime(
                            raw_desc, bg_tag, scene_title=current_hdr_text, bg_label=bg_desc_val[:80]
                        )
                        if bg_tag not in bg_context or bg_context[bg_tag].get("regime") == "AUTO":
                            bg_context[bg_tag] = {
                                "scene_title": current_hdr_text,
                                "bg_desc": bg_desc_val,
                                "regime": regime,
                                "is_flashback": is_fb,
                            }

            if len(clip_map) >= 10:
                max_mapped = max(k for k in clip_map.keys() if isinstance(k, int))
                count_err = abs(max_mapped - expected_max_clip) if expected_max_clip > 0 else 0
                med_dur = float(np.median(row_durs)) if row_durs else 20.0
                dur_bonus = 0.0
                if project_model == "seedance" and med_dur >= 20.0:
                    dur_bonus = 50.0
                elif project_model in ("omni", "hybrid") and med_dur <= 12.0:
                    dur_bonus = 50.0
                bangtra_bonus = 20.0 if "BANGTRA" in bname.upper() else 0.0
                score = bangtra_bonus + dur_bonus - count_err * 1.5
                if score > best_score:
                    best_score = score
                    best_result = (clip_map, cfile, bg_context)
        except Exception:
            pass

    return best_result


def cluster_timeline_scenes(seg_records: list, clip_bg_map: dict = None, bg_context: dict = None) -> list:
    """
    Groups consecutive timeline segments into Continuous Scene Clusters:
    - Priority 1 (Ground-Truth): Uses clip_bg_map from BANGTRA.txt / sf-board.json.
      Transition clips without a BG tag inherit the current scene's BG.
    - Priority 2 (Visual Fallback): Uses a sustained multi-segment window only if no ground-truth map exists.
    """
    n = len(seg_records)
    if n == 0:
        return []
    bg_context = bg_context or {}

    default_m = {
        "p5": 0.05, "p25": 0.15, "mid": 0.28, "p95": 0.62, "p99": 0.76,
        "contrast": 0.55, "rb": 0.13, "shd_rb": 0.06, "mid_rb": 0.13, "hi_rb": 0.15,
        "neutral_rb": 0.04, "neutral_gm": -0.008,
        "gm": -0.012, "sat": 0.15, "hsv_s": 0.45,
        "sharp": 0.042, "hf_noise": 0.030,
        "skin_rb": 0.16, "skin_sat": 0.34, "skin_hi": 0.50
    }
    ms = [{**default_m, **(r.get("metrics") or {})} for r in seg_records]

    if clip_bg_map:
        raw_bgs = []
        for idx_r, r in enumerate(seg_records):
            mname = r["material_name"]
            ckey = extract_clip_key(mname)
            if idx_r == 0 and r.get("clip_model") == "seedance" and any(x.get("clip_model") == "omni" for x in seg_records[1:15]):
                raw_bgs.append("SEEDANCE_HOOK:BG1")
                continue
            raw_bgs.append(clip_bg_map.get(ckey) or clip_bg_map.get(str(ckey), ""))

        for i in range(n):
            if not raw_bgs[i]:
                prev_bg = next((raw_bgs[j] for j in range(i - 1, -1, -1) if raw_bgs[j] and not raw_bgs[j].startswith("SEEDANCE_HOOK")), "")
                next_bg = next((raw_bgs[j] for j in range(i + 1, n) if raw_bgs[j]), "")
                raw_bgs[i] = prev_bg or next_bg or "BG_DEFAULT"

        boundaries = [0]
        for i in range(1, n):
            if raw_bgs[i] != raw_bgs[i - 1]:
                boundaries.append(i)
        boundaries.append(n)
        source_mode = "GROUND-TRUTH"
    else:
        boundaries = [0]
        last_cut = 0
        for i in range(1, n):
            if seg_records[i]["material_name"] == seg_records[i - 1]["material_name"]:
                continue
            if i - last_cut < 4:
                continue
            w_prev = ms[max(last_cut, i - 4):i]
            w_next = ms[i:min(n, i + 4)]
            if len(w_next) < 2:
                continue
            prev_mid = float(np.median([x["mid"] for x in w_prev]))
            next_mid = float(np.median([x["mid"] for x in w_next]))
            prev_rb = float(np.median([x["rb"] for x in w_prev]))
            next_rb = float(np.median([x["rb"] for x in w_next]))
            prev_p5 = float(np.median([x["p5"] for x in w_prev]))
            next_p5 = float(np.median([x["p5"] for x in w_next]))
            next2_mid = ms[min(n - 1, i + 1)]["mid"]
            next2_rb = ms[min(n - 1, i + 1)]["rb"]
            sustained_shift = (
                (abs(next_mid - prev_mid) > 0.070 and abs(next2_mid - prev_mid) > 0.052)
                or (abs(next_rb - prev_rb) > 0.048 and abs(next2_rb - prev_rb) > 0.036)
                or (abs(next_p5 - prev_p5) > 0.045)
            )
            if sustained_shift:
                boundaries.append(i)
                last_cut = i
        boundaries.append(n)
        raw_bgs = ["AUTO" for _ in range(n)]
        source_mode = "VISUAL-FALLBACK"

    clusters = []
    for b_idx in range(len(boundaries) - 1):
        s_idx, e_idx = boundaries[b_idx], boundaries[b_idx + 1]
        seg_range = list(range(s_idx, e_idx))
        valid_seg_range = [j for j in seg_range if ms[j].get("p95", 1.0) >= 0.04] or seg_range
        c_ms = [ms[j] for j in valid_seg_range]
        anchor = {k: round(float(np.median([x[k] for x in c_ms])), 4) for k in default_m}
        bg_code = raw_bgs[s_idx]
        ctx = bg_context.get(bg_code, {})
        regime = ctx.get("regime", "AUTO")
        if not regime or regime == "AUTO":
            bg_up = bg_code.upper()
            if bg_up.startswith("SEEDANCE_HOOK"):
                regime = "SEEDANCE_HOOK"
            elif "_DAWN" in bg_up or "_DUSK" in bg_up or "GOLD" in bg_up or "SUNSET" in bg_up:
                regime = "DAWN_DUSK_GOLD"
            elif "_NIGHT" in bg_up or "NIGH" in bg_up or "RAIN" in bg_up:
                regime = "NIGHT_WARM" if anchor["rb"] > 0.045 else "NIGHT_COOL"
            elif "_DAY" in bg_up:
                regime = "WARM_INT" if anchor["rb"] > 0.140 else "DAY_CRISP"
            elif anchor["mid"] <= 0.255:
                regime = "NIGHT_WARM" if anchor["rb"] > 0.045 else "NIGHT_COOL"
            elif anchor["rb"] >= 0.165:
                regime = "DAWN_DUSK_GOLD" if anchor["mid"] >= 0.34 else "WARM_INT"
            elif anchor["rb"] <= 0.020:
                regime = "COOL_TECH" if anchor["mid"] <= 0.28 else "DAY_CRISP"
            else:
                regime = "WARM_INT" if anchor["rb"] >= 0.125 else "DAY_CRISP"
        elif regime == "DAWN_DUSK_GOLD" and anchor["rb"] < 0.068 and anchor.get("neutral_rb", 0.0) < 0.022:
            # Empirically cool/neutral early morning light (e.g. "rạng sáng" / cold gray dawn) -> DAY_CRISP
            regime = "DAY_CRISP"

        anchor_seg = min(
            valid_seg_range,
            key=lambda j: abs(ms[j]["mid"] - anchor["mid"]) * 1.5
            + abs(ms[j].get("neutral_rb", 0.0) - anchor.get("neutral_rb", 0.0)) * 2.0
            + abs(ms[j]["rb"] - anchor["rb"]) * 0.5,
        )
        outlier_seg = max(
            valid_seg_range,
            key=lambda j: abs(ms[j]["mid"] - anchor["mid"]) * 1.5
            + abs(ms[j].get("neutral_rb", 0.0) - anchor.get("neutral_rb", 0.0)) * 2.2
            + abs(ms[j]["rb"] - anchor["rb"]) * 0.45,
        )
        out_d_mid = round(ms[outlier_seg]["mid"] - anchor["mid"], 3)
        out_d_rb = round(ms[outlier_seg]["rb"] - anchor["rb"], 3)
        out_d_nrb = round(ms[outlier_seg].get("neutral_rb", 0.0) - anchor.get("neutral_rb", 0.0), 3)

        clusters.append({
            "cluster_id": b_idx,
            "bg_code": bg_code,
            "regime": regime,
            "is_flashback": bool(ctx.get("is_flashback", False)),
            "scene_title": ctx.get("scene_title", ""),
            "bg_label": ctx.get("bg_label", ""),
            "bg_desc": ctx.get("bg_desc", ""),
            "source_mode": source_mode,
            "start_seg": s_idx,
            "end_seg": e_idx - 1,
            "start_idx": s_idx,
            "end_idx": e_idx - 1,
            "seg_indices": seg_range,
            "anchor_seg": anchor_seg,
            "outlier_seg": outlier_seg,
            "outlier_delta_mid": out_d_mid,
            "outlier_delta_rb": out_d_rb,
            "outlier_delta_neutral_rb": out_d_nrb,
            "clips": sorted(set(seg_records[j]["material_name"] for j in seg_range)),
            "anchor": anchor,
        })

    # Unify `regime` across recurring clusters that share the exact same physical location & time-of-day `(base_loc, tod)`
    loc_tod_votes = {}
    for c in clusters:
        bg_up = c["bg_code"].upper()
        if bg_up in ("AUTO", "BG_DEFAULT") or bg_up.startswith("SEEDANCE_HOOK"):
            continue
        base_loc = bg_up.split(":")[-1] if ":" in bg_up else bg_up
        sc_t_low = (c.get("scene_title") or "").lower()
        tod = (
            "NIGHT" if ("NIGHT" in bg_up or any(k in sc_t_low for k in ("đêm", "tối", "khuya", "night"))) else (
                "EVENING" if "EVENING" in bg_up else (
                    "DUSK" if ("DUSK" in bg_up or any(k in sc_t_low for k in ("hoàng hôn", "chiều", "dusk", "sunset"))) else (
                        "DAWN" if ("DAWN" in bg_up or any(k in sc_t_low for k in ("rạng sáng", "bình minh", "dawn"))) else (
                            "NOON" if ("NOON" in bg_up or "trưa" in sc_t_low) else (
                                "DAY" if ("DAY" in bg_up or "sáng" in sc_t_low) else ""
                            )
                        )
                    )
                )
            )
        )
        lt_key = (base_loc, tod)
        c["_lt_key"] = lt_key
        reg = c.get("regime", "DAY_CRISP")
        loc_tod_votes.setdefault(lt_key, {})[reg] = loc_tod_votes.get(lt_key, {}).get(reg, 0) + len(c["seg_indices"])

    for c in clusters:
        lt_key = c.pop("_lt_key", None)
        if lt_key and lt_key in loc_tod_votes:
            best_reg = max(loc_tod_votes[lt_key].items(), key=lambda kv: kv[1])[0]
            c["regime"] = best_reg

    return clusters


def get_default_plan_path(proj_dir: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(proj_dir))
    return f"/tmp/capcut_plan_{slug}.json"


def get_default_scene_report_path(proj_dir: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(proj_dir))
    return f"/tmp/{slug}_scene_report.txt"


def generate_scene_report_and_starter_plan(
    proj_dir: str,
    project_model: str,
    gt_path: str,
    clusters: list,
    seg_records: list,
    preview_pages: list,
) -> tuple[str, str]:
    """
    Writes two ready-to-read files so the AI Colorist NEVER needs to run ad-hoc `python3` commands
    on `/tmp/capcut_metrics_<slug>.json`, `sf-board.json`, or `lut_optical_db.json`:
      1. `/tmp/<Project>_scene_report.txt`: Complete human-and-LLM-readable report containing:
         - Curated Clean 35mm LUT Reference Palette (from `lut_optical_db.json`)
         - Every Scene Cluster (`scene_title`, `bg_desc`, `regime`, `Auto-LUT`, `Anchor` metrics)
         - Every Child Segment (`Seg #0..#N-1` with `Mid`, `P5`, `P95`, `RB`, `NRB`, `SkRB`, `Sat`, and `[OUTLIER]` tags)
      2. `/tmp/capcut_plan_<slug>.json`: Pre-populated Starter AI Grading Plan JSON for all scenes with
         location-locked clean 35mm LUTs, `target_mid`, `preserve_warm_props`, and automatic biases for dark/tungsten scenes.
    """
    report_path = get_default_scene_report_path(proj_dir)
    plan_path = get_default_plan_path(proj_dir)

    # Choose clean, verified neutral 35mm LUTs by `(regime, base_loc_family)`
    loc_lut_map = {}
    day_pool = ("SKINTONE (10).cube", "FILM LOOK (15).cube", "FILM LOOK (5).cube", "HOLLYWOOD (1).cube")
    warm_pool = ("SKINTONE (10).cube", "THE ROMANCE 09.cube", "HOLLYWOOD (2).cube")
    gold_pool = ("THE ROMANCE 09.cube", "THE ROMANCE 04.cube")
    night_lift_pool = ("THE ROMANCE 01.cube", "SKINTONE (9).cube", "THE ROMANCE 09.cube")
    night_cool_tungsten_pool = ("SKINTONE (4).cube", "SKINTONE (9).cube")
    cool_tech_pool = ("FILM LOOK (11).cube", "THE ROMANCE 07.cube")

    plan_scenes = {}
    plan_seg_overrides = {}
    loc_idx_counter = 0

    # Group clusters by `(regime, loc_family)` so returning to the same physical location uses the exact same LUT
    family_groups = {}
    for c in clusters:
        bg_code = c.get("bg_code", "")
        base_loc = bg_code.split(":")[-1] if ":" in bg_code else bg_code
        loc_family = re.sub(r"[A-Z]$", "", base_loc.upper())
        reg = c.get("regime", "DAY_CRISP")
        family_groups.setdefault((reg, loc_family), []).append(c)

    for (reg, loc_fam), g_clusters in family_groups.items():
        weights = [max(1, len(x["seg_indices"])) for x in g_clusters]
        avg_mid = float(np.average([x["anchor"]["mid"] for x in g_clusters], weights=weights))
        avg_nrb = float(np.average([x["anchor"].get("neutral_rb", 0.0) for x in g_clusters], weights=weights))

        if reg == "DAWN_DUSK_GOLD":
            lut_choice = gold_pool[loc_idx_counter % len(gold_pool)]
            lut_val = 0.21
        elif reg == "NIGHT_WARM":
            if avg_nrb > 0.080:
                lut_choice = night_cool_tungsten_pool[0]
                lut_val = 0.19
            elif avg_mid < 0.250:
                lut_choice = night_lift_pool[0]
                lut_val = 0.19
            else:
                lut_choice = night_lift_pool[loc_idx_counter % len(night_lift_pool)]
                lut_val = 0.19
        elif reg in ("NIGHT_COOL", "COOL_TECH"):
            lut_choice = cool_tech_pool[loc_idx_counter % len(cool_tech_pool)]
            lut_val = 0.19
        elif reg == "WARM_INT":
            if avg_mid < 0.260:
                lut_choice = "THE ROMANCE 01.cube"
                lut_val = 0.19
            else:
                lut_choice = warm_pool[loc_idx_counter % len(warm_pool)]
                lut_val = 0.20
        else:
            # DAY_CRISP
            sc_titles = " ".join((x.get("scene_title") or "") for x in g_clusters).lower()
            if any(k in sc_titles for k in ("phòng họp", "boardroom", "văn phòng", "office", "tower")):
                lut_choice = "FILM LOOK (15).cube"
                lut_val = 0.19
            else:
                lut_choice = day_pool[0]
                lut_val = 0.20
        loc_lut_map[(reg, loc_fam)] = (lut_choice, lut_val, round(avg_mid, 3))
        loc_idx_counter += 1

    for c in clusters:
        cid = c["cluster_id"]
        reg = c.get("regime", "DAY_CRISP")
        bg_code = c.get("bg_code", "")
        base_loc = bg_code.split(":")[-1] if ":" in bg_code else bg_code
        loc_fam = re.sub(r"[A-Z]$", "", base_loc.upper())
        lut_choice, lut_val, grp_mid = loc_lut_map.get((reg, loc_fam), ("SKINTONE (10).cube", 0.20, c["anchor"]["mid"]))

        a = c["anchor"]
        anc_mid = a["mid"]
        anc_nrb = a.get("neutral_rb", 0.0)
        anc_rb = a["rb"]

        # Blend scene target_mid gently toward the location family average
        raw_t_mid = 0.65 * anc_mid + 0.35 * grp_mid
        if anc_mid < 0.160:
            t_mid = round(max(0.205, anc_mid + 0.060), 3)
        elif anc_mid < 0.240:
            t_mid = round(max(0.235, anc_mid + 0.035), 3)
        elif anc_mid > 0.455:
            t_mid = round(min(0.435, anc_mid - 0.020), 3)
        else:
            t_mid = round(raw_t_mid, 3)

        preserve_warm = bool(anc_nrb <= 0.080)
        sc_entry = {
            "regime": reg,
            "lut": lut_choice,
            "lut_val": lut_val,
            "target_mid": t_mid,
            "preserve_warm_props": preserve_warm,
        }
        if anc_mid < 0.160:
            sc_entry["bright_bias"] = 0.035
            sc_entry["shadow_bias"] = 0.045
        elif anc_mid < 0.230:
            sc_entry["bright_bias"] = 0.015
            sc_entry["shadow_bias"] = 0.025

        if not preserve_warm and anc_nrb > 0.080:
            sc_entry["temp_bias"] = round(max(-0.055, -(anc_nrb - 0.045) * 0.60), 3)
            sc_entry["sat_bias"] = -0.020
            if lut_choice == "SKINTONE (4).cube" and anc_mid >= 0.30:
                sc_entry["bright_bias"] = round(sc_entry.get("bright_bias", 0.0) + 0.020, 3)

        sc_title = c.get("scene_title") or bg_code
        sc_entry["note"] = f"{sc_title} ({bg_code})"
        plan_scenes[str(cid)] = sc_entry
        c["auto_plan"] = sc_entry

    starter_plan = {
        "global": {
            "temp_bias": 0.0,
            "bright_bias": 0.0,
            "contrast_bias": 0.005,
            "sat_bias": 0.0,
            "lut_val_bias": 0.0,
        },
        "scenes": plan_scenes,
        "segments": plan_seg_overrides,
    }
    try:
        with open(plan_path, "w", encoding="utf-8") as pf:
            json.dump(starter_plan, pf, indent=2, ensure_ascii=False)
    except Exception:
        pass

    lines = []
    lines.append(f"=== CAPCUT AI COLORIST SCENE & SEGMENT REPORT: {os.path.basename(proj_dir)} ===")
    lines.append(f"Model: {project_model.upper()} | Scenes: {len(clusters)} | Segments: {len(seg_records)} | Ground-Truth: {gt_path or 'Visual-Fallback'}")
    lines.append(f"Starter AI Plan Pre-Generated: {plan_path} (use edit_file to refine specific scenes/segments!)")
    lines.append(f"Visual Scene Strip Sheets: {', '.join(preview_pages)}")
    lines.append("")
    lines.append("=== CURATED CLEAN NEUTRAL 35MM LUT PALETTE (from lut_optical_db.json — DO NOT run python3 to query DB) ===")
    lines.append("  * DAY_CRISP (Outdoor/Daylight) : SKINTONE (10).cube (mid=0.290, midRB=+0.008, sknRB=+0.295, satGain=1.08)")
    lines.append("                                 | FILM LOOK (15).cube (mid=0.330, midRB=-0.012, sknRB=+0.347, satGain=1.22)")
    lines.append("                                 | FILM LOOK (5).cube  (mid=0.281, midRB=-0.018, sknRB=+0.355, satGain=1.24)")
    lines.append("                                 | HOLLYWOOD (1).cube  (mid=0.242, midRB=+0.009, sknRB=+0.320, satGain=1.19)")
    lines.append("  * WARM_INT (Indoor Day/Warm)   : SKINTONE (10).cube (mid=0.290, midRB=+0.008) | THE ROMANCE 09.cube (mid=0.305, midRB=+0.013)")
    lines.append("                                 | THE ROMANCE 01.cube (mid=0.449, midRB=+0.026, lifts dark interiors cleanly)")
    lines.append("                                 | HOLLYWOOD (2).cube  (mid=0.322, midRB=+0.027, sknRB=+0.329, satGain=1.19)")
    lines.append("  * DAWN_DUSK_GOLD (Golden Hour) : THE ROMANCE 09.cube (mid=0.305, midRB=+0.013, satGain=1.01)")
    lines.append("                                 | THE ROMANCE 04.cube (mid=0.253, midRB=+0.027, satGain=1.11)")
    lines.append("  * NIGHT_WARM (Night / Lantern) : THE ROMANCE 01.cube (mid=0.449, midRB=+0.026, ideal for lifting dark night/attic shots)")
    lines.append("                                 | SKINTONE (4).cube   (mid=0.259, midRB=-0.004, satGain=0.89, ideal for cooling heavy orange tungsten NRB>0.08)")
    lines.append("                                 | SKINTONE (9).cube   (mid=0.264, midRB=+0.001, satGain=0.88)")
    lines.append("  * NIGHT_COOL / COOL_TECH       : FILM LOOK (11).cube (mid=0.277, midRB=-0.026) | THE ROMANCE 07.cube (mid=0.225, midRB=-0.019)")
    lines.append("")
    lines.append("=== COMPLETE SCENE CLUSTERS & PER-SEGMENT SCOPE METRICS ===")

    for c in clusters:
        a = c["anchor"]
        ap = c.get("auto_plan", {})
        sc_t = c.get("scene_title") or ""
        bg_d = (c.get("bg_desc") or "").replace("\n", " ")[:130]
        lines.append(
            f"\n=== Scene[{c['cluster_id']:2d}] {c['bg_code']:12s} | {c.get('regime','AUTO'):14s} | "
            f"AutoLUT: {ap.get('lut',''):20s} ({int(round(ap.get('lut_val',0.2)*100))}%) | "
            f"segs {c['start_seg']}..{c['end_seg']} ({len(c['seg_indices'])} segs) | {sc_t} ==="
        )
        if bg_d:
            lines.append(f"  BG: {bg_d}")
        lines.append(
            f"  Anchor: Mid={a['mid']:.3f} P5={a['p5']:.3f} P95={a['p95']:.3f} RB={a['rb']:+.3f} "
            f"NRB={a.get('neutral_rb',0.0):+.3f} SkRB={a['skin_rb']:.3f} Sat={a['sat']:.3f} -> TargetMid={ap.get('target_mid', a['mid']):.3f}"
        )
        for s_idx in c["seg_indices"]:
            sr = seg_records[s_idx]
            sm = sr.get("metrics") or a
            dm = sm.get("mid", a["mid"]) - a["mid"]
            dnr = sm.get("neutral_rb", 0.0) - a.get("neutral_rb", 0.0)
            flag = ""
            if abs(dm) >= 0.045 or abs(dnr) >= 0.025:
                flag = f"  <-- [OUTLIER dMid={dm:+.3f} dNRB={dnr:+.3f}]"
            lines.append(
                f"  Seg #{s_idx:3d} ({sr['material_name']:10s} @ {sr['start_sec']:5.1f}s+{sr['dur_sec']:4.1f}s): "
                f"Mid={sm.get('mid',0):.3f} P5={sm.get('p5',0):.3f} P95={sm.get('p95',0):.3f} "
                f"RB={sm.get('rb',0):+.3f} NRB={sm.get('neutral_rb',0):+.3f} SkRB={sm.get('skin_rb',0):.3f} Sat={sm.get('sat',0):.3f}{flag}"
            )

    try:
        with open(report_path, "w", encoding="utf-8") as rf:
            rf.write("\n".join(lines) + "\n")
    except Exception:
        pass

    return report_path, plan_path


def extract_rgb_frame(video_path: str, ts_sec: float, w: int = 216, h: int = 122) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "1", "-v", "error", "-ss", f"{max(0.0, ts_sec):.2f}", "-i", video_path,
        "-vf", f"scale={w}:{h}", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
    ]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    if len(raw) >= w * h * 3:
        return np.frombuffer(raw[:w * h * 3], dtype=np.uint8).reshape((h, w, 3)).astype(np.float32) / 255.0
    return np.zeros((h, w, 3), dtype=np.float32)


def render_raw_scene_contact_sheet(
    all_seg_tuples: list,
    seg_records: list,
    clusters: list,
    out_jpg: str,
    proj_dir: str = "",
    frames_arr: np.ndarray = None,
):
    """
    Renders a Multi-Shot Scene Strip Contact Sheet (3 representative shots per Scene:
    [First Shot | Median Anchor Shot | Max-Outlier Shot]) so the AI Colorist can visually inspect
    both the full scene look (faces, costumes, props, background lighting) and intra-scene camera-angle jumps.
    Reuses `frames_arr` `(N, 122, 216, 3)` uint8 in memory for zero-FFmpeg instantaneous rendering.
    """
    from PIL import Image, ImageDraw
    tile_w, tile_h = 216, 122
    strip_w = tile_w * 3 + 10
    header_h, sub_h = 24, 18
    block_h = header_h + sub_h + tile_h + 6
    cols = 2
    rows = max(1, (len(clusters) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * strip_w + 12, rows * block_h + 8), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)

    def _fetch_scene_strip(idx_c):
        idx, c = idx_c
        s_first = c["start_seg"]
        s_anc = c.get("anchor_seg", c["seg_indices"][len(c["seg_indices"]) // 2])
        s_out = c.get("outlier_seg", c["end_seg"])
        if s_out == s_anc and len(c["seg_indices"]) > 1:
            s_out = c["end_seg"] if c["end_seg"] != s_anc else c["start_seg"]
        shot_ids = [s_first, s_anc, s_out]
        frames = []
        for s_id in shot_ids:
            rec = seg_records[s_id]
            if frames_arr is not None and s_id < len(frames_arr):
                u8 = frames_arr[s_id]
            else:
                _, _, v, _ = all_seg_tuples[s_id]
                ts = rec["start_sec"] + 0.5 * rec["dur_sec"]
                vp = resolve_video_path(v.get("path", ""), proj_dir)
                u8 = np.clip(extract_rgb_frame(vp, ts, tile_w, tile_h) * 255.0, 0, 255).astype(np.uint8)
            frames.append((s_id, rec, u8))
        return idx, c, frames

    strips = [_fetch_scene_strip(ic) for ic in enumerate(clusters)]

    for idx, c, frames in strips:
        r_i, c_i = divmod(idx, cols)
        x0 = c_i * strip_w + 6
        y0 = r_i * block_h + 4
        a = c["anchor"]
        reg_lbl = c.get("regime", "")[:10]
        hdr = (
            f"S{c['cluster_id']:02d} ({c['bg_code'][:16]}|{reg_lbl}) Segs {c['start_seg']}..{c['end_seg']} | "
            f"Mid:{a['mid']:.2f} RB:{a['rb']:+.2f} NeutRB:{a.get('neutral_rb',0.0):+.2f} SkinRB:{a.get('skin_rb',0.2):.2f}"
        )
        draw.text((x0 + 4, y0 + 4), hdr[:86], fill=(245, 225, 130))

        role_names = ("1.First", "2.Anchor", "3.Outlier")
        for t_i, (s_id, rec, u8) in enumerate(frames):
            tx = x0 + t_i * (tile_w + 2)
            ty = y0 + header_h
            m = rec.get("metrics") or a
            if t_i == 2:
                dm = m.get("mid", a["mid"]) - a["mid"]
                dnr = m.get("neutral_rb", a.get("neutral_rb", 0.0)) - a.get("neutral_rb", 0.0)
                sub_lbl = f"{role_names[t_i]} #{s_id} (dMid:{dm:+.2f} dNRB:{dnr:+.2f})"
                sub_col = (255, 175, 145) if (abs(dm) > 0.040 or abs(dnr) > 0.020) else (185, 215, 195)
            else:
                sub_lbl = f"{role_names[t_i]} #{s_id} (M:{m.get('mid',0):.2f} NRB:{m.get('neutral_rb',0):+.2f})"
                sub_col = (185, 215, 235)
            draw.text((tx + 3, ty + 1), sub_lbl[:36], fill=sub_col)
            tile_img = Image.fromarray(u8)
            sheet.paste(tile_img, (tx, ty + sub_h))

    sheet.save(out_jpg, quality=86)
    page_paths = [out_jpg]
    if len(clusters) > 16:
        page_rows = 7
        page_h = page_rows * block_h + 8
        total_pages = (rows + page_rows - 1) // page_rows
        base_no_ext = os.path.splitext(out_jpg)[0]
        page_paths = []
        for p_idx in range(total_pages):
            y_start = p_idx * page_rows * block_h
            y_end = min(sheet.height, y_start + page_h)
            crop_img = sheet.crop((0, y_start, sheet.width, y_end))
            p_path = f"{base_no_ext}_p{p_idx + 1}.jpg"
            crop_img.save(p_path, quality=88)
            page_paths.append(p_path)
    return page_paths


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Run segment-level source_timerange diagnostics, auto model detection & Ground-Truth scene mapping.")
    parser.add_argument("project", help="Project ID (e.g. 1030, '1106 (1)') or full path.")
    parser.add_argument("--out", default="", help="Optional output JSON path to save segment diagnostics.")
    args = parser.parse_args()

    proj_dir = resolve_project_dir(args.project)
    root_path = os.path.join(proj_dir, "draft_info.json")
    with open(root_path, "r", encoding="utf-8") as f:
        root_draft = json.load(f)

    all_seg_tuples = collect_all_video_segments(root_draft)
    project_model, clip_model_by_path, model_stats = detect_source_models(all_seg_tuples, proj_dir)
    print(
        f"Auto-Detected Video Model: {project_model.upper()} "
        f"(unique sources={model_stats['total_unique_sources']}, >15s={model_stats['gt_15s_count']}, "
        f"10-15s={model_stats['10_to_15s_count']}, <=10s={model_stats['le_10s_count']}, buckets={model_stats['duration_buckets_sec']})"
    )

    video_paths = list({resolve_video_path(v.get("path", ""), proj_dir) for _, _, v, _ in all_seg_tuples if v.get("path")})
    clip_nums = [extract_clip_number(v.get("material_name") or os.path.basename(v.get("path", ""))) for _, _, v, _ in all_seg_tuples]
    expected_max_clip = max([c for c in clip_nums if c > 0] or [0])

    clip_bg_map, gt_path, bg_context = discover_ground_truth_scene_map(
        video_paths, expected_max_clip=expected_max_clip, project_model=project_model
    )
    if gt_path:
        print(f"Found Ground-Truth Scene Map: {gt_path} ({len(clip_bg_map)} clips mapped to BG locations)")
    else:
        print("WARNING: No *BANGTRA*.txt / sf-board.json found in video folders — using Visual Fallback clustering.")

    print(f"Project '{os.path.basename(proj_dir)}': Measuring {len(all_seg_tuples)} child video segments at exact source_timerange...")
    unique_jobs = {}
    for _, seg, v, _ in all_seg_tuples:
        vpath = resolve_video_path(v.get("path", ""), proj_dir)
        st_sec = seg.get("source_timerange", {}).get("start", 0) / 1e6
        dur_sec = seg.get("source_timerange", {}).get("duration", 5000000) / 1e6
        ck = (vpath, round(st_sec, 1), round(dur_sec, 1))
        if ck not in unique_jobs and os.path.exists(vpath):
            unique_jobs[ck] = (vpath, st_sec, dur_sec)

    cache_by_range = {}
    frame_by_range = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {ck: pool.submit(extract_segment_metrics, vp, st, dur, True) for ck, (vp, st, dur) in unique_jobs.items()}
        for ck, fut in futs.items():
            m_res, u8_res = fut.result()
            cache_by_range[ck] = m_res
            frame_by_range[ck] = u8_res

    seg_records = []
    frames_list = []
    for idx, (_, seg, v, container_lbl) in enumerate(all_seg_tuples):
        vpath = resolve_video_path(v.get("path", ""), proj_dir)
        mname = v.get("material_name") or os.path.basename(vpath)
        st_sec = seg.get("source_timerange", {}).get("start", 0) / 1e6
        dur_sec = seg.get("source_timerange", {}).get("duration", 5000000) / 1e6
        src_dur = round(float(v.get("duration", 0)) / 1e6, 2)
        cache_key = (vpath, round(st_sec, 1), round(dur_sec, 1))
        m = cache_by_range.get(cache_key, {})
        frames_list.append(frame_by_range.get(cache_key, np.zeros((122, 216, 3), dtype=np.uint8)))
        sc_obj = seg.get("clip", {}).get("scale", {}) if isinstance(seg.get("clip"), dict) else {}
        seg_scale = max(float(sc_obj.get("x", 1.0)), float(sc_obj.get("y", 1.0)))
        rec = {
            "seg_idx": idx,
            "seg_id": seg.get("id", ""),
            "container": container_lbl,
            "material_name": mname,
            "source_dur_sec": src_dur,
            "clip_model": clip_model_by_path.get(vpath, project_model),
            "start_sec": round(st_sec, 2),
            "dur_sec": round(dur_sec, 2),
            "scale": round(seg_scale, 3),
            "metrics": m,
        }
        # Top-level metric aliases so s["mid"] and s["metrics"]["mid"] both work without KeyError
        for mk in ("mid", "p5", "p25", "p50", "p75", "p95", "contrast", "rb", "neutral_rb", "gm", "sat", "skin_rb", "skin_sat"):
            if mk in m:
                rec[mk] = m[mk]
        seg_records.append(rec)

    frames_arr = np.stack(frames_list, axis=0) if frames_list else np.zeros((0, 122, 216, 3), dtype=np.uint8)
    frames_cache_path = get_default_frames_cache_path(proj_dir)
    try:
        np.save(frames_cache_path, frames_arr)
    except Exception:
        pass

    vals = [r["metrics"] for r in seg_records if r["metrics"]]
    if vals:
        print("\n=== EMPIRICAL SEGMENT METRICS SUMMARY (Medians & P10..P90) ===")
        for k in ["p5", "p25", "mid", "p95", "contrast", "rb", "neutral_rb", "gm", "sat", "hsv_s", "skin_rb", "skin_sat", "sharp"]:
            arr = [x.get(k, 0.0) for x in vals]
            print(f"  {k:10s}: median={np.median(arr):+.4f} | p10={np.percentile(arr, 10):+.3f} .. p90={np.percentile(arr, 90):+.3f}")

    clusters = cluster_timeline_scenes(seg_records, clip_bg_map, bg_context)

    preview_jpg = f"/tmp/{os.path.basename(proj_dir).replace(' ', '_')}_raw_scenes.jpg"
    preview_pages = [preview_jpg]
    try:
        preview_pages = render_raw_scene_contact_sheet(all_seg_tuples, seg_records, clusters, preview_jpg, proj_dir, frames_arr=frames_arr)
    except Exception as exc:
        print(f"Note: Could not render raw scene preview sheet ({exc})")

    report_path, plan_path = generate_scene_report_and_starter_plan(
        proj_dir, project_model, gt_path, clusters, seg_records, preview_pages
    )

    print(f"\n=== SCENE CLUSTERS ({len(clusters)} scenes, Mode={clusters[0]['source_mode'] if clusters else 'NONE'}) ===")
    for c in clusters:
        a = c["anchor"]
        sc_t = (c.get("scene_title") or "")[:52]
        print(
            f"  Scene[{c['cluster_id']:2d}] ({c['bg_code']:12s} | {c.get('regime','AUTO'):14s}) Segs {c['start_seg']:3d}..{c['end_seg']:3d} ({len(c['seg_indices']):2d} segs) | "
            f"Anchor: Mid={a['mid']:.3f} RB={a['rb']:+.3f} NeutRB={a.get('neutral_rb',0.0):+.3f} SkinRB={a['skin_rb']:.3f} | "
            f"MaxOutlier=#{c['outlier_seg']:3d} (dMid={c['outlier_delta_mid']:+.3f} dNRB={c.get('outlier_delta_neutral_rb',0.0):+.3f}) | {sc_t}"
        )

    print(f"\n[MANDATORY AI COLORIST STEP 1] Rendered Multi-Shot Scene Strip sheet: {preview_jpg}")
    if len(preview_pages) > 1:
        print(f"-> High-Res Paginated Scene Sheets ({len(preview_pages)} pages): {', '.join(preview_pages)}")
    print(f"-> Full Scene & Per-Segment Scope Report (with Clean 35mm LUT Table): {report_path}")
    print(f"-> Pre-Populated Starter AI Grading Plan JSON: {plan_path}")
    print("-> AI Colorist MUST call read_file on the Scene Strip sheets AND the Scene Report (NEVER run python3 one-liners!)")

    payload = {
        "project_model": project_model,
        "model_stats": model_stats,
        "ground_truth_source": gt_path,
        "raw_preview_sheet": preview_jpg,
        "raw_preview_pages": preview_pages,
        "scene_report_path": report_path,
        "starter_plan_path": plan_path,
        "clip_bg_map": {str(k): v for k, v in clip_bg_map.items()},
        "bg_context": bg_context,
        "clusters": clusters,
        "scenes": clusters,
        "scene_clusters": clusters,
        "segments": seg_records,
    }
    default_cache = get_default_metrics_cache_path(proj_dir)
    for out_p in ({default_cache, args.out} - {""}):
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Completed analysis in {time.time() - t0:.1f}s (cached to {args.out or default_cache})")


if __name__ == "__main__":
    main()
