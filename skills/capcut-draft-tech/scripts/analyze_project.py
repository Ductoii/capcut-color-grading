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


def extract_segment_metrics(video_path: str, start_sec: float = 0.0, dur_sec: float = 5.0) -> dict:
    """
    Samples up to 2 representative frames strictly inside [start_sec, start_sec + dur_sec]
    at 160x90 resolution so camera-angle cuts inside multi-shot clips are measured accurately and fast.
    """
    if dur_sec <= 1.8:
        sample_times = [max(0.0, start_sec + 0.50 * dur_sec)]
    else:
        sample_times = [
            max(0.0, start_sec + 0.28 * dur_sec),
            max(0.0, start_sec + 0.72 * dur_sec),
        ]

    frames = []
    for ts in sample_times:
        cmd = [
            "ffmpeg", "-nostdin", "-v", "error", "-ss", f"{ts:.2f}", "-i", video_path,
            "-vf", "scale=160:90", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
        ]
        raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        if len(raw) >= 160 * 90 * 3:
            arr = np.frombuffer(raw[:160 * 90 * 3], dtype=np.uint8).reshape((90, 160, 3)).astype(np.float32) / 255.0
            frames.append(arr)

    if not frames:
        cmd = [
            "ffmpeg", "-nostdin", "-v", "error", "-ss", "0.1", "-i", video_path,
            "-vf", "scale=160:90", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
        ]
        raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        if len(raw) >= 160 * 90 * 3:
            frames.append(np.frombuffer(raw[:160 * 90 * 3], dtype=np.uint8).reshape((90, 160, 3)).astype(np.float32) / 255.0)

    if not frames:
        return {
            "p1": 0.02, "p5": 0.05, "p25": 0.14, "mid": 0.28, "p50": 0.27, "p75": 0.40, "p95": 0.62, "p99": 0.75,
            "contrast": 0.57, "iqr_ratio": 0.45, "rb": 0.12, "shd_rb": 0.06, "mid_rb": 0.13, "hi_rb": 0.15,
            "gm": -0.012, "sat": 0.15, "hsv_s": 0.44, "sharp": 0.040, "hf_noise": 0.028,
            "skin_rb": 0.16, "skin_sat": 0.34, "skin_hi": 0.50
        }

    per_frame = []
    for arr in frames:
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        p1, p5, p25, p50, p75, p95, p99 = np.percentile(luma, [1, 5, 25, 50, 75, 95, 99])
        mid = float(np.mean(luma))
        contrast = float(p95 - p5)
        iqr_ratio = float((p75 - p25) / max(contrast, 1e-4))
        rb = float(np.mean(r - b))
        shd_rb = float(np.mean((r - b)[luma <= p25]))
        mid_rb = float(np.mean((r - b)[(luma > p25) & (luma < p75)]))
        hi_rb = float(np.mean((r - b)[luma >= p75]))
        gm = float(np.mean(g - 0.5 * (r + b)))
        cmax = np.max(arr, axis=2)
        cmin = np.min(arr, axis=2)
        chroma = cmax - cmin
        sat = float(np.mean(chroma))
        hsv_s_map = chroma / np.maximum(cmax, 1e-4)
        hsv_s = float(np.mean(hsv_s_map))

        # Central-frame human skin locus (excludes top sky / bottom floor and pure yellow/wood surfaces)
        rc, gc, bc = r[10:78, 16:144], g[10:78, 16:144], b[10:78, 16:144]
        lc = luma[10:78, 16:144]
        sc_hsv = hsv_s_map[10:78, 16:144]
        rb_c = rc - bc
        gb_ratio = (gc - bc) / np.maximum(rb_c, 1e-4)
        skin_mask = (rc > gc) & (gc > bc) & (rb_c > 0.060) & (gb_ratio >= 0.20) & (gb_ratio <= 0.68) & (lc > 0.16) & (lc < 0.88)
        if np.sum(skin_mask) >= 12:
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
        per_frame.append({
            "p1": p1, "p5": p5, "p25": p25, "mid": mid, "p50": p50, "p75": p75, "p95": p95, "p99": p99,
            "contrast": contrast, "iqr_ratio": iqr_ratio, "rb": rb, "shd_rb": shd_rb, "mid_rb": mid_rb,
            "hi_rb": hi_rb, "gm": gm, "sat": sat, "hsv_s": hsv_s, "sharp": sharp, "hf_noise": hf_noise,
            "skin_rb": skin_rb, "skin_sat": skin_sat, "skin_hi": skin_hi
        })

    keys = per_frame[0].keys()
    return {k: round(float(np.mean([f[k] for f in per_frame])), 4) for k in keys}


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


def classify_script_regime(text: str) -> tuple[str, bool]:
    """
    Infers the lighting/scene regime and flashback flag from ground-truth script descriptions
    (scene headers, BG descriptions, and clip prompts).
    """
    t = (text or "").lower()
    is_flashback = any(k in t for k in ("mười năm trước", "10 năm trước", "years ago", "sinh nhật 15", "quá khứ", "flashback", "hồi tưởng"))
    if any(k in t for k in ("bão tuyết", "mưa đêm", "đêm mưa", "blizzard", "storm", "rainy night", "/coldtone:", "ngoài trời đêm", "3 giờ sáng", "3:00 am")):
        return "NIGHT_COOL", is_flashback
    if any(k in t for k in ("server", "máy chủ", "hành lang kỹ thuật", "phòng khám", "clinic", "hospital", "bệnh viện", "kho đá", "icedock", "phòng thay đồ", "locker")):
        return "COOL_TECH", is_flashback
    if any(k in t for k in ("bình minh", "hoàng hôn", "chiều tà", "nắng chiều", "dawn", "sunrise", "sunset", "dusk", "golden hour", "gold", "5:45 pm", "6 giờ chiều", "chiều tối")):
        return "DAWN_DUSK_GOLD", is_flashback
    if any(k in t for k in ("đêm", "tối", "khuya", "nửa đêm", "night", "evening", "midnight", "/amberlook:", "tiệc tối", "8 giờ tối", "9 giờ tối")):
        return "NIGHT_WARM", is_flashback
    if any(k in t for k in ("bếp", "kitchen", "căn hộ", "apartment", "vườn", "garden", "công viên", "park", "thư phòng", "phòng ăn", "dining", "sảnh", "lobby", "foyer", "salon", "nhà")):
        return "WARM_INT", is_flashback
    if any(k in t for k in ("văn phòng", "office", "phòng họp", "meeting", "boardroom", "điều trần", "hearing", "sân đỗ", "nhà chứa", "hangar", "tarmac", "ban ngày", "sáng", "trưa")):
        return "DAY_CRISP", is_flashback
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
                sf_lookup = {}
                bg_desc_by_id = {}
                for sc in data.get("scenes", []):
                    for sf in sc.get("sfs", []):
                        rid = sf.get("id", "")
                        if rid:
                            sf_lookup[rid] = sf
                        rdesc = " ".join(str(sf.get(k) or "") for k in ("name", "title", "desc", "prompt")).strip()
                        if rid:
                            bg_desc_by_id[rid.upper()] = rdesc
                        if re.match(r"^(BG|IMG)\d+[b-z]$", rid, re.IGNORECASE):
                            rlow = rdesc.lower()
                            if any(kw in rlow for kw in ("đêm", "tối", "night", "evening", "dusk", "veranda", "ngoài", "hiên", "phòng cúp")):
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
                            if c_up in distinct_variants:
                                return c_up
                            return re.sub(r"[A-Z]$", "", c_up)
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
                        bg_desc = bg_desc_by_id.get(str(raw_bg).upper(), "") or bg_desc_by_id.get(base_bg, "")
                        combined_desc = f"{sc_title} | {bg_desc} | {shot_prompt[:180]}"
                        regime, is_fb = classify_script_regime(combined_desc)
                        bg_context[bg_tag] = {
                            "scene_title": sc_title,
                            "bg_desc": bg_desc[:140],
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
                        raw_desc = f"{current_hdr_text} | {ref_bg_descs.get(f'{base_b}{suf_b}', ref_bg_descs.get(base_b, ''))} | {tail_str[:120]}"
                        regime, is_fb = classify_script_regime(raw_desc)
                        if bg_tag not in bg_context or bg_context[bg_tag].get("regime") == "AUTO":
                            bg_context[bg_tag] = {
                                "scene_title": current_hdr_text or tail_str[:60],
                                "bg_desc": ref_bg_descs.get(f"{base_b}{suf_b}", ref_bg_descs.get(base_b, "")),
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
                        raw_desc = f"{current_hdr_text} | {ref_bg_descs.get(f'{base_b}{suf_b}', '')} | {body[:160]}"
                        regime, is_fb = classify_script_regime(raw_desc)
                        bg_context[bg_tag] = {
                            "scene_title": current_hdr_text,
                            "bg_desc": ref_bg_descs.get(f"{base_b}{suf_b}", ""),
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
        c_ms = ms[s_idx:e_idx]
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
        clusters.append({
            "cluster_id": b_idx,
            "bg_code": bg_code,
            "regime": regime,
            "is_flashback": bool(ctx.get("is_flashback", False)),
            "scene_title": ctx.get("scene_title", ""),
            "bg_desc": ctx.get("bg_desc", ""),
            "source_mode": source_mode,
            "start_seg": s_idx,
            "end_seg": e_idx - 1,
            "seg_indices": list(range(s_idx, e_idx)),
            "clips": sorted(set(seg_records[j]["material_name"] for j in range(s_idx, e_idx))),
            "anchor": anchor,
        })

    return clusters


def extract_rgb_frame(video_path: str, ts_sec: float, w: int = 320, h: int = 180) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(0.0, ts_sec):.2f}", "-i", video_path,
        "-vf", f"scale={w}:{h}", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
    ]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    if len(raw) >= w * h * 3:
        return np.frombuffer(raw[:w * h * 3], dtype=np.uint8).reshape((h, w, 3)).astype(np.float32) / 255.0
    return np.zeros((h, w, 3), dtype=np.float32)


def render_raw_scene_contact_sheet(all_seg_tuples: list, seg_records: list, clusters: list, out_jpg: str, proj_dir: str = ""):
    from PIL import Image, ImageDraw
    w, h, pad_top = 280, 158, 24
    cols = 4
    rows = max(1, (len(clusters) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * w, rows * (h + pad_top)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)

    def _fetch_tile(idx_c):
        idx, c = idx_c
        mid_seg = c["seg_indices"][len(c["seg_indices"]) // 2]
        _, _, v, _ = all_seg_tuples[mid_seg]
        rec = seg_records[mid_seg]
        ts = rec["start_sec"] + 0.5 * rec["dur_sec"]
        vp = resolve_video_path(v.get("path", ""), proj_dir)
        return idx, c, extract_rgb_frame(vp, ts, w, h)

    with ThreadPoolExecutor(max_workers=12) as pool:
        tiles = list(pool.map(_fetch_tile, enumerate(clusters)))

    for idx, c, rgb in tiles:
        tile = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
        r_i, c_i = divmod(idx, cols)
        x0, y0 = c_i * w, r_i * (h + pad_top)
        sheet.paste(tile, (x0, y0 + pad_top))
        a = c["anchor"]
        reg_lbl = c.get("regime", "")[:8]
        lbl = f"S{c['cluster_id']:02d} ({c['bg_code'][:12]}|{reg_lbl}) Mid:{a['mid']:.2f} RB:{a['rb']:+.2f}"
        draw.text((x0 + 5, y0 + 5), lbl, fill=(240, 220, 140))

    sheet.save(out_jpg, quality=84)


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
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {ck: pool.submit(extract_segment_metrics, vp, st, dur) for ck, (vp, st, dur) in unique_jobs.items()}
        for ck, fut in futs.items():
            cache_by_range[ck] = fut.result()

    seg_records = []
    for idx, (_, seg, v, container_lbl) in enumerate(all_seg_tuples):
        vpath = resolve_video_path(v.get("path", ""), proj_dir)
        mname = v.get("material_name") or os.path.basename(vpath)
        st_sec = seg.get("source_timerange", {}).get("start", 0) / 1e6
        dur_sec = seg.get("source_timerange", {}).get("duration", 5000000) / 1e6
        src_dur = round(float(v.get("duration", 0)) / 1e6, 2)
        cache_key = (vpath, round(st_sec, 1), round(dur_sec, 1))
        m = cache_by_range.get(cache_key, {})
        sc_obj = seg.get("clip", {}).get("scale", {}) if isinstance(seg.get("clip"), dict) else {}
        seg_scale = max(float(sc_obj.get("x", 1.0)), float(sc_obj.get("y", 1.0)))
        seg_records.append({
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
        })

    vals = [r["metrics"] for r in seg_records if r["metrics"]]
    if vals:
        print("\n=== EMPIRICAL SEGMENT METRICS SUMMARY (Medians & P10..P90) ===")
        for k in ["p5", "p25", "mid", "p95", "contrast", "rb", "gm", "sat", "hsv_s", "skin_rb", "skin_sat", "sharp"]:
            arr = [x.get(k, 0.0) for x in vals]
            print(f"  {k:10s}: median={np.median(arr):+.4f} | p10={np.percentile(arr, 10):+.3f} .. p90={np.percentile(arr, 90):+.3f}")

    clusters = cluster_timeline_scenes(seg_records, clip_bg_map, bg_context)
    print(f"\n=== SCENE CLUSTERS ({len(clusters)} scenes, Mode={clusters[0]['source_mode'] if clusters else 'NONE'}) ===")
    for c in clusters:
        a = c["anchor"]
        print(
            f"  Scene[{c['cluster_id']:2d}] ({c['bg_code']:18s} | {c.get('regime','AUTO'):14s}) Segs {c['start_seg']:3d}..{c['end_seg']:3d} ({len(c['seg_indices']):2d} segs) | "
            f"Anchor: P5={a['p5']:.3f} Mid={a['mid']:.3f} R-B={a['rb']:+.3f} G-M={a['gm']:+.3f} Sat={a['sat']:.3f}"
        )

    preview_jpg = f"/tmp/{os.path.basename(proj_dir).replace(' ', '_')}_raw_scenes.jpg"
    try:
        render_raw_scene_contact_sheet(all_seg_tuples, seg_records, clusters, preview_jpg, proj_dir)
        print(f"\nRendered raw scene contact sheet: {preview_jpg}")
    except Exception as exc:
        print(f"Note: Could not render raw scene preview sheet ({exc})")

    payload = {
        "project_model": project_model,
        "model_stats": model_stats,
        "ground_truth_source": gt_path,
        "raw_preview_sheet": preview_jpg,
        "clip_bg_map": {str(k): v for k, v in clip_bg_map.items()},
        "bg_context": bg_context,
        "segments": seg_records,
    }
    default_cache = get_default_metrics_cache_path(proj_dir)
    for out_p in ({default_cache, args.out} - {""}):
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Completed analysis in {time.time() - t0:.1f}s (cached to {args.out or default_cache})")


if __name__ == "__main__":
    main()
