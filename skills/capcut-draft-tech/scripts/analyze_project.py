#!/usr/bin/env python3
"""Segment-level (source_timerange) 16-metric video diagnostics & Ground-Truth BANGTRA/Script Scene Mapping for CapCut projects."""
import argparse
import glob
import json
import os
import re
import subprocess
import numpy as np

CAPCUT_PROJECTS_ROOT = "/Users/daho100/Movies/CapCut/User Data/Projects/com.lveditor.draft"


def extract_segment_metrics(video_path: str, start_sec: float = 0.0, dur_sec: float = 5.0) -> dict:
    """
    Samples up to 3 frames strictly inside the segment's [start_sec, start_sec + dur_sec] window
    on the timeline so camera-angle cuts inside a multi-shot clip are measured accurately.
    """
    if dur_sec <= 0.2:
        sample_times = [max(0.0, start_sec + 0.5 * dur_sec)]
    elif dur_sec <= 2.5:
        sample_times = [max(0.0, start_sec + 0.35 * dur_sec), max(0.0, start_sec + 0.70 * dur_sec)]
    else:
        sample_times = [
            max(0.0, start_sec + 0.20 * dur_sec),
            max(0.0, start_sec + 0.50 * dur_sec),
            max(0.0, start_sec + 0.80 * dur_sec),
        ]

    frames = []
    for ts in sample_times:
        cmd = [
            "ffmpeg", "-v", "error", "-ss", f"{ts:.2f}", "-i", video_path,
            "-vf", "scale=160:90", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
        ]
        raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        if len(raw) >= 160 * 90 * 3:
            arr = np.frombuffer(raw[:160 * 90 * 3], dtype=np.uint8).reshape((90, 160, 3)).astype(np.float32) / 255.0
            frames.append(arr)

    if not frames:
        cmd = [
            "ffmpeg", "-v", "error", "-ss", "0.1", "-i", video_path,
            "-vf", "scale=160:90", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
        ]
        raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        if len(raw) >= 160 * 90 * 3:
            frames.append(np.frombuffer(raw[:160 * 90 * 3], dtype=np.uint8).reshape((90, 160, 3)).astype(np.float32) / 255.0)

    if not frames:
        return {
            "p1": 0.02, "p5": 0.05, "p25": 0.14, "mid": 0.26, "p50": 0.25, "p75": 0.38, "p95": 0.60, "p99": 0.74,
            "contrast": 0.55, "iqr_ratio": 0.44, "rb": 0.12, "shd_rb": 0.06, "mid_rb": 0.13, "hi_rb": 0.15,
            "gm": -0.012, "sat": 0.15, "hsv_s": 0.45, "sharp": 0.040, "hf_noise": 0.028
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
        hsv_s = float(np.mean(chroma / np.maximum(cmax, 1e-4)))
        gx = np.abs(np.diff(luma, axis=1)).mean()
        gy = np.abs(np.diff(luma, axis=0)).mean()
        sharp = float(0.5 * (gx + gy))
        blur = 0.25 * (luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:])
        hf_noise = float(np.std(luma[1:-1, 1:-1] - blur))
        per_frame.append({
            "p1": p1, "p5": p5, "p25": p25, "mid": mid, "p50": p50, "p75": p75, "p95": p95, "p99": p99,
            "contrast": contrast, "iqr_ratio": iqr_ratio, "rb": rb, "shd_rb": shd_rb, "mid_rb": mid_rb,
            "hi_rb": hi_rb, "gm": gm, "sat": sat, "hsv_s": hsv_s, "sharp": sharp, "hf_noise": hf_noise
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


def discover_ground_truth_scene_map(video_paths: list) -> tuple[dict, str]:
    """
    Searches the video folders and parent folders for ground-truth script files:
    - sf-board.json (SF Board .project)
    - *BANGTRA*.txt
    - *PROMPT*.txt
    Returns (clip_num_to_base_bg_dict, source_file_path).
    Crucially normalizes camera angles of the same room (e.g. BG1a and BG1b -> BG1)
    so reverse angles in the same location share one unified Scene Target!
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
                distinct_variants = set()
                sf_lookup = {}
                for sc in data.get("scenes", []):
                    for sf in sc.get("sfs", []):
                        rid = sf.get("id", "")
                        if rid:
                            sf_lookup[rid] = sf
                        rdesc = (sf.get("name") or sf.get("title") or sf.get("desc") or "").lower()
                        if re.match(r"^(BG|IMG)\d+[b-z]$", rid, re.IGNORECASE):
                            if any(kw in rdesc for kw in ("đêm", "tối", "night", "evening", "dusk", "veranda", "ngoài")):
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
                    for sf in list(sc.get("sfs", [])) + list(sc.get("shots", [])):
                        cid = sf.get("id", "")
                        if not cid.upper().startswith("V-"):
                            continue
                        raw_bg = sf.get("sf") or (sf.get("refs") or {}).get("bg") or ((sf.get("refs") or {}).get("bgs") or [""])[0]
                        base_bg = _resolve_sfb_bg(raw_bg, sc_id)
                        ckey = extract_clip_key(cid)
                        clip_map[ckey] = f"{sc_id}:{base_bg}"
                        clip_map[str(ckey)] = f"{sc_id}:{base_bg}"
                if clip_map:
                    return clip_map, sfb
            except Exception:
                pass

    # 2. Check *BANGTRA*.txt or *PROMPT*.txt
    candidates = []
    for sdir in ordered_dirs:
        candidates.extend(sorted(glob.glob(os.path.join(sdir, "*BANGTRA*.txt"))))
        candidates.extend(sorted(glob.glob(os.path.join(sdir, "*PROMPT*.txt"))))

    for cfile in candidates:
        if "ANH_REF" in os.path.basename(cfile):
            continue
        try:
            lines = open(cfile, "r", encoding="utf-8", errors="ignore").readlines()
            clip_map = {}
            era_idx = 0
            era_label = "E0"
            for line in lines:
                line_s = line.strip()
                if line_s.startswith("----") and line_s.endswith("----"):
                    era_idx += 1
                    era_label = f"E{era_idx}"
                    continue
                if "THẺ CHỮ" in line_s or "THE CHU" in line_s.upper():
                    era_idx += 1
                    m_tc = re.search(r"THẺ CHỮ\s*:\s*([^|]+)", line_s, re.IGNORECASE)
                    tc_short = re.sub(r"[^A-Z0-9]+", "_", m_tc.group(1).strip().upper())[:14].strip("_") if m_tc else f"E{era_idx}"
                    era_label = f"E{era_idx}_{tc_short}"
                    continue
                # Match table row: "001 | 30 | 9 | BG1a, CHAR1a..."
                m_row = re.match(r"^(\d{1,3})\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([^|]+)\|", line_s)
                if m_row:
                    cnum = int(m_row.group(1))
                    refs_str = m_row.group(2)
                    m_bg = re.search(r"(BG\d+)([a-z])?", refs_str)
                    if m_bg:
                        base_b, suf_b = m_bg.group(1), (m_bg.group(2) or "").lower()
                        bg_norm = f"{base_b}C" if suf_b == "c" else base_b
                        clip_map[cnum] = f"{era_label}:{bg_norm}"
                    else:
                        clip_map[cnum] = ""
                    continue
                # Match prompt line: "Scene 001: ... BG1a ..."
                m_pr = re.match(r"^Scene\s+(\d{1,3})\s*:(.*)$", line_s, re.IGNORECASE)
                if m_pr:
                    cnum = int(m_pr.group(1))
                    body = m_pr.group(2)
                    if "THẺ CHỮ" in body or "title card" in body.lower():
                        era_idx += 1
                        era_label = f"E{era_idx}"
                        continue
                    m_bg = re.search(r"(BG\d+)([a-z])?", body)
                    if m_bg:
                        base_b, suf_b = m_bg.group(1), (m_bg.group(2) or "").lower()
                        bg_norm = f"{base_b}C" if suf_b == "c" else base_b
                        clip_map[cnum] = f"{era_label}:{bg_norm}"
            if len(clip_map) >= 10:
                return clip_map, cfile
        except Exception:
            pass

    return {}, ""


def cluster_timeline_scenes(seg_records: list, clip_bg_map: dict = None) -> list:
    """
    Groups consecutive timeline segments into Continuous Scene Clusters:
    - Priority 1 (Ground-Truth): Uses clip_bg_map from BANGTRA.txt / sf-board.json (merging reverse angles BG1a/BG1b -> BG1).
      Transition clips without a BG tag inherit the current scene's BG if surrounded by the same BG.
    - Priority 2 (Visual Fallback): Uses a sustained multi-segment window only if no ground-truth map exists.
    """
    n = len(seg_records)
    if n == 0:
        return []

    default_m = {
        "p5": 0.05, "p25": 0.15, "mid": 0.28, "p95": 0.62, "p99": 0.76,
        "contrast": 0.55, "rb": 0.13, "gm": -0.012, "sat": 0.15, "hsv_s": 0.45,
        "sharp": 0.042, "hf_noise": 0.030
    }
    ms = [r.get("metrics") or default_m for r in seg_records]

    if clip_bg_map:
        # Assign each segment its ground-truth base BG code
        raw_bgs = []
        for idx_r, r in enumerate(seg_records):
            mname = r["material_name"]
            ckey = extract_clip_key(mname)
            if idx_r == 0 and (mname.startswith("ord_") or (ckey == -1 and r.get("dur_sec", 0) >= 20.0)):
                raw_bgs.append("SEEDANCE_HOOK:BG1")
                continue
            raw_bgs.append(clip_bg_map.get(ckey) or clip_bg_map.get(str(ckey), ""))

        # Fill empty/transition clips from nearest known BG
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
        clusters.append({
            "cluster_id": b_idx,
            "bg_code": raw_bgs[s_idx],
            "source_mode": source_mode,
            "start_seg": s_idx,
            "end_seg": e_idx - 1,
            "seg_indices": list(range(s_idx, e_idx)),
            "clips": sorted(set(seg_records[j]["material_name"] for j in range(s_idx, e_idx))),
            "anchor": anchor,
        })

    return clusters


def collect_all_video_segments(root_draft: dict) -> list[tuple[dict, dict, dict, str]]:
    """
    Collect EVERY child video segment (`video con`) across all Compound Clips (`materials.drafts`,
    including multi-level nested Compound Clips) AND any direct video segments on the root timeline
    (`root_draft.tracks`), sorted in exact chronological timeline order.
    Returns list of tuples: (draft_obj, seg_obj, video_material_obj, container_label).
    """
    timed_items = []
    visited_drafts = set()
    root_nested = {nd["id"]: nd for nd in root_draft.get("materials", {}).get("drafts", []) if isinstance(nd, dict) and "draft" in nd}

    def _scan_draft(d_obj: dict, label: str, offset_us: int = 0):
        vids = {v["id"]: v for v in d_obj.get("materials", {}).get("videos", [])}
        local_nested = {nd["id"]: nd for nd in d_obj.get("materials", {}).get("drafts", []) if isinstance(nd, dict) and "draft" in nd}
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


def extract_rgb_frame(video_path: str, ts_sec: float, w: int = 320, h: int = 180) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{max(0.0, ts_sec):.2f}", "-i", video_path,
        "-vf", f"scale={w}:{h}", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
    ]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    if len(raw) >= w * h * 3:
        return np.frombuffer(raw[:w * h * 3], dtype=np.uint8).reshape((h, w, 3)).astype(np.float32) / 255.0
    return np.zeros((h, w, 3), dtype=np.float32)


def resolve_video_path(vpath: str, proj_dir: str) -> str:
    if not vpath:
        return ""
    if vpath.startswith("##_draftpath_placeholder_"):
        vpath = re.sub(r"^##_draftpath_placeholder_[^#]+_##", proj_dir, vpath)
    return vpath


def render_raw_scene_contact_sheet(all_seg_tuples: list, seg_records: list, clusters: list, out_jpg: str, proj_dir: str = ""):
    from PIL import Image, ImageDraw
    w, h, pad_top = 320, 180, 26
    cols = 4
    rows = max(1, (len(clusters) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * w, rows * (h + pad_top)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)

    for idx, c in enumerate(clusters):
        mid_seg = c["seg_indices"][len(c["seg_indices"]) // 2]
        _, _, v, _ = all_seg_tuples[mid_seg]
        rec = seg_records[mid_seg]
        ts = rec["start_sec"] + 0.5 * rec["dur_sec"]
        vp = resolve_video_path(v.get("path", ""), proj_dir)
        rgb = extract_rgb_frame(vp, ts, w, h)
        tile = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
        r_i, c_i = divmod(idx, cols)
        x0, y0 = c_i * w, r_i * (h + pad_top)
        sheet.paste(tile, (x0, y0 + pad_top))
        a = c["anchor"]
        lbl = f"S{c['cluster_id']:02d} ({c['bg_code'][:12]}) Seg {c['start_seg']}..{c['end_seg']} | Mid:{a['mid']:.2f} RB:{a['rb']:+.2f}"
        draw.text((x0 + 6, y0 + 6), lbl, fill=(240, 220, 140))

    sheet.save(out_jpg, quality=88)


def resolve_project_dir(project_arg: str) -> str:
    if os.path.isdir(project_arg):
        return os.path.abspath(project_arg)
    candidate = os.path.join(CAPCUT_PROJECTS_ROOT, project_arg)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(f"CapCut project not found: {project_arg}")


def main():
    parser = argparse.ArgumentParser(description="Run segment-level source_timerange diagnostics & Ground-Truth scene mapping.")
    parser.add_argument("project", help="Project ID (e.g. 1030, '1026 - THU SKILL MOI') or full path.")
    parser.add_argument("--out", default="", help="Optional output JSON path to save segment diagnostics.")
    args = parser.parse_args()

    proj_dir = resolve_project_dir(args.project)
    root_path = os.path.join(proj_dir, "draft_info.json")
    with open(root_path, "r", encoding="utf-8") as f:
        root_draft = json.load(f)

    all_seg_tuples = collect_all_video_segments(root_draft)
    video_paths = list({resolve_video_path(v.get("path", ""), proj_dir) for _, _, v, _ in all_seg_tuples if v.get("path")})
    clip_bg_map, gt_path = discover_ground_truth_scene_map(video_paths)
    if gt_path:
        print(f"Found Ground-Truth Scene Map: {gt_path} ({len(clip_bg_map)} clips mapped to BG locations)")
    else:
        print("WARNING: No *BANGTRA*.txt / sf-board.json found in video folders — using Visual Fallback clustering (confirm with user!).")

    print(f"Project '{os.path.basename(proj_dir)}': Measuring {len(all_seg_tuples)} child video segments (video con) at exact source_timerange...")
    from concurrent.futures import ThreadPoolExecutor
    unique_jobs = {}
    for _, seg, v, _ in all_seg_tuples:
        vpath = resolve_video_path(v.get("path", ""), proj_dir)
        st_sec = seg.get("source_timerange", {}).get("start", 0) / 1e6
        dur_sec = seg.get("source_timerange", {}).get("duration", 5000000) / 1e6
        ck = (vpath, round(st_sec, 1), round(dur_sec, 1))
        if ck not in unique_jobs and os.path.exists(vpath):
            unique_jobs[ck] = (vpath, st_sec, dur_sec)

    cache_by_range = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {ck: pool.submit(extract_segment_metrics, vp, st, dur) for ck, (vp, st, dur) in unique_jobs.items()}
        for ck, fut in futs.items():
            cache_by_range[ck] = fut.result()

    seg_records = []
    for idx, (_, seg, v, container_lbl) in enumerate(all_seg_tuples):
        vpath = resolve_video_path(v.get("path", ""), proj_dir)
        mname = v.get("material_name") or os.path.basename(vpath)
        st_sec = seg.get("source_timerange", {}).get("start", 0) / 1e6
        dur_sec = seg.get("source_timerange", {}).get("duration", 5000000) / 1e6
        cache_key = (vpath, round(st_sec, 1), round(dur_sec, 1))
        m = cache_by_range.get(cache_key, {})
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
            "metrics": m,
        })

    vals = [r["metrics"] for r in seg_records if r["metrics"]]
    if vals:
        print("\n=== EMPIRICAL SEGMENT METRICS SUMMARY (Medians & P10..P90) ===")
        for k in ["p5", "p25", "mid", "p95", "contrast", "rb", "gm", "sat", "hsv_s", "sharp", "hf_noise"]:
            arr = [x[k] for x in vals]
            print(f"  {k:10s}: median={np.median(arr):+.4f} | p10={np.percentile(arr, 10):+.3f} .. p90={np.percentile(arr, 90):+.3f}")

    clusters = cluster_timeline_scenes(seg_records, clip_bg_map)
    print(f"\n=== SCENE CLUSTERS ({len(clusters)} scenes, Mode={clusters[0]['source_mode'] if clusters else 'NONE'}) ===")
    angle_jumps = []
    for c in clusters:
        a = c["anchor"]
        print(
            f"  Scene[{c['cluster_id']:2d}] ({c['bg_code']:8s}) Segs {c['start_seg']:3d}..{c['end_seg']:3d} ({len(c['seg_indices']):2d} segs) | "
            f"Anchor: P5={a['p5']:.3f} Mid={a['mid']:.3f} R-B={a['rb']:+.3f} G-M={a['gm']:+.3f} Sat={a['sat']:.3f}"
        )
        for s_idx in c["seg_indices"]:
            r = seg_records[s_idx]
            m = r["metrics"]
            if not m:
                continue
            d_rb = m["rb"] - a["rb"]
            d_mid = m["mid"] - a["mid"]
            if "(" in r["material_name"] or "_" in r["material_name"] or abs(d_rb) > 0.042 or abs(d_mid) > 0.060:
                angle_jumps.append((s_idx, c["cluster_id"], c["bg_code"], r["material_name"], r["start_sec"], round(d_rb, 3), round(d_mid, 3), m["mid"], m["rb"]))

    if angle_jumps:
        print(f"\n=== DETECTED INTRA-SCENE CAMERA ANGLE JUMPS / VARIANT SEGMENTS ({len(angle_jumps)} segments) ===")
        for s_idx, cid, bgc, mname, st_s, d_rb, d_mid, mid_v, rb_v in angle_jumps[:25]:
            print(
                f"  Seg[{s_idx:3d}] (Scene {cid:2d} {bgc}) {mname:14s} @{st_s:5.1f}s | "
                f"Mid={mid_v:.3f} (dMid={d_mid:+.3f}) | R-B={rb_v:+.3f} (dRB={d_rb:+.3f})"
            )

    preview_jpg = f"/tmp/{os.path.basename(proj_dir).replace(' ', '_')}_raw_scenes.jpg"
    try:
        render_raw_scene_contact_sheet(all_seg_tuples, seg_records, clusters, preview_jpg, proj_dir)
        print(f"\n[MANDATORY AI VISUAL CHECK] Rendered raw scene contact sheet: {preview_jpg}")
        print("-> AI MUST call read_file / view_file on this image to visually inspect the raw scene colors & lighting before grading!")
    except Exception as exc:
        print(f"Note: Could not render raw scene preview sheet ({exc})")

    if args.out:
        payload = {
            "ground_truth_source": gt_path,
            "raw_preview_sheet": preview_jpg,
            "clip_bg_map": {str(k): v for k, v in clip_bg_map.items()},
            "segments": seg_records,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved {len(seg_records)} segment-level diagnostics to {args.out}")
        print("REMINDER: (1) View the raw scene sheet image with read_file/view_file, (2) Ask the user Model Source & Emotional Direction before running apply_grade.py!")


if __name__ == "__main__":
    main()
