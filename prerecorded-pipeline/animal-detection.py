#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVP pipeline:
Video/Stream → 1 FPS frames → SpeciesNet (det+cls via CLI) → JSONL → Debug JPGs → MP4 segments

Per-run isolation:
- Each execution writes to out_mvp/run_###/
  e.g., out_mvp/run_001/{speciesnet_predictions.json, results.jsonl, debug/, segments/}

Requirements in your venv:
  pip install speciesnet opencv-python tqdm
"""

import re
import sys
import json
import tempfile
import subprocess
from pathlib import Path

import cv2
from tqdm import tqdm

# ========== CONFIG ==========
VIDEO_SRC = "/Users/brightonyoung/ML-Projects/Vision-ML/input-videos/Short-Trail-Cam.mp4"
OUT_ROOT  = Path("./out_mvp")
ANALYSIS_FPS = 1.0   # process 1 frame per second
COUNTRY = "USA"      # optional; set None to skip geofilter
CONF_DET_THRESH = 0.20  # ignore very low-confidence detections
END_GAP_SECONDS = 2      # end a segment if no animal for >= this many seconds

# ========== RUN DIR MANAGEMENT ==========
def _next_run_dir(root: Path) -> Path:
    """
    Find the next run directory name (run_001, run_002, ...).
    """
    root.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r"^run_(\d{3})$")
    max_n = 0
    for p in root.iterdir():
        if p.is_dir():
            m = pat.match(p.name)
            if m:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
    next_n = max_n + 1
    return root / f"run_{next_n:03d}"

def init_run_dirs():
    """
    Create the per-run directory structure and return useful paths.
    """
    run_dir = _next_run_dir(OUT_ROOT)
    debug_dir = run_dir / "debug"
    seg_dir   = run_dir / "segments"
    debug_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    pred_json = run_dir / "speciesnet_predictions.json"
    jsonl_path = run_dir / "results.jsonl"
    return run_dir, debug_dir, seg_dir, pred_json, jsonl_path

# ========== HELPERS ==========
def timestamp_from_frame(frame_idx, fps):
    return (frame_idx / fps) if fps and fps > 0 else 0.0

def _extract_common_name(label: str | None) -> str | None:
    """
    SpeciesNet labels look like:
      '<uuid>;...;genus;species;common name'
    We want just the human-friendly common name (last non-empty segment).
    """
    if not label:
        return None
    parts = [p.strip() for p in label.split(";") if p.strip()]
    if not parts:
        return None
    common = parts[-1].replace("_", " ").strip()
    if common.lower() == "blank":
        return None
    return common.lower()

def _format_species_label(label: str | None, score: float | None) -> str:
    """
    Returns 'species | 0.98' (lowercase). Empty string if missing.
    """
    if label is None or score is None:
        return ""
    name = _extract_common_name(label)
    if not name:
        return ""
    return f"{name} | {score:.2f}"

def draw_annotations(img_bgr, detections, species_label=None, species_score=None):
    """
    detections: list of dicts with normalized bbox [x,y,w,h], 'conf'
    species_label/species_score: top-1 label/score for the *frame* (current MVP)
    """
    h, w = img_bgr.shape[:2]
    color = (0, 255, 0)
    text = _format_species_label(species_label, species_score)
    for det in detections:
        x, y, bw, bh = det["bbox"]
        x1, y1 = int(x * w), int(y * h)
        x2, y2 = int((x + bw) * w), int((y + bh) * h)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        if text:
            cv2.putText(
                img_bgr, text, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
            )
    return img_bgr

def open_segment_writer(path, fps, size_wh):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, size_wh)

# ========== STEP 1: EXTRACT FRAMES @1 FPS ==========
def extract_frames(video_path, out_dir, analysis_fps=1.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Step interval in frames
    step = max(1, int(round(src_fps / analysis_fps))) if analysis_fps > 0 else int(src_fps)
    saved = []

    pbar = tqdm(total=total, desc="Extracting")
    frame_idx = 0
    sample_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            ts_sec = timestamp_from_frame(frame_idx, src_fps)
            out_path = out_dir / f"frame_{sample_idx:06d}_t{ts_sec:.2f}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append((frame_idx, ts_sec, out_path))
            sample_idx += 1
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return saved  # list of (frame_idx, ts_sec, jpg_path)

# ========== STEP 2: RUN SPECIESNET CLI ==========
def run_speciesnet_on_folder(folder, out_json, country=None):
    cmd = [
        sys.executable, "-m", "speciesnet.scripts.run_model",
        "--folders", str(folder),
        "--predictions_json", str(out_json)
    ]
    if country:
        cmd += ["--country", country]
    print("[SpeciesNet] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# ========== STEP 3: READ SPECIESNET OUTPUT ==========
def load_speciesnet_predictions(pred_json):
    with open(pred_json, "r") as f:
        data = json.load(f)
    # SpeciesNet outputs either {"predictions":[...]} or list; handle both
    if isinstance(data, dict) and "predictions" in data:
        return data["predictions"]
    if isinstance(data, list):
        return data
    raise ValueError("Unrecognized SpeciesNet output format")

# Helper: pick animal detections and top-1 class
def parse_prediction_entry(entry, conf_thresh=0.2):
    """
    Returns:
      animal_dets: list of {'bbox': [x,y,w,h], 'conf': float}
      top1_label, top1_score
    """
    animal_dets = []
    for det in entry.get("detections", []):
        # MegaDetector categories: "1"=animal, "2"=person, "3"=vehicle
        cat = str(det.get("category", ""))
        conf = float(det.get("conf", 0.0))
        if cat == "1" and conf >= conf_thresh:
            animal_dets.append({"bbox": det["bbox"], "conf": conf})

    top1_label, top1_score = None, None
    cls = entry.get("classifications") or {}
    classes = cls.get("classes", [])
    scores = cls.get("scores", [])
    if classes and scores and len(classes) == len(scores):
        top1_label = classes[0]
        top1_score = float(scores[0])

    # Some builds add a "prediction" and "prediction_score"
    if entry.get("prediction") and entry.get("prediction_score") is not None:
        top1_label = entry["prediction"]
        top1_score = float(entry["prediction_score"])

    return animal_dets, top1_label, top1_score

# ========== STEP 4: WRITE JSONL, DEBUG IMAGES, AND SEGMENTS ==========
def write_outputs_and_segments(extracted, preds, jsonl_path, debug_dir, seg_dir, end_gap_s):
    # Map: filename → prediction entry
    pred_by_file = { Path(p.get("filepath")).name: p for p in preds if p.get("filepath") }

    with open(jsonl_path, "w") as jsonl_f:
        in_segment = False
        seg_writer = None
        seg_index = 0
        last_animal_time = None

        for (frame_idx, ts_sec, jpg_path) in extracted:
            entry = pred_by_file.get(jpg_path.name)
            dets, top_label, top_score = [], None, None
            if entry:
                dets, top_label, top_score = parse_prediction_entry(entry, CONF_DET_THRESH)

            # JSONL line (one per processed frame)
            json_line = {
                "video": str(VIDEO_SRC),
                "frame_index": frame_idx,
                "t_seconds": ts_sec,
                "filepath": str(jpg_path),
                "detections": [{"bbox": d["bbox"], "conf": d["conf"], "category": "1", "label": "animal"} for d in dets],
                "classifications": entry.get("classifications") if entry else None,
                "prediction": top_label,
                "prediction_score": top_score
            }
            jsonl_f.write(json.dumps(json_line) + "\n")

            # Debug image
            img = cv2.imread(str(jpg_path))
            dbg = draw_annotations(img.copy(), dets, top_label, top_score)
            dbg_name = f"{jpg_path.stem}_debug.jpg"
            cv2.imwrite(str(debug_dir / dbg_name), dbg)

            # Segment logic (1 FPS segments in MVP)
            animal_here = len(dets) > 0
            if animal_here:
                last_animal_time = ts_sec
                if not in_segment:
                    in_segment = True
                    seg_index += 1
                    seg_path = seg_dir / f"segment_{seg_index:03d}.mp4"
                    seg_writer = open_segment_writer(seg_path, 1.0, (img.shape[1], img.shape[0]))
            else:
                if in_segment and last_animal_time is not None:
                    if (ts_sec - last_animal_time) >= end_gap_s:
                        in_segment = False
                        seg_writer.release()
                        seg_writer = None

            if in_segment and seg_writer is not None:
                seg_writer.write(dbg)

        if seg_writer is not None:
            seg_writer.release()

# ========== MAIN ==========
def main():
    # Make per-run dirs
    run_dir, DEBUG_DIR, SEG_DIR, PRED_JSON, JSONL_PATH = init_run_dirs()

    print(f"[Run] Output dir: {run_dir}")
    print(f"      Debug dir : {DEBUG_DIR}")
    print(f"      Segments  : {SEG_DIR}")
    print(f"      JSONL     : {JSONL_PATH}")
    print(f"      SpeciesNet: {PRED_JSON}")

    # Extract frames into a temp dir (not persisted)
    with tempfile.TemporaryDirectory(prefix="frames_") as tmpdir:
        frame_dir = Path(tmpdir)
        print(f"[Frames] Extracting to: {frame_dir}")
        extracted = extract_frames(VIDEO_SRC, frame_dir, ANALYSIS_FPS)

        # Run SpeciesNet CLI on extracted frames → writes PRED_JSON in run dir
        run_speciesnet_on_folder(frame_dir, PRED_JSON, country=COUNTRY)

        # Load SpeciesNet predictions
        preds = load_speciesnet_predictions(PRED_JSON)

        # Emit JSONL, debug images, and MP4 segments into this run dir
        write_outputs_and_segments(
            extracted=extracted,
            preds=preds,
            jsonl_path=JSONL_PATH,
            debug_dir=DEBUG_DIR,
            seg_dir=SEG_DIR,
            end_gap_s=END_GAP_SECONDS
        )

    print("\nDone.")
    print(f"- Run folder            : {run_dir}")
    print(f"- SpeciesNet JSON       : {PRED_JSON}")
    print(f"- Results JSONL         : {JSONL_PATH}")
    print(f"- Debug frames (JPGs)   : {DEBUG_DIR}")
    print(f"- Segments (MP4 @1 FPS) : {SEG_DIR}")

if __name__ == "__main__":
    main()
