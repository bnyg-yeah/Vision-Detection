#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lite pipeline: read freshest frame → detect/classify → save annotated debug JPGs → write MP4 segments.
Outputs go to ./out_lite (overwritten each run).
"""

import os
import re
import sys
import json
import time
import shutil
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from threading import Thread, Lock

import cv2
import numpy as np

# ===== CONFIG =====
VIDEO_SRC = ""  # file path or YouTube URL
RUN_DIR = Path("out_lite")
CONF_DET_THRESH = 0.50
COUNTRY = None                      # optional geo prior (e.g., "USA")
SEGMENT_FILENAME_FMT = "segment_{:03d}.mp4"
SEGMENT_WRITER_FPS = 1.5            # one annotated frame per inference
DEBUG_IMAGE_EVERY_INFERENCE = True

# Category-id → name map
CATEGORY_MAP = {"1": "animal", "2": "human", "3": "vehicle"}

# ===== OUTPUT SETUP =====
def init_run_dirs():
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR, ignore_errors=True)
    debug_dir = RUN_DIR / "debug"
    seg_dir = RUN_DIR / "segments"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    return RUN_DIR, debug_dir, seg_dir, RUN_DIR / "results.jsonl", RUN_DIR / "events.log"

# ===== UTILS =====
TARGET_W, TARGET_H = 1280, 720

def to_720p(bgr):
    return cv2.resize(bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

def _extract_common_name(label: str | None) -> str | None:
    if not label:
        return None
    parts = [p.strip() for p in label.split(";") if p.strip()]
    if not parts:
        return None
    common = parts[-1].replace("_", " ").strip()
    if common.lower() == "blank":
        return None
    return common

# ===== SPECIESNET CLI =====
def _speciesnet_supports_one_pass() -> bool:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "speciesnet.scripts.run_model", "-h"],
            stderr=subprocess.STDOUT
        ).decode(errors="ignore")
        if os.getenv("SPECIESNET_ONE_PASS", "").strip().lower() in ("1", "true", "yes"):
            return True
        return ("--det_and_cls" in out) or ("--det-and-cls" in out)
    except Exception:
        return False

def run_speciesnet_on_folder(folder, out_json, country=None, det_and_cls=False):
    cmd = [
        sys.executable, "-m", "speciesnet.scripts.run_model",
        "--folders", str(folder),
        "--predictions_json", str(out_json),
    ]
    if country:
        cmd += ["--country", country]
    if det_and_cls and _speciesnet_supports_one_pass():
        cmd += ["--det_and_cls"]
    subprocess.run(cmd, check=True)

def load_speciesnet_predictions(pred_json):
    with open(pred_json, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "predictions" in data:
        return data["predictions"]
    if isinstance(data, list):
        return data
    raise ValueError("Unrecognized SpeciesNet output format")

def classify_animal_crops(frame_bgr, animal_dets, country=None, pad_px=4):
    if not animal_dets:
        return []
    h, w = frame_bgr.shape[:2]
    crop_dir = Path(tempfile.mkdtemp(prefix="animal_crops_"))
    crop_paths = []
    for i, det in enumerate(animal_dets):
        x, y, bw, bh = det["bbox"]
        x1, y1 = int(x * w), int(y * h)
        x2, y2 = int((x + bw) * w), int((y + bh) * h)
        x1, y1 = max(0, x1 - pad_px), max(0, y1 - pad_px)
        x2, y2 = min(w, x2 + pad_px), min(h, y2 + pad_px)
        crop = frame_bgr[y1:y2, x1:x2]
        p = crop_dir / f"crop_{i:03d}.jpg"
        cv2.imwrite(str(p), crop)
        crop_paths.append(p)

    pred_json = crop_dir / "predictions.json"
    run_speciesnet_on_folder(crop_dir, pred_json, country=country)
    preds = load_speciesnet_predictions(pred_json)

    def entry_basename(e):
        for k in ("image", "img", "path", "file", "filepath", "filename", "source"):
            if k in e:
                return os.path.basename(str(e[k]))
        return None

    by_name = {entry_basename(e): e for e in preds if isinstance(e, dict)}
    results = []
    for i, p in enumerate(crop_paths):
        e = by_name.get(p.name)
        if e is None:
            e = preds[i] if i < len(preds) else {}
        results.append({
            "prediction": e.get("prediction"),
            "prediction_score": e.get("prediction_score")
        })

    shutil.rmtree(crop_dir, ignore_errors=True)
    return results

def parse_prediction_entry(entry, conf_thresh=0.2):
    dets = []
    for det in entry.get("detections", []):
        cat = det.get("category", "")
        cat_str = str(cat) if cat is not None else ""
        conf = float(det.get("conf", 0.0))
        if conf >= conf_thresh:
            label_norm = (det.get("label") or CATEGORY_MAP.get(cat_str) or str(det.get("label") or cat_str)).lower()
            dets.append({
                "bbox": det["bbox"],
                "conf": conf,
                "category": cat_str,
                "label": label_norm,
            })
    return dets, entry.get("prediction"), entry.get("prediction_score")

# ===== DRAW =====
def draw_annotations(img_bgr, detections):
    h, w = img_bgr.shape[:2]
    for det in detections:
        x, y, bw, bh = det["bbox"]
        x1, y1 = int(x * w), int(y * h)
        x2, y2 = int((x + bw) * w), int((y + bh) * h)

        label = (det.get("label") or "unknown").lower()
        if label == "animal":
            color = (0, 255, 0)
        elif label in ("human", "person"):
            color = (0, 0, 255)
        elif label == "vehicle":
            color = (255, 0, 0)
        else:
            color = (255, 255, 255)

        conf = det.get("conf")
        tid = det.get("track_id")
        if label == "animal" and det.get("species_label") and det.get("species_score") is not None:
            sp_name = _extract_common_name(det.get("species_label")) or "animal"
            txt = f"{sp_name} {det.get('species_score'):.2f}"
        else:
            main = "human" if label in ("person", "human") else label
            txt = f"{main} {conf:.2f}" if conf is not None else main
        if tid is not None:
            txt = f"{txt} id.{tid}"

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_bgr, txt, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img_bgr

def open_segment_writer(path, fps, size_wh):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, max(fps, 1.0), size_wh)

# ===== CAPTURE (freshest-frame only) =====
class LiveCapture:
    def __init__(self, src: str):
        self.src = src
        self.cap = None
        self.lock = Lock()
        self.last_frame = None
        self.last_pos_ms = 0.0
        self.last_arrival_mono = 0.0
        self.last_frame_id = -1
        self._updated_since_read = False
        self.dropped_overwrites_total = 0
        self.stopped = False
        self.ready = False
        self.start_wallclock = time.time()
        self._fail_streak = 0

    def _open(self):
        if isinstance(self.src, str) and self.src.startswith("http") and "youtube.com" in self.src:
            print(f"[Stream] Resolving YouTube stream with yt-dlp: {self.src}")
            ytdlp_cmd = ["yt-dlp", "-f", "best[ext=mp4][height<=720]", "-g", self.src]
            stream_url = subprocess.check_output(ytdlp_cmd).decode().strip().split("\n")[0]
            cap = cv2.VideoCapture(stream_url)
        else:
            cap = cv2.VideoCapture(str(self.src))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.src}")
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
        except Exception:
            pass
        self.cap = cap
        self.start_wallclock = time.time()
        self.ready = True
        self._fail_streak = 0

    def start(self):
        self._open()
        t = Thread(target=self._loop, daemon=True)
        t.start()

    def _reopen(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self._open()

    def _loop(self):
        fid = 0
        while not self.stopped:
            ok, frame = self.cap.read()
            now_mono = time.monotonic()

            if not ok or frame is None:
                self._fail_streak += 1
                if self._fail_streak > 150:
                    print("[Stream] Read fail streak; re-resolving stream...")
                    self._reopen()
                time.sleep(0.01)
                continue
            else:
                self._fail_streak = 0

            frame = to_720p(frame)
            pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0

            with self.lock:
                if self._updated_since_read and self.last_frame is not None:
                    self.dropped_overwrites_total += 1
                self.last_frame = frame
                self.last_pos_ms = float(pos_ms)
                self.last_arrival_mono = float(now_mono)
                self.last_frame_id = fid
                self._updated_since_read = True

            fid += 1

    def read_latest(self):
        with self.lock:
            if self.last_frame is None:
                return None, None, None, None
            frame = self.last_frame.copy()
            pos_ms = self.last_pos_ms
            arr = self.last_arrival_mono
            fid = self.last_frame_id
            self._updated_since_read = False
            return frame, pos_ms, arr, fid

    def close(self):
        self.stopped = True
        time.sleep(0.05)
        if self.cap:
            self.cap.release()

# ===== Simple IoU tracker (IDs per inference) =====
class IoUTracker:
    def __init__(self, iou_thresh=0.3, max_age_s=2.0):
        self.iou_thresh = 0.3 if iou_thresh is None else iou_thresh
        self.max_age_s = 2.0 if max_age_s is None else max_age_s
        self.tracks = []
        self.next_id = 1

    @staticmethod
    def _to_xyxy(b):
        x, y, w, h = b
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            raise ValueError(f"bbox not normalized in [0,1]: {b}")
        return x, y, x + w, y + h

    @staticmethod
    def _iou(b1, b2):
        x1, y1, x2, y2 = IoUTracker._to_xyxy(b1)
        X1, Y1, X2, Y2 = IoUTracker._to_xyxy(b2)
        ix1, iy1 = max(x1, X1), max(y1, Y1)
        ix2, iy2 = min(x2, X2), min(y2, Y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        a1 = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        a2 = max(0.0, (X2 - X1)) * max(0.0, (Y2 - Y1))
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, dets, now_s):
        active = [t for t in self.tracks if (now_s - t["ts"]) <= self.max_age_s]
        self.tracks = active
        T, D = len(self.tracks), len(dets)
        if T == 0 and D == 0:
            return dets
        used_t, used_d, pairs = set(), set(), []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(dets):
                pairs.append((self._iou(t["bbox"], d["bbox"]), ti, di))
        pairs.sort(reverse=True, key=lambda x: x[0])
        for iou, ti, di in pairs:
            if iou < self.iou_thresh:
                break
            if ti in used_t or di in used_d:
                continue
            dets[di]["track_id"] = self.tracks[ti]["id"]
            self.tracks[ti]["bbox"] = dets[di]["bbox"]
            self.tracks[ti]["ts"] = now_s
            used_t.add(ti); used_d.add(di)
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            tid = self.next_id; self.next_id += 1
            d["track_id"] = tid
            self.tracks.append({"id": tid, "bbox": d["bbox"], "ts": now_s})
        return dets

# ===== MAIN =====
def main():
    run_dir, DEBUG_DIR, SEG_DIR, JSONL_PATH, LOG_PATH = init_run_dirs()
    print(f"[Run] Output dir: {run_dir}")
    print(f"      Debug dir : {DEBUG_DIR}")
    print(f"      Segments  : {SEG_DIR}")
    print(f"      Results JSONL: {JSONL_PATH}")
    print(f"      Event log : {LOG_PATH}")

    lc = LiveCapture(VIDEO_SRC)
    lc.start()
    print("[Stream] Capture thread started. Lite mode (no motion gating).")

    temp_frame_dir = Path(tempfile.mkdtemp(prefix="speciesnet_in_"))
    temp_frame_path = temp_frame_dir / "frame.jpg"

    jsonl_file = open(JSONL_PATH, "w")
    log_file = open(LOG_PATH, "w")

    in_segment = False
    seg_writer = None
    seg_index = 0

    def ensure_writer(frame_shape):
        nonlocal in_segment, seg_writer, seg_index
        if not in_segment:
            in_segment = True
            seg_index += 1
            seg_path = SEG_DIR / SEGMENT_FILENAME_FMT.format(seg_index)
            h, w = frame_shape[:2]
            seg_writer = open_segment_writer(seg_path, SEGMENT_WRITER_FPS, (w, h))
            print(f"[Segment] Opened: {seg_path}")

    tracker = IoUTracker()

    try:
        while True:
            frame, pos_ms, arrival_mono, frame_id = lc.read_latest()
            if frame is None:
                time.sleep(0.01)
                continue
            cv2.imwrite(str(temp_frame_path), frame)

            temp_out_dir = Path(tempfile.mkdtemp(prefix="speciesnet_out_"))
            pred_json_path = temp_out_dir / "predictions.json"

            run_speciesnet_on_folder(temp_frame_dir, pred_json_path, country=COUNTRY, det_and_cls=True)
            preds = load_speciesnet_predictions(pred_json_path)
            shutil.rmtree(temp_out_dir, ignore_errors=True)

            entry = preds[0] if preds else {}
            dets, top_label, top_score = parse_prediction_entry(entry, CONF_DET_THRESH)

            animals = [d for d in dets if (d.get("label") == "animal")]
            if animals and not _speciesnet_supports_one_pass():
                animal_species = classify_animal_crops(frame, animals, country=COUNTRY)
                for det, sp in zip(animals, animal_species):
                    det["species_label"] = sp.get("prediction")
                    det["species_score"] = sp.get("prediction_score")

            dets = tracker.update(dets, time.monotonic())

            for i, d in enumerate(dets, start=1):
                if "track_id" in d:
                    d["track_persistent_id"] = d["track_id"]
                d["track_id"] = i

            dbg_frame = draw_annotations(frame.copy(), dets)

            ensure_writer(dbg_frame.shape)
            if seg_writer is not None:
                seg_writer.write(dbg_frame)

            if DEBUG_IMAGE_EVERY_INFERENCE:
                ts_int = int(time.time())
                fid_str = f"{int(frame_id):08d}" if frame_id is not None else "none"
                out_path = DEBUG_DIR / f"infer_{ts_int}_{fid_str}.jpg"
                cv2.imwrite(str(out_path), dbg_frame)

            counts = {}
            for d in dets:
                lbl = d.get("label") or "unknown"
                counts[lbl] = counts.get(lbl, 0) + 1

            processed_ts = time.time()
            json_line = {
                "video": str(VIDEO_SRC),
                "processed_ts": processed_ts,
                "source_ts": (lc.start_wallclock + (pos_ms / 1000.0)) if (pos_ms and pos_ms > 0) else None,
                "detections": dets,
                "counts": counts,
                "prediction": top_label,
                "prediction_score": top_score,
                "frame_id": int(frame_id) if frame_id is not None else None,
                "inference_ran": True,
            }
            json.dump(json_line, open(JSONL_PATH, "a")); open(JSONL_PATH, "a").write("\n")
            # Keep a readable copy too
            with open(JSONL_PATH, "a") as jf:
                jf.write(json.dumps(json_line, indent=2) + "\n\n")

            species_names = []
            for d in dets:
                if d.get("label") == "animal" and d.get("species_label"):
                    nm = _extract_common_name(d.get("species_label"))
                    if nm:
                        species_names.append(nm)
            species_summary = ", ".join(sorted(set(species_names))) if species_names else "none"
            counts_str = ", ".join(f"{n} {k}(s)" for k, n in counts.items() if n > 0) or "0"
            log_file.write(
                f"{datetime.fromtimestamp(processed_ts):%Y-%m-%d %H:%M:%S} - Detected {counts_str} | Species: {species_summary}\n"
            )
            log_file.flush()

    except KeyboardInterrupt:
        print("\n[Interrupt] Stopped.")
    finally:
        if seg_writer:
            seg_writer.release()
        lc.close()
        jsonl_file.close()
        log_file.close()
        try:
            for f in temp_frame_dir.iterdir():
                f.unlink()
            temp_frame_dir.rmdir()
        except Exception:
            pass

    print("\nDone.")
    print(f"- Run folder   : {run_dir}")
    print(f"- Results JSONL: {JSONL_PATH}")
    print(f"- Event log    : {LOG_PATH}")

if __name__ == "__main__":
    main()
