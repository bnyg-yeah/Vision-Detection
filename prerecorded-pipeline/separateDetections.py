#!/usr/bin/env python3
"""
Enrich SpeciesNet detections with per-box species labels by re-classifying crops.

Pipeline:
1) Read base SpeciesNet JSON (from full-frame inference).
2) Crop every detection bbox from the original frame images (OpenCV).
3) Run SpeciesNet ONCE on the crops folder (your existing CLI).
4) Merge crop predictions back into the original JSON as:
     det["species"], det["species_score"], det["species_top3"].

Auto-workers:
- If --workers < 0 (default), choose 0 for small jobs, else min(8, max(1, cpu_count-1)).
- You can still force a value with --workers N.

Requires: pip install opencv-python tqdm
"""

import argparse, json, sys, shutil, subprocess, os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor

JsonLike = Union[Dict[str, Any], List[Any]]

# --------------------------- CLI ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Per-detection species enrichment using SpeciesNet on crops.")
    ap.add_argument("--base_json", required=True, help="SpeciesNet predictions JSON (from full frames).")
    ap.add_argument("--frames_dir", default=None,
                    help="Directory with original frame JPGs (optional if JSON has absolute image paths).")
    ap.add_argument("--out_json", required=True, help="Output enriched JSON path.")
    ap.add_argument("--crops_dir", default=None,
                    help="Optional scratch dir for crops. Default: '<out_stem>_crops/'.")
    ap.add_argument("--pad_pct", type=float, default=0.10,
                    help="Padding around bbox as fraction of width/height (default 0.10).")
    ap.add_argument("--min_crop", type=int, default=16,
                    help="Skip crops smaller than this (either side < min_crop).")
    ap.add_argument("--country", default=None, help="Forwarded to SpeciesNet, e.g. 'USA'.")
    ap.add_argument("--admin1", default=None, help="Forwarded to SpeciesNet, e.g. 'AL'.")
    ap.add_argument("--python_bin", default=sys.executable, help="Python to run SpeciesNet module.")
    ap.add_argument("--speciesnet_module", default="speciesnet.scripts.run_model",
                    help="Module path for SpeciesNet CLI.")
    ap.add_argument("--keep_crops", action="store_true", help="Keep crop images for debugging.")
    ap.add_argument("--workers", type=int, default=-1,
                    help="Threaded cropping workers: -1=auto (default), 0=single-thread, N=threads.")
    ap.add_argument("--opencv_threads", type=int, default=None,
                    help="If set, calls cv2.setNumThreads(N) to limit OpenCV internal threading.")
    return ap.parse_args()

# --------------------------- JSON IO ---------------------------

def load_json(path: Union[str, Path]) -> JsonLike:
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj: JsonLike, path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# --------------------------- Frame iteration ---------------------------

def _unwrap_predictions(base: JsonLike) -> JsonLike:
    # Accept {"predictions":[...]} wrapper
    if isinstance(base, dict) and "predictions" in base and isinstance(base["predictions"], list):
        return base["predictions"]
    return base

def iter_frames_with_index(base: JsonLike) -> Iterable[Tuple[int, str, Dict[str, Any]]]:
    """
    Yield (frame_idx, frame_key, record).
    Accepts:
      - dict: { "image_path.jpg": {...}, ... }
      - list: [ {...}, {...} ] where record contains an image path field.
      - {"predictions":[ ... ]} wrapper (SpeciesNet style)
    """
    base = _unwrap_predictions(base)

    if isinstance(base, dict):
        for i, (k, rec) in enumerate(base.items()):
            yield i, k, rec if isinstance(rec, dict) else {}
    elif isinstance(base, list):
        for i, rec in enumerate(base):
            rec = rec if isinstance(rec, dict) else {}
            key = (
                rec.get("image_path") or rec.get("frame_path") or rec.get("path")
                or rec.get("img") or rec.get("image") or rec.get("filepath") or ""
            )
            yield i, key, rec
    else:
        raise TypeError("Unsupported base.json structure (expected dict or list).")

def get_detections(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Be lenient about the field name
    for key in ["detections", "objects", "boxes", "bboxes", "detections_xywh", "detections_xywh_norm", "detections_xyxy"]:
        dets = rec.get(key)
        if isinstance(dets, list):
            return dets
    return []

def resolve_frame_path(frame_key: str, rec: Dict[str, Any], frames_dir: Union[str, Path, None]) -> Union[str, None]:
    # Prefer explicit fields first (including SpeciesNet's "filepath"):
    for c in [rec.get("image_path"), rec.get("frame_path"), rec.get("path"),
              rec.get("img"), rec.get("image"), rec.get("filepath"), frame_key]:
        if c and Path(c).exists():
            return str(Path(c))
    # If not found, try frames_dir + filename (fallback)
    if frames_dir:
        filename = None
        for c in [rec.get("image_path"), rec.get("path"), rec.get("filepath"),
                  frame_key, rec.get("img"), rec.get("image")]:
            if c:
                filename = Path(c).name
                break
        if not filename:
            filename = rec.get("filename") or rec.get("image_name")
        if filename:
            p = Path(frames_dir) / filename
            if p.exists():
                return str(p)
    return None

# --------------------------- BBox utilities ---------------------------

def bbox_to_xyxy(bbox: Union[Dict[str, Any], List[float], Tuple[float, ...]]) -> Tuple[float, float, float, float]:
    """Accepts dict or [4]-list in either (x,y,w,h) or (x1,y1,x2,y2). Returns (x1,y1,x2,y2)."""
    if isinstance(bbox, dict):
        k = set(bbox.keys())
        if {"x","y","w","h"} <= k:
            return float(bbox["x"]), float(bbox["y"]), float(bbox["x"]+bbox["w"]), float(bbox["y"]+bbox["h"])
        if {"x1","y1","x2","y2"} <= k:
            return float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])
        if {"xmin","ymin","xmax","ymax"} <= k:
            return float(bbox["xmin"]), float(bbox["ymin"]), float(bbox["xmax"]), float(bbox["ymax"])
        # Common dumps: {"xywh":[...]} or {"xyxy":[...]} or {"bbox":[...]}
        for arr_key in ["xyxy", "bbox", "xywh"]:
            if arr_key in bbox and isinstance(bbox[arr_key], (list, tuple)) and len(bbox[arr_key]) == 4:
                x0, y0, a, b = [float(v) for v in bbox[arr_key]]
                if arr_key in ("xyxy",) or (a > x0 and b > y0):
                    return x0, y0, a, b
                return x0, y0, x0 + a, y0 + b
        raise ValueError(f"Unsupported bbox dict keys: {k}")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x0, y0, a, b = [float(v) for v in bbox]
        # Heuristic: if a > x0 and b > y0, it's likely xyxy; else xywh
        if a > x0 and b > y0:
            return x0, y0, a, b
        return x0, y0, x0 + a, y0 + b
    raise ValueError(f"Unsupported bbox type: {type(bbox)}")

def pad_and_clip(x1: float, y1: float, x2: float, y2: float,
                 W: int, H: int, pad_pct: float) -> Tuple[int,int,int,int]:
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx, cy = x1 + w/2.0, y1 + h/2.0
    nx1 = max(0, int(round(cx - (w * (0.5 + pad_pct)))))
    ny1 = max(0, int(round(cy - (h * (0.5 + pad_pct)))))
    nx2 = min(W, int(round(cx + (w * (0.5 + pad_pct)))))
    ny2 = min(H, int(round(cy + (h * (0.5 + pad_pct)))))
    if nx2 <= nx1: nx2 = min(W, nx1+1)
    if ny2 <= ny1: ny2 = min(H, ny1+1)
    return nx1, ny1, nx2, ny2

# --------------------------- Cropping ---------------------------

def crop_and_save(img_path: Union[str, Path],
                  bbox_any: Union[Dict[str, Any], List[float], Tuple[float, ...]],
                  out_path: Union[str, Path],
                  pad_pct: float,
                  min_side: int) -> bool:
    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im is None:
        return False
    H, W = im.shape[:2]

    # Parse to xyxy in input units
    x1, y1, x2, y2 = bbox_to_xyxy(bbox_any)

    # If values look normalized (all in [0,1]), scale to pixels
    vals = [x1, y1, x2, y2]
    if all(0.0 <= v <= 1.0 for v in vals):
        x1, y1, x2, y2 = x1 * W, y1 * H, x2 * W, y2 * H

    x1, y1, x2, y2 = pad_and_clip(x1, y1, x2, y2, W, H, pad_pct)

    if (x2 - x1) < min_side or (y2 - y1) < min_side:
        return False

    crop = im[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        return False
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return True

def build_crops(base: JsonLike,
                frames_dir: Union[str, None],
                crops_dir: Union[str, Path],
                pad_pct: float,
                min_side: int,
                workers: int) -> List[Dict[str, Any]]:
    """
    Returns a list of mappings:
      {
        "crop_path": "...",
        "frame_idx": int,
        "frame_key": str,
        "det_index": int
      }
    """
    Path(crops_dir).mkdir(parents=True, exist_ok=True)
    tasks: List[Tuple[str, Tuple[str,int,str,int,Any]]] = []
    mapping: List[Dict[str, Any]] = []

    # collect tasks
    for frame_idx, frame_key, rec in iter_frames_with_index(base):
        dets = get_detections(rec)
        if not dets:
            continue
        frame_path = resolve_frame_path(frame_key, rec, frames_dir)
        if not frame_path or not Path(frame_path).exists():
            print(f"[warn] cannot find frame image for key='{frame_key}'. Skipping.", file=sys.stderr)
            continue
        for j, det in enumerate(dets):
            bbox = det.get("bbox") or det.get("box") or det.get("rect") or det.get("xyxy")
            if bbox is None:
                continue
            crop_name = f"crop_{frame_idx:06d}_{j:03d}.jpg"
            crop_path = str(Path(crops_dir) / crop_name)
            tasks.append((frame_path, (crop_path, frame_idx, frame_key, j, bbox)))

    # execute (single-thread or threaded)
    if workers <= 0:
        for frame_path, t in tqdm(tasks, desc="Cropping detections"):
            crop_path, frame_idx, frame_key, det_index, bbox = t
            ok = crop_and_save(frame_path, bbox, crop_path, pad_pct, min_side)
            if ok:
                mapping.append({"crop_path": crop_path, "frame_idx": frame_idx,
                                "frame_key": frame_key, "det_index": det_index})
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = []
            for frame_path, t in tasks:
                crop_path, frame_idx, frame_key, det_index, bbox = t
                futs.append(ex.submit(crop_and_save, frame_path, bbox, crop_path, pad_pct, min_side))
            for (frame_path, t), fut in tqdm(zip(tasks, futs), total=len(tasks), desc="Cropping detections"):
                ok = fut.result()
                if ok:
                    crop_path, frame_idx, frame_key, det_index, _ = t
                    mapping.append({"crop_path": crop_path, "frame_idx": frame_idx,
                                    "frame_key": frame_key, "det_index": det_index})

    return mapping

# --------------------------- SpeciesNet call ---------------------------

def run_speciesnet_on_crops(crops_dir: Union[str, Path],
                            crop_preds_json: Union[str, Path],
                            python_bin: str,
                            speciesnet_module: str,
                            country: Union[str, None],
                            admin1: Union[str, None]) -> None:
    cmd = [python_bin, "-m", speciesnet_module,
           "--folders", str(crops_dir),
           "--predictions_json", str(crop_preds_json)]
    if country:
        cmd += ["--country", country]
    if admin1:
        cmd += ["--admin1_region", admin1]
    print("[info] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# --------------------------- Merge predictions ---------------------------

def extract_topk_from_pred(pred_rec: Dict[str, Any], k: int = 3) -> List[Tuple[str, float]]:
    """
    Works with:
      - list of {label/name/species, score/prob}
      - dict mapping {label: score}
      - SpeciesNet crop JSON variants with {"classifications":{"classes":[...],"scores":[...]}}
    Returns list[(label, score)] sorted desc.
    """
    # SpeciesNet-like: {"classifications":{"classes":[...], "scores":[...]}}
    cls = pred_rec.get("classifications")
    if isinstance(cls, dict) and isinstance(cls.get("classes"), list) and isinstance(cls.get("scores"), list):
        pairs = list(zip(cls["classes"], cls["scores"]))
        # already sorted? be safe and sort:
        pairs.sort(key=lambda p: float(p[1]), reverse=True)
        out = []
        for label, score in pairs[:k]:
            # Label might be a semicolon taxonomy string; use as-is (user can parse later)
            out.append((str(label), float(score)))
        return out

    # Generic list form: [{"label":..., "score":...}, ...]
    cl = pred_rec.get("species") or pred_rec.get("species_scores") or pred_rec.get("classifications")
    if isinstance(cl, list):
        sorted_list = sorted(
            cl, key=lambda d: float(d.get("score", d.get("prob", 0.0))), reverse=True
        )
        out = []
        for d in sorted_list[:k]:
            label = d.get("label") or d.get("species") or d.get("name")
            if label is None:
                continue
            score = float(d.get("score", d.get("prob", 0.0)))
            out.append((str(label), score))
        return out

    # Dict mapping form: {"lion":0.9, ...}
    if isinstance(cl, dict):
        items = sorted(cl.items(), key=lambda kv: float(kv[1]), reverse=True)[:k]
        return [(str(k), float(v)) for k, v in items]

    return []

def build_crop_lookup(crop_preds: JsonLike) -> Dict[str, Dict[str, Any]]:
    """Map filename -> prediction record."""
    lookup: Dict[str, Dict[str, Any]] = {}

    # Accept {"predictions":[...]} from SpeciesNet on crops
    crop_preds = _unwrap_predictions(crop_preds)

    if isinstance(crop_preds, dict):
        for k, v in crop_preds.items():
            lookup[Path(k).name] = v if isinstance(v, dict) else {}
    elif isinstance(crop_preds, list):
        for rec in crop_preds:
            if not isinstance(rec, dict):
                continue
            path = rec.get("image_path") or rec.get("path") or rec.get("image") or rec.get("filepath")
            if path:
                lookup[Path(path).name] = rec
    return lookup

def enrich_base_with_species(base: JsonLike,
                             mapping: List[Dict[str, Any]],
                             crop_lookup: Dict[str, Dict[str, Any]]) -> None:
    """In-place mutation: add species fields to detections."""
    def get_rec(base_obj: JsonLike, frame_idx: int, frame_key: str) -> Dict[str, Any]:
        base_u = _unwrap_predictions(base_obj)
        if isinstance(base_u, list):
            return base_u[frame_idx]
        if isinstance(base_u, dict):
            if frame_key in base_u and isinstance(base_u[frame_key], dict):
                return base_u[frame_key]
            # fallback: match by embedded path fields
            for rec in base_u.values():
                if isinstance(rec, dict):
                    if rec.get("image_path") == frame_key or rec.get("path") == frame_key or rec.get("filepath") == frame_key:
                        return rec
        raise KeyError(f"Could not find record for frame_idx={frame_idx}, frame_key='{frame_key}'")

    for m in mapping:
        crop_name = Path(m["crop_path"]).name
        pred = crop_lookup.get(crop_name)
        if not pred:
            print(f"[warn] no prediction for crop {crop_name}", file=sys.stderr)
            continue

        topk = extract_topk_from_pred(pred, k=3)
        if not topk:
            continue

        best_label, best_score = topk[0]
        top3 = [{"label": l, "score": s} for (l, s) in topk]

        try:
            rec = get_rec(base, m["frame_idx"], m["frame_key"])
            dets = get_detections(rec)
            j = m["det_index"]
            if j < 0 or j >= len(dets):
                print(f"[warn] det_index out of range for frame {m['frame_key']}", file=sys.stderr)
                continue
            det = dets[j]
            det["species"] = best_label
            det["species_score"] = float(best_score)
            det["species_top3"] = top3
        except Exception as e:
            print(f"[warn] failed to enrich detection: {e}", file=sys.stderr)

# --------------------------- Auto workers ---------------------------

def estimate_total_crops(base: JsonLike, frames_dir: Union[str, Path, None]) -> int:
    """Quickly count how many detections we'll attempt to crop (frame exists + dets present)."""
    total = 0
    for _, frame_key, rec in iter_frames_with_index(base):
        dets = get_detections(rec)
        if not dets:
            continue
        if resolve_frame_path(frame_key, rec, frames_dir):
            total += len(dets)
    return total

def autotune_workers(n_tasks: int) -> int:
    """Heuristic: if small job, stay serial; else use a modest pool."""
    if n_tasks < 200:
        return 0
    cores = os.cpu_count() or 4
    return min(8, max(1, cores - 1))

# --------------------------- Main ---------------------------

def main():
    args = parse_args()

    if args.opencv_threads is not None:
        try:
            cv2.setNumThreads(int(args.opencv_threads))
            print(f"[info] OpenCV internal threads set to {int(args.opencv_threads)}")
        except Exception as e:
            print(f"[warn] cv2.setNumThreads failed: {e}", file=sys.stderr)

    base = load_json(args.base_json)

    # Decide worker count
    workers = int(args.workers)
    if workers < 0:
        n_tasks = estimate_total_crops(base, args.frames_dir)
        workers = autotune_workers(n_tasks)
        print(f"[info] planned crops: {n_tasks} -> auto workers = {workers}")

    # crops workspace
    crops_dir = args.crops_dir or (Path(args.out_json).with_name(Path(args.out_json).stem + "_crops"))
    crops_dir = Path(crops_dir)

    # fresh run: clear previous crops
    if crops_dir.exists():
        shutil.rmtree(crops_dir)

    # 1) Crop detections
    mapping = build_crops(
        base=base,
        frames_dir=args.frames_dir,
        crops_dir=crops_dir,
        pad_pct=args.pad_pct,
        min_side=args.min_crop,
        workers=workers,
    )
    if not mapping:
        print("[info] No detections to enrich — writing base JSON as-is.")
        save_json(base, args.out_json)
        return

    # 2) Run SpeciesNet once over all crops
    crop_preds_json = Path(args.out_json).with_name(Path(args.out_json).stem + "_crops_predictions.json")
    run_speciesnet_on_crops(
        crops_dir=crops_dir,
        crop_preds_json=crop_preds_json,
        python_bin=args.python_bin,
        speciesnet_module=args.speciesnet_module,
        country=args.country,
        admin1=args.admin1,
    )

    # 3) Merge crop predictions back
    crop_preds = load_json(crop_preds_json)
    crop_lookup = build_crop_lookup(crop_preds)
    enrich_base_with_species(base, mapping, crop_lookup)

    # 4) Save enriched JSON
    save_json(base, args.out_json)
    print(f"[done] wrote enriched JSON → {args.out_json}")

    # Optional cleanup
    if not args.keep_crops:
        try:
            shutil.rmtree(crops_dir)
        except Exception:
            pass
        print("[info] removed crops directory")
    else:
        print(f"[info] kept crops in: {crops_dir}")
        print(f"[info] crops predictions: {crop_preds_json}")

if __name__ == "__main__":
    main()
