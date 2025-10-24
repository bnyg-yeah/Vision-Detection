# pip install opencv-python tqdm
import json, re, cv2, numpy as np
from pathlib import Path
from tqdm import tqdm

# ========= PATHS & BASIC SETTINGS =========
JSON_PATH = "/Users/brightonyoung/ML-Projects/Vision-ML/output/TwoAnimalsSeparatedOutput.json"
VIDEO_IN  = "/Users/brightonyoung/ML-Projects/Vision-ML/TwoAnimals.mp4"
VIDEO_OUT = "/Users/brightonyoung/ML-Projects/Vision-ML/output-videos/TwoAnimalsSeparatedAnnotated.mp4"

# Must match your frame extraction fps (e.g., ffmpeg -vf "fps=5" -> 5.0)
JSON_FPS  = 5.0

# ========= DISPLAY =========
BOX_COLOR      = (0, 255, 0)
TEXT_COLOR     = (255, 255, 255)
BOX_THICKNESS  = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.6
FONT_THICKNESS = 2
MIN_BOX_W, MIN_BOX_H = 4, 4

# ========= HELPERS =========
def ascii_safe(s: str) -> str:
    if not s: return ""
    s = (s.replace("•","-").replace("·","-").replace("–","-").replace("—","-")
           .replace("’","'").replace("“",'"').replace("”",'"')
           .replace("™","").replace("©","").replace("®",""))
    return s.encode("ascii","replace").decode("ascii")

def parse_index_from_path(path):
    m = re.search(r"ts_(\d+)\.jpg$", path)
    return int(m.group(1)) if m else None

def detect_bbox_schema(b, W, H):
    x, y, a, b2 = b
    if min(x,y,a,b2) < 0:                # negatives -> absolute xyxy (safe fallback)
        return "xyxy_abs"
    if max(x,y,a,b2) <= 1.0000001:       # normalized
        return "xywh_norm" if (x+a<=1.0000001 and y+b2<=1.0000001) else "xyxy_norm"
    return "xywh_abs" if (x+a<=W+1 and y+b2<=H+1) else "xyxy_abs"

def to_xyxy_pixels(b, schema, W, H):
    x, y, a, b2 = b
    if schema == "xywh_norm": return int(x*W), int(y*H), int((x+a)*W), int((y+b2)*H)
    if schema == "xyxy_norm": return int(x*W), int(y*H), int(a*W),      int(b2*H)
    if schema == "xywh_abs":  return int(x),   int(y),   int(x+a),      int(y+b2)
    if schema == "xyxy_abs":  return int(x),   int(y),   int(a),        int(b2)
    return int(x), int(y), int(a), int(b2)

def frame_topk_species_names(meta, k=5):
    """Fallback only (if a detection has no 'species')."""
    if not meta: return []
    cl = meta.get("classifications") or {}
    classes = cl.get("classes") or []
    scores  = cl.get("scores") or []
    if not classes or not scores: return []
    idx = list(np.argsort(-np.array(scores)))[:k]
    out = []
    for i in idx:
        s = classes[i] or ""
        parts = [p.strip() for p in s.split(";")]
        name  = parts[-1] if parts else s
        out.append(ascii_safe(name))
    return out

def display_species_name(s: str) -> str:
    """
    SpeciesNet labels often look like:
    'uuid;class;order;family;genus;species;common name'
    Return just the common name; if missing, fall back to 'genus species'.
    """
    if not s:
        return ""
    parts = [p.strip() for p in str(s).split(";")]
    if len(parts) >= 7:
        return ascii_safe(parts[-1])
    genus   = parts[4] if len(parts) > 4 else ""
    species = parts[5] if len(parts) > 5 else ""
    sci = " ".join(x for x in [genus, species] if x)
    return ascii_safe(sci or parts[-1])

# ========= LOAD JSON =========
with open(JSON_PATH, "r") as f:
    raw = json.load(f)

pred_list = raw.get("predictions")
if pred_list is None:
    if isinstance(raw, list):
        pred_list = raw
    elif isinstance(raw, dict):
        pred_list = [{"filepath": k, **(v if isinstance(v, dict) else {})} for k, v in raw.items()]
    else:
        raise TypeError("Unsupported JSON format.")

by_idx = {}
for p in pred_list:
    fp = p.get("filepath") or p.get("image_path") or p.get("path") or p.get("image") or ""
    idx = parse_index_from_path(fp)
    if idx is not None:
        by_idx[idx] = p

# ========= OPEN VIDEO =========
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"Could not open: {VIDEO_IN}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Infer bbox schema from the first available detection
schema = None
for k in sorted(by_idx.keys()):
    ds = (by_idx[k].get("detections") or [])
    if ds:
        b0 = ds[0].get("bbox") or ds[0].get("box") or ds[0].get("rect") or ds[0].get("xyxy")
        if b0 is not None:
            schema = detect_bbox_schema(b0, W, H)
            break
print("Detected bbox schema:", schema or "unknown (defaulting to xyxy_abs)")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
Path(VIDEO_OUT).parent.mkdir(parents=True, exist_ok=True)
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (W, H))

# ========= MAIN LOOP =========
for i in tqdm(range(n_frames), desc="Annotating"):
    ok, frame = cap.read()
    if not ok: break

    # Map this video frame time to nearest JSON index
    t  = i / fps
    jf = int(round(t * JSON_FPS))
    meta = by_idx.get(jf)

    # Timestamp
    hh = int(t // 3600); mm = int((t % 3600) // 60); ss = int(t % 60)
    ts = ascii_safe(f"{hh:02d}:{mm:02d}:{ss:02d}  frame {i}")
    cv2.putText(frame, ts, (10, 22), FONT, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, ts, (10, 22), FONT, 0.6, (255,255,255), 1, cv2.LINE_AA)

    detections = (meta.get("detections") if meta else []) or []
    dets_sorted = sorted(detections, key=lambda d: (d.get("bbox", [0,0,0,0])[0]))

    # Frame-level fallback names (used only if a det lacks per-box species)
    frame_names = frame_topk_species_names(meta, k=max(1, len(dets_sorted)))
    used_fallback = []

    for det in dets_sorted:
        b = det.get("bbox") or det.get("box") or det.get("rect") or det.get("xyxy") or [0,0,0,0]
        x1,y1,x2,y2 = to_xyxy_pixels(b, schema or "xyxy_abs", W, H)

        # Clamp & skip tiny
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        if (x2-x1) < MIN_BOX_W or (y2-y1) < MIN_BOX_H: 
            continue

        # Draw box
        cv2.rectangle(frame, (x1,y1), (x2,y2), BOX_COLOR, BOX_THICKNESS)

        # ---- PER-DETECTION LABEL (ALWAYS DRAW) ----
        # Prefer enriched per-box fields
        species_raw = det.get("species")
        if not species_raw and isinstance(det.get("species_top3"), list) and det["species_top3"]:
            species_raw = det["species_top3"][0].get("label")

        species = display_species_name(species_raw) if species_raw else None
        sp_prob = det.get("species_score")
        det_conf = det.get("conf", det.get("score", None))

        # Fallback to frame-level guesses if species missing
        if not species:
            species = "unknown"
            for cand in frame_names:
                if cand and cand not in used_fallback:
                    species = cand
                    used_fallback.append(cand)
                    break
            sp_prob = None

        # Build tag and draw it above the box
        parts = [ascii_safe(species)]
        if isinstance(sp_prob, (int,float)): parts.append(f"{sp_prob*100:.0f}%")
        if isinstance(det_conf, (int,float)): parts.append(f"det {det_conf*100:.0f}%")
        tag = " | ".join(parts)

        tw, th = cv2.getTextSize(tag, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        y_text = max(0, y1-5)
        cv2.rectangle(frame, (x1-3, y_text-th-6), (x1+tw+6, y_text+4), (0,0,0), -1)
        cv2.putText(frame, tag, (x1, y_text), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    out.write(frame)

cap.release()
out.release()
print("Wrote", VIDEO_OUT)
