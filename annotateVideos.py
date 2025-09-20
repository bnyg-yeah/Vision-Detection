# pip install opencv-python tqdm
import json, re, cv2, numpy as np
from tqdm import tqdm

# ========= PATHS & BASIC SETTINGS =========
JSON_PATH = "/Users/brightonyoung/ML-Projects/Vision-ML/output/TwoAnimalsOutput.json"  # your JSON file
VIDEO_IN  = "/Users/brightonyoung/ML-Projects/Vision-ML/TwoAnimals.mp4"
VIDEO_OUT = "/Users/brightonyoung/ML-Projects/Vision-ML/output-videos/TwoAnimals3.mp4"

# JSON frames-per-second of classification/detection outputs
JSON_FPS  = 5.0

# ======== RUNTIME CLASSIFIER CONFIG (EDIT THESE TO MATCH YOUR MODEL) ========
MIN_CROP_SIZE   = 16          # skip tiny crops (width or height < MIN_CROP_SIZE)
USE_CACHE       = True        # cache per-detection species within this run
NAME_STYLE      = "common"    # "common" or "scientific"
MODEL_INPUT_SIZE = (224, 224) # e.g., (224, 224) or None if your function handles resizing

# ========= DISPLAY =========
BOX_COLOR      = (0, 255, 0)
TEXT_COLOR     = (255, 255, 255)
BOX_THICKNESS  = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.6
FONT_THICKNESS = 2

# ========= HELPERS =========
def ascii_safe(s: str) -> str:
    """Make text safe for cv2.putText (ASCII only)."""
    if not s:
        return ""
    # Common replacements
    s = (s.replace("•", "-")
           .replace("·", "-")
           .replace("–", "-")
           .replace("—", "-")
           .replace("’", "'")
           .replace("“", '"')
           .replace("”", '"')
           .replace("™", "")
           .replace("©", "")
           .replace("®", ""))
    # Drop/replace any remaining non-ASCII
    return s.encode("ascii", "replace").decode("ascii")

def parse_index_from_path(path):
    # .../ts_0000000034.jpg -> 34
    m = re.search(r"ts_(\d+)\.jpg$", path)
    return int(m.group(1)) if m else None

def species_name_from_taxon_string(s, prefer="common"):
    """
    Taxon string format: 'uuid;class;order;family;genus;species;common name'
    Return either 'common name' or 'genus species' (scientific).
    """
    if not s:
        return ""
    parts = [p.strip() for p in s.split(";")]
    common  = parts[-1] if parts else ""
    genus   = parts[4] if len(parts) > 4 else ""
    species = parts[5] if len(parts) > 5 else ""
    sci = " ".join(x for x in [genus, species] if x)
    name = (sci if (prefer == "scientific" and sci) else (common or sci or s))
    return ascii_safe(name)

def detect_bbox_schema(b, W, H):
    """
    Guess bbox schema.
    Returns: 'xywh_norm', 'xyxy_norm', 'xywh_abs', 'xyxy_abs'
    """
    x, y, a, b2 = b
    all_nonneg = min(x,y,a,b2) >= 0
    if not all_nonneg:
        return "xyxy_abs"
    all_le1 = max(x,y,a,b2) <= 1.0000001
    if all_le1:
        if (x + a <= 1.0000001) and (y + b2 <= 1.0000001):
            return "xywh_norm"
        return "xyxy_norm"
    else:
        if (x + a <= W + 1) and (y + b2 <= H + 1):
            return "xywh_abs"
        return "xyxy_abs"

def to_xyxy_pixels(b, schema, W, H):
    x, y, a, b2 = b
    if schema == "xywh_norm":
        return int(x*W), int(y*H), int((x+a)*W), int((y+b2)*H)
    if schema == "xyxy_norm":
        return int(x*W), int(y*H), int(a*W), int(b2*H)
    if schema == "xywh_abs":
        return int(x), int(y), int(x+a), int(y+b2)
    if schema == "xyxy_abs":
        return int(x), int(y), int(a), int(b2)
    return int(x), int(y), int(a), int(b2)

def draw_text_with_bg(img, text, org, font_scale=FONT_SCALE, thickness=FONT_THICKNESS,
                      text_color=TEXT_COLOR, bg_color=(0,0,0), alpha=0.35):
    text = ascii_safe(text)
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = org
    y_top = max(0, y - th - 4)
    overlay = img.copy()
    cv2.rectangle(overlay, (x-3, y_top), (x+tw+6, y+4), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    cv2.putText(img, text, (x, y), FONT, font_scale, text_color, thickness, cv2.LINE_AA)

def crop_from_bbox(frame_bgr, x1, y1, x2, y2):
    # Ensure sane order/bounds
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    x1 = max(0, min(frame_bgr.shape[1]-1, x1))
    x2 = max(0, min(frame_bgr.shape[1]-1, x2))
    y1 = max(0, min(frame_bgr.shape[0]-1, y1))
    y2 = max(0, min(frame_bgr.shape[0]-1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop_bgr = frame_bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    if MODEL_INPUT_SIZE:
        crop_rgb = cv2.resize(crop_rgb, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    return crop_rgb

# ========= SPECIES CLASSIFIER HOOK (SpeciesNet) =========
# Replace this stub with your SpeciesNet call. The expected return is:
#   classes: list[str] of taxon strings 'uuid;...;genus;species;common'
#   scores:  list[float] probabilities for each class
def classify_species(crop_rgb: np.ndarray):
    """
    Example wiring (pseudo-code):

    # Preprocess for SpeciesNet
    x = crop_rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2,0,1))[None, ...]  # NCHW

    # logits = speciesnet_model(x)          # torch / onnxruntime / tf
    # probs = softmax(logits)
    # topk_idx = np.argsort(-probs[0])[:5]
    # classes = [id_to_taxon[i] for i in topk_idx]
    # scores  = [float(probs[0, i]) for i in topk_idx]
    # return classes, scores
    """
    raise NotImplementedError("Wire this to your SpeciesNet model.")

def top1_species_from_crop(crop_rgb):
    try:
        classes, scores = classify_species(crop_rgb)
    except NotImplementedError:
        return None, 0.0
    if not classes or not scores:
        return None, 0.0
    j = int(np.argmax(scores))
    return species_name_from_taxon_string(classes[j], prefer=NAME_STYLE), float(scores[j])

# ========= LOAD JSON =========
with open(JSON_PATH, "r") as f:
    raw = json.load(f)

preds = raw.get("predictions", [])
by_idx = {}
for p in preds:
    idx = parse_index_from_path(p.get("filepath", ""))
    if idx is not None:
        by_idx[idx] = p

# ========= OPEN VIDEO =========
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"Could not open: {VIDEO_IN}")

fps = cap.get(cv2.CAP_PROP_FPS)
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Infer bbox schema from the first available detection
schema = None
for k in sorted(by_idx.keys()):
    ds = by_idx[k].get("detections", [])
    if ds:
        schema = detect_bbox_schema(ds[0]["bbox"], W, H)
        break
if schema is None:
    print("No detections in JSON; continuing without boxes.")
else:
    print("Detected bbox schema:", schema)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (W, H))

# Optional cache: key by (json_index, x1,y1,x2,y2)
species_cache = {} if USE_CACHE else None

def frame_topk_species_names(meta, k=5):
    """Return up to k species names (ordered by score desc) from frame-level classifications."""
    classes = (meta.get("classifications", {}) or {}).get("classes", []) or []
    scores  = (meta.get("classifications", {}) or {}).get("scores", []) or []
    if not classes or not scores:
        return []
    idx = list(np.argsort(-np.array(scores)))[:k]
    return [species_name_from_taxon_string(classes[i], prefer=NAME_STYLE) for i in idx]

# ========= MAIN LOOP =========
for i in tqdm(range(n_frames), desc="Annotating"):
    ok, frame = cap.read()
    if not ok:
        break

    # Map this video frame time to nearest JSON index
    t  = i / fps
    jf = int(round(t * JSON_FPS))
    meta = by_idx.get(jf)

    # Small corner time stamp (ASCII-only)
    hh = int(t // 3600); mm = int((t % 3600) // 60); ss = int(t % 60)
    ts = ascii_safe(f"{hh:02d}:{mm:02d}:{ss:02d}  frame {i}")
    cv2.putText(frame, ts, (10, 22), FONT, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, ts, (10, 22), FONT, 0.6, (255,255,255), 1, cv2.LINE_AA)

    detections = meta.get("detections", []) if meta else []
    # Sort detections left->right to make frame-level fallback assignment stable
    dets_sorted = sorted(detections, key=lambda d: (d.get("bbox", [0,0,0,0])[0]))

    # Precompute frame-level top-k names for fallback & enforce uniqueness
    frame_names = frame_topk_species_names(meta, k=max(1, len(dets_sorted))) if meta else []
    used_names = []

    for det_idx, det in enumerate(dets_sorted):
        b = det.get("bbox", [0,0,0,0])
        x1,y1,x2,y2 = to_xyxy_pixels(b, schema or "xyxy_abs", W, H)

        # Clamp & sort
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1

        cv2.rectangle(frame, (x1,y1), (x2,y2), BOX_COLOR, BOX_THICKNESS)

        # Skip tiny crops to avoid garbage classifications
        if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
            species = "unknown"
            sp_prob = None
        else:
            cache_key = (jf, x1, y1, x2, y2) if species_cache is not None else None
            if species_cache is not None and cache_key in species_cache:
                species, sp_prob = species_cache[cache_key]
            else:
                crop = crop_from_bbox(frame, x1, y1, x2, y2)
                sp_name, sp_score = (None, 0.0)
                if crop is not None and crop.size:
                    sp_name, sp_score = top1_species_from_crop(crop)

                if sp_name:
                    species, sp_prob = sp_name, sp_score
                else:
                    # Distinct frame-level fallback (left->right)
                    species = "unknown"
                    for cand in frame_names:
                        if cand and cand not in used_names:
                            species = cand
                            used_names.append(cand)
                            break
                    sp_prob = None

                if species_cache is not None:
                    species_cache[cache_key] = (species, sp_prob)

        # Label text: ASCII-only; avoid bullets or fancy symbols.
        det_conf = det.get("conf", 0.0)
        if sp_prob is not None and sp_prob > 0:
            tag = f"{species} | {sp_prob*100:.0f}% | det {det_conf*100:.0f}%"
        else:
            tag = f"{species} | det {det_conf*100:.0f}%"
        draw_text_with_bg(frame, tag, (x1, max(0, y1-5)))

    out.write(frame)

cap.release()
out.release()
print("Wrote", VIDEO_OUT)
