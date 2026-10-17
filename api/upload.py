"""Wafere încărcate de utilizator: fișier -> una sau mai multe hărți 0/1/2 (0 fundal, 1 die bun, 2 die picat).

Formate: JSON, CSV/TXT/TSV, NPY/NPZ (fără pickle) și imagini (PNG/JPG/BMP/WEBP).
Mesajele de eroare și avertismentele ajung în UI, de aceea sunt în engleză.
"""
import io
import json
import re
from pathlib import Path

import numpy as np

MAX_BYTES = 20 * 1024 * 1024      # per fișier
MAX_WAFERS = 200                  # per cerere
MIN_SIDE, MAX_SIDE = 4, 1024      # dimensiunea unei hărți numerice
PREVIEW_MAX = 128                 # hărțile mai mari se trimit înapoi micșorate la 64x64
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
TEXT_EXT = {".csv", ".txt", ".tsv"}


class UploadError(ValueError):
    pass


# ---------- intrare: fișier -> [(id sau None, array, avertismente)] ----------

def parse_file(name, data):
    """Întoarce lista de wafere din fișier: {"wafer_id", "source_file", "format", "shape", "wafer", "warnings", "stats"}."""
    if len(data) > MAX_BYTES:
        raise UploadError(f"file is larger than {MAX_BYTES // (1024 * 1024)} MB")
    if not data:
        raise UploadError("file is empty")
    ext = Path(name).suffix.lower()
    if ext == ".json":
        fmt, raw = "JSON", _from_json(data)
    elif ext in TEXT_EXT:
        fmt, raw = ext[1:].upper(), _from_text(data)
    elif ext == ".npy":
        fmt, raw = "NPY", _from_array(None, _load_npy(data))
    elif ext == ".npz":
        fmt, raw = "NPZ", _from_npz(data)
    elif ext in IMAGE_EXT:
        fmt, raw = "Image", [(None, *_from_image(data))]
    elif ext == ".pkl":
        raise UploadError("pickle files are not accepted for security reasons; export the maps to JSON or NPY")
    else:
        raise UploadError(f"unsupported file type '{ext or name}'. Use JSON, CSV, TXT, NPY, NPZ, PNG or JPG")
    if not raw:
        raise UploadError("no wafer maps found in the file")

    stem = Path(name).stem or "wafer"
    out = []
    for k, (wid, arr, warns) in enumerate(raw):
        try:
            m, more = _normalize(arr)
        except UploadError as e:
            label = f"{stem} #{k + 1}" if len(raw) > 1 else stem
            raise UploadError(f"{label}: {e}") from None
        default = stem if len(raw) == 1 else f"{stem}_{k + 1:02d}"
        out.append(_item(str(wid) if wid not in (None, "") else default, name, fmt, m, list(warns) + more))
    return out


def _item(wafer_id, source_file, fmt, m, warnings):
    shape = list(m.shape)
    if max(shape) > PREVIEW_MAX:
        m = block_mode(m, 64)
        warnings = warnings + [f"Downsampled from {shape[0]}×{shape[1]} to 64×64."]
    dies = int((m > 0).sum())
    fail = int((m == 2).sum())
    if fail == 0:
        warnings = warnings + ["No failing dies (value 2) found. If your map uses 0 = pass and 1 = fail, "
                               "convert it to 1 = pass and 2 = fail."]
    return {"wafer_id": wafer_id, "source_file": source_file, "format": fmt, "shape": shape,
            "wafer": m.astype(int).tolist(), "warnings": warnings,
            "stats": {"dies": dies, "fail": fail, "fail_pct": round(100 * fail / dies, 1) if dies else 0.0}}


def _from_json(data):
    try:
        obj = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise UploadError(f"invalid JSON: {e}") from None
    if isinstance(obj, dict) and isinstance(obj.get("wafers"), list):
        obj = obj["wafers"]
    if _is_matrix(obj):
        return [(None, obj, [])]
    if isinstance(obj, dict):
        return [_json_entry(obj, None)]
    if isinstance(obj, list):
        return [_json_entry(e, None) if isinstance(e, dict) else (None, e, []) for e in obj]
    raise UploadError("JSON must be a 2-D array, an object with a \"wafer\" array, or a list of those")


def _json_entry(e, default_id):
    m = next((e[k] for k in ("wafer", "map", "wafer_map", "waferMap", "data") if k in e), None)
    if m is None:
        raise UploadError("JSON object has no \"wafer\" (or \"map\") array")
    wid = next((e[k] for k in ("wafer_id", "id", "name") if k in e), default_id)
    return (wid, m, [])


def _is_matrix(x):
    return isinstance(x, list) and len(x) > 0 and all(isinstance(r, list) for r in x) \
        and all(not isinstance(v, (list, dict)) for r in x for v in r)


def _from_text(data):
    """Grilă de numere (separate prin virgulă, spațiu, tab sau ;); mai multe wafere = blocuri separate de o linie goală.
    O linie care începe cu # sau cu text devine ID-ul wafer-ului următor."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise UploadError("text file is not UTF-8") from None
    blocks, rows, wid, pending = [], [], None, None
    for line in text.splitlines() + [""]:
        s = line.strip()
        if not s:
            if rows:
                blocks.append((wid, rows, []))
                rows, wid = [], None
            continue
        cells = [c for c in re.split(r"[,\s;]+", s) if c]
        if all(re.fullmatch(r"-?\d+(\.\d+)?", c) for c in cells):
            if not rows:
                wid, pending = pending, None
            rows.append([float(c) for c in cells])
        else:
            if rows:   # text în mijlocul unui bloc = începe alt wafer
                blocks.append((wid, rows, []))
                rows, wid = [], None
            pending = s.lstrip("#").strip().split(",")[0].strip() or None
    return blocks


def _load_npy(data):
    try:
        return np.load(io.BytesIO(data), allow_pickle=False)
    except ValueError as e:
        raise UploadError(f"cannot read NPY (object arrays are not accepted): {e}") from None


def _from_npz(data):
    try:
        z = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as e:
        raise UploadError(f"cannot read NPZ: {e}") from None
    with z:
        keys = list(z.files)
        if "maps" in keys:          # formatul history.npz al proiectului
            keys = ["maps"]
        out = []
        for k in keys:
            out += _from_array(k if len(keys) > 1 else None, z[k])
    return out


def _from_array(wid, a):
    a = np.asarray(a)
    if a.ndim == 2:
        return [(wid, a, [])]
    if a.ndim == 3:
        prefix = f"{wid}_" if wid else ""
        # fără nume (NPY): parse_file pune <fișier>_01, _02...; în NPZ prefixul e cheia
        return [((f"{prefix}{i + 1:03d}" if prefix else None) if a.shape[0] > 1 else wid, a[i], [])
                for i in range(a.shape[0])]
    raise UploadError(f"expected a 2-D map or a 3-D stack of maps, got shape {a.shape}")


# ---------- validare ----------

def _normalize(arr):
    try:
        a = np.asarray(arr, dtype=np.float64)
    except (ValueError, TypeError):
        raise UploadError("map is not a rectangular numeric grid (all rows must have the same length)") from None
    if a.ndim != 2:
        raise UploadError(f"map must be 2-D, got shape {a.shape}")
    h, w = a.shape
    if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
        raise UploadError(f"map size {h}×{w} is outside the supported range ({MIN_SIDE}–{MAX_SIDE} per side)")
    if not np.isfinite(a).all():
        raise UploadError("map contains NaN or infinite values")
    vals = np.unique(a)
    if set(vals.tolist()) <= {0.0, 1.0, 2.0}:
        return a.astype(np.uint8), []
    if len(vals) <= 3:
        lut = {v: i + (3 - len(vals)) for i, v in enumerate(vals)}   # 2 niveluri -> 1/2, 3 niveluri -> 0/1/2
        m = np.vectorize(lut.get)(a).astype(np.uint8)
        names = {0: "background", 1: "pass", 2: "fail"}
        desc = ", ".join(f"{v:g} → {names[i]}" for v, i in lut.items())
        return m, [f"Values remapped by rank ({desc})."]
    raise UploadError(f"map has {len(vals)} distinct values; expected 0 = background, 1 = pass die, 2 = fail die")


def block_mode(m, size):
    """Micșorează o hartă de etichete la size x size: în fiecare bloc câștigă eticheta cea mai frecventă."""
    h, w = m.shape
    ys = (np.arange(h) * size // h)[:, None].repeat(w, 1)
    xs = (np.arange(w) * size // w)[None, :].repeat(h, 0)
    counts = np.zeros((size, size, 3), dtype=np.int32)
    np.add.at(counts, (ys, xs, np.clip(m, 0, 2).astype(np.int64)), 1)
    return counts.argmax(axis=2).astype(np.uint8)


# ---------- imagini ----------

def _from_image(data):
    """Imagine a unei hărți -> etichete 0/1/2. Trei grupuri de culori (k-means): fundalul e grupul de pe margine,
    die-urile picate sunt grupul mai „cald”/saturat (roșu, galben); la imagini gri, grupul mai mic."""
    from PIL import Image, UnidentifiedImageError
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except (UnidentifiedImageError, OSError) as e:
        raise UploadError(f"cannot read image: {e}") from None
    im.thumbnail((512, 512), Image.NEAREST)
    rgba = np.asarray(im.convert("RGBA"), dtype=np.float32)
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    h, w = alpha.shape
    if min(h, w) < 16:
        raise UploadError(f"image is too small ({w}×{h} px)")

    transparent = alpha < 128
    px = rgb[~transparent]
    if len(px) < 50:
        raise UploadError("image is empty or fully transparent")
    centers = _kmeans(px, 2 if transparent.any() else 3)   # cu transparență, fundalul e deja cunoscut
    lab = np.full((h, w), -1, dtype=np.int64)
    lab[~transparent] = _assign(px, centers)

    k = len(centers)
    if transparent.any():          # fundalul = transparența; cele 2 grupuri rămase sunt die-uri
        bg, rest = None, list(range(k))
    else:
        border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        bg = int(np.bincount(border, minlength=k).argmax())
        rest = [i for i in range(k) if i != bg]
    if len(rest) < 2:
        raise UploadError("could not separate pass and fail dies in the image")

    def warmth(c):
        r, g, b = c
        return (r - b) + 0.5 * (max(c) - min(c))
    a_, b_ = rest
    counts = {i: int((lab == i).sum()) for i in rest}
    if abs(warmth(centers[a_]) - warmth(centers[b_])) > 25:
        fail = a_ if warmth(centers[a_]) > warmth(centers[b_]) else b_
        rule = "warmer colour = fail"
    else:
        fail = a_ if counts[a_] < counts[b_] else b_
        rule = "minority colour = fail"
    passing = b_ if fail == a_ else a_

    m = np.zeros((h, w), dtype=np.uint8)
    m[lab == passing] = 1
    m[lab == fail] = 2
    ys, xs = np.nonzero(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]          # decupăm marginile goale
    side = max(m.shape)                                           # pătrat, ca wafer-ul să nu fie turtit
    sq = np.zeros((side, side), dtype=np.uint8)
    oy, ox = (side - m.shape[0]) // 2, (side - m.shape[1]) // 2
    sq[oy:oy + m.shape[0], ox:ox + m.shape[1]] = m
    return block_mode(sq, 64), [f"Converted from image ({w}×{h} px, {rule}). Check the preview before analysing."]


def _kmeans(px, k, iters=12):
    rng = np.random.default_rng(0)
    sample = px[rng.choice(len(px), size=min(len(px), 20000), replace=False)]
    lum = sample @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    centers = np.stack([sample[np.argsort(lum)[int(q * (len(sample) - 1))]] for q in np.linspace(0.05, 0.95, k)])
    for _ in range(iters):
        a = _assign(sample, centers)
        centers = np.stack([sample[a == i].mean(0) if (a == i).any() else centers[i] for i in range(k)])
    return centers


def _assign(px, centers):
    return ((px[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)


# ---------- ID-uri și exemple ----------

def unique_ids(items, taken=()):
    seen = set(taken)
    for it in items:
        base = re.sub(r"[^\w.\-]+", "_", it["wafer_id"]).strip("_") or "wafer"
        wid, n = base, 2
        while wid in seen:
            wid, n = f"{base}_{n}", n + 1
        it["wafer_id"] = wid
        seen.add(wid)
    return items


def sample_wafers(maps, labels, classes, per_class=1):
    """Câte un wafer real din istoric pentru fiecare clasă (pentru fișierul de exemplu din UI)."""
    out = []
    for ci, c in enumerate(classes):
        idx = np.nonzero(labels == ci)[0][:per_class]
        out += [{"wafer_id": f"sample_{str(c).replace('-', '').lower()}_{i + 1}", "true_class": str(c),
                 "wafer": np.asarray(maps[j]).astype(int).tolist()} for i, j in enumerate(idx)]
    return out


def to_csv(wafers):
    parts = []
    for w in wafers:
        parts.append(f"# {w['wafer_id']}\n" + "\n".join(",".join(str(v) for v in row) for row in w["wafer"]))
    return "\n\n".join(parts) + "\n"
