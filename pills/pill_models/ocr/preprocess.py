"""Reversible image preparation and attention proposals; never modify source images."""
from io import BytesIO
import math
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
import hashlib, json

def raw_pill(row):
    raw = Path(row['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['sha']:
        raise ValueError('Source photo SHA mismatch')
    with Image.open(BytesIO(raw)) as im:
        im.load()
        im = im.convert('RGB')
        assert im.size == (row['width'], row['height'])
        x, y, w, h = json.loads(row['bbox'])
        pad = max(8, int(max(w, h) * 0.12))
        box = [max(0, x - pad), max(0, y - pad), min(im.width, x + w + pad), min(im.height, y + h + pad)]
        return (im.crop(box), box)

def encode95(image):
    b = BytesIO()
    image.save(b, format='JPEG', quality=95, subsampling=0)
    b.seek(0)
    with Image.open(b) as im:
        im.load()
        return im.convert('RGB')

def fitted(image, size, upscale=True, jpeg=True):
    im = image.convert('RGB')
    corners = np.asarray(im)[[0, 0, -1, -1], [0, -1, 0, -1]]
    color = tuple((int(v) for v in corners.mean(0)))
    im = im.copy()
    if upscale:
        scale = size / max(im.size)
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.LANCZOS)
    else:
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
    out = Image.new('RGB', (size, size), color)
    out.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return encode95(out) if jpeg else out

def local_contrast(image):
    gray = image.convert('L')
    a = np.asarray(gray, dtype=np.float32)
    blur = np.asarray(gray.filter(ImageFilter.GaussianBlur(max(3, min(image.size) * 0.06))), dtype=np.float32)
    detail = a - blur
    scale = max(3.0, float(np.percentile(np.abs(detail), 95)))
    enhanced = np.clip(128 + detail * (70 / scale), 0, 255)
    out = np.clip(0.55 * a + 0.45 * enhanced, 0, 255).astype('uint8')
    return Image.fromarray(out).convert('RGB')

def rotate90(image, k):
    return image if k % 4 == 0 else image.transpose({1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_270}[k % 4])

def unrotate_box(box, k, size):
    points = []
    for x, y in [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]:
        if k % 4 == 1:
            x, y = (size - y, x)
        elif k % 4 == 2:
            x, y = (size - x, size - y)
        elif k % 4 == 3:
            x, y = (y, size - x)
        points.append((x, y))
    return [min((p[0] for p in points)), min((p[1] for p in points)), max((p[0] for p in points)), max((p[1] for p in points))]

def attention_box(weights, text_length, size=224, mass=0.9):
    if text_length < 1:
        return [0, 0, size, size]
    values = weights[:min(text_length, len(weights))].float().mean(0)
    grid = math.isqrt(values.numel())
    assert grid * grid == values.numel()
    heat = values.reshape(grid, grid).clamp_min(0)
    heat = heat / heat.sum().clamp_min(1e-09)
    low = (1 - mass) / 2
    pad = size / grid * 0.5
    bounds = []
    for marginal in [heat.sum(0), heat.sum(1)]:
        cumulative = marginal.cumsum(0).cpu().numpy()
        lo = int(np.searchsorted(cumulative, low))
        hi = int(np.searchsorted(cumulative, 1 - low)) + 1
        bounds.append((max(0, lo * size / grid - pad), min(size, hi * size / grid + pad)))
    return [bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]]

def focus_raw(raw, rotated_box, k, size=224):
    base = raw.copy()
    base.thumbnail((size, size), Image.Resampling.LANCZOS)
    rw, rh = base.size
    left = (size - rw) // 2
    top = (size - rh) // 2
    x1, y1, x2, y2 = unrotate_box(rotated_box, k, size)
    box = [max(0, math.floor((x1 - left) * raw.width / rw)), max(0, math.floor((y1 - top) * raw.height / rh)), min(raw.width, math.ceil((x2 - left) * raw.width / rw)), min(raw.height, math.ceil((y2 - top) * raw.height / rh))]
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        box = [0, 0, raw.width, raw.height]
    return (rotate90(raw.crop(box), k), box)
