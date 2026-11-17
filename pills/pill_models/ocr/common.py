"""Portable v12 functions; image preparation and CER math are copied verbatim."""
from pathlib import Path
from contextlib import closing
from collections import defaultdict, Counter
import hashlib, json, math, os, random, sqlite3, sys, time
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
ROOT = Path(__file__).resolve().parents[2]
SEED = 20261116

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.next')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), 'utf8')
    os.replace(tmp, path)

def lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.next')
    with tmp.open('w', encoding='utf8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n')
    os.replace(tmp, path)

def read(path):
    return json.loads(Path(path).read_text('utf8'))

def read_lines(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]

def normalized(text, ignore_space=True):
    return ''.join(text.split()) if ignore_space else text

def alignment(gt, pred):
    """Levenshtein alignment, including insertions; never whole-string 0/1 scoring."""
    dp = [list(range(len(pred) + 1))] + [[i] + [0] * len(pred) for i in range(1, len(gt) + 1)]
    for i, a in enumerate(gt, 1):
        for j, b in enumerate(pred, 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a != b))
    i, j = (len(gt), len(pred))
    ops = []
    while i or j:
        if i and j and (dp[i][j] == dp[i - 1][j - 1] + (gt[i - 1] != pred[j - 1])):
            ops.append(dict(op='equal' if gt[i - 1] == pred[j - 1] else 'substitute', gt=gt[i - 1], pred=pred[j - 1], gt_index=i - 1, pred_index=j - 1))
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(dict(op='delete', gt=gt[i - 1], pred='', gt_index=i - 1, pred_index=None))
            i -= 1
        else:
            ops.append(dict(op='insert', gt='', pred=pred[j - 1], gt_index=None, pred_index=j - 1))
            j -= 1
    return list(reversed(ops))

class Meter:

    def __init__(self, ignore_space=True):
        self.space = ignore_space
        self.counts = Counter()

    def add(self, gt, pred):
        gt = normalized(gt, self.space)
        pred = normalized(pred, self.space)
        a = alignment(gt, pred)
        self.counts.update((x['op'] for x in a))
        self.counts['reference_characters'] += len(gt)
        self.counts['samples'] += 1
        self.counts['blank_reference_samples'] += not bool(gt)
        self.counts['blank_false_positive_samples'] += not bool(gt) and bool(pred)
        self.counts['exact_matches_reference_only'] += gt == pred
        return a

    def result(self):
        c = self.counts
        err = sum((c[k] for k in ['substitute', 'delete', 'insert']))
        n = c['reference_characters']
        return dict(c, character_edits=err, cer=err / n if n else None, char_edit_accuracy=1 - err / n if n else None, whitespace_ignored=self.space, case_sensitive=True, punctuation_preserved=True, definition='CER=(substitutions+deletions+insertions)/GT characters; abc vs abd => 1/3, two correct characters; can exceed 1')

def metrics(records):
    primary, raw = (Meter(), Meter(False))
    for r in records:
        primary.add(r['expected'], r['predicted'])
        raw.add(r['expected'], r['predicted'])
    return dict(primary=primary.result(), raw=raw.result())

def original(row):
    from io import BytesIO
    raw = Path(row['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row['sha'], 'Photo changed'
    with Image.open(BytesIO(raw)) as im:
        im.load()
        assert im.size == (row['width'], row['height'])
        return im.convert('RGB')

def crop_views(row):
    """Whole-pill tight crop plus conservative image-only imprint proposal; GT never read."""
    im = original(row)
    x, y, w, h = json.loads(row['bbox'])
    pad = max(4, round(max(w, h) * 0.025))
    whole = im.crop((max(0, x - pad), max(0, y - pad), min(im.width, x + w + pad), min(im.height, y + h + pad)))
    gray = whole.convert('L')
    small = gray.copy()
    small.thumbnail((384, 384), Image.Resampling.LANCZOS)
    a = np.asarray(small, dtype=np.float32)
    bg = np.asarray(small.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    hh, ww = a.shape
    yy, xx = np.mgrid[:hh, :ww]
    mask = ((xx - ww / 2) / (ww * 0.47)) ** 2 + ((yy - hh / 2) / (hh * 0.47)) ** 2 < 1
    energy = np.abs(a - bg)
    vals = energy[mask]
    hot = mask & (energy > max(2.0, float(np.percentile(vals, 78))))
    ys, xs = np.nonzero(hot)
    box = (0, 0, whole.width, whole.height)
    if len(xs) > 20:
        left, right = np.percentile(xs, [0.5, 99.5])
        top, bottom = np.percentile(ys, [0.5, 99.5])
        cx, cy = ((left + right) / 2, (top + bottom) / 2)
        bw = max(ww * 0.72, (right - left) * 1.25)
        bh = max(hh * 0.72, (bottom - top) * 1.25)
        left = max(0, cx - bw / 2)
        right = min(ww, cx + bw / 2)
        top = max(0, cy - bh / 2)
        bottom = min(hh, cy + bh / 2)
        box = (math.floor(left * whole.width / ww), math.floor(top * whole.height / hh), math.ceil(right * whole.width / ww), math.ceil(bottom * whole.height / hh))
    roi = whole.crop(box)
    return (whole, roi, box)

def fit(im, size):
    color = tuple((int(v) for v in np.asarray(im)[[0, 0, -1, -1], [0, -1, 0, -1]].mean(0)))
    out = Image.new('RGB', (size, size), color)
    s = size / max(im.size)
    resized = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.Resampling.LANCZOS)
    out.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return out

def safe_rotate(im, angle):
    color = tuple((int(v) for v in np.asarray(im)[[0, 0, -1, -1], [0, -1, 0, -1]].mean(0)))
    return im.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor=color)

def image(row, size, view='roi', rng=None):
    whole, roi, _ = crop_views(row)
    im = roi if view == 'roi' else whole
    if rng is not None:
        im = safe_rotate(im, rng.uniform(-180, 180))
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.9, 1.1))
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.2))
        if rng.random() < 0.15:
            im = ImageOps.grayscale(im).convert('RGB')
    return fit(im, size)

def plots(run, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not history:
        return
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for a, key, label in zip(ax.flat, ['train_acc', 'train_loss', 'val_acc', 'val_loss'], ['TRAIN ACC (1-CER)', 'TRAIN LOSS (CE)', 'VAL ACC (1-CER)', 'VAL LOSS (CE)']):
        points = [(r['epoch'], r[key]) for r in history if r.get(key) is not None]
        if points:
            a.plot(*zip(*points), marker='o')
        a.set_title(label)
        a.set_xlabel('epoch')
        a.grid(alpha=0.25)
    fig.suptitle('Original GT preserved; accuracy uses character edit distance')
    fig.tight_layout()
    fig.savefig(Path(run) / 'learning_curves.png', dpi=160)
    plt.close(fig)
    write(Path(run) / 'history.json', history)
