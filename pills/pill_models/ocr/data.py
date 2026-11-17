from pathlib import Path
from contextlib import closing
from io import BytesIO
import hashlib, json, random, shutil, sqlite3, os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from .model import PAD
PREPROCESS_VERSION = 'pill-crop-letterbox-rgb-v1'

def load_rows(manifest, split):
    if split not in ('train', 'val', 'test'):
        raise ValueError('unknown split')
    with closing(sqlite3.connect(Path(manifest).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute('SELECT * FROM samples WHERE split=? ORDER BY sample_id', (split,))]

def letterbox(image, size):
    image = image.convert('RGB')
    corners = np.asarray(image)[[0, 0, -1, -1], [0, -1, 0, -1], :]
    color = tuple((int(v) for v in corners.mean(0)))
    image = image.copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    result = Image.new('RGB', (size, size), color)
    result.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return result

def crop_original(row, size):
    raw = Path(row['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['sha']:
        raise ValueError('원본 사진 SHA-256 불일치: ' + row['sample_id'])
    with Image.open(BytesIO(raw)) as image:
        image.load()
        image = image.convert('RGB')
        if image.size != (row['width'], row['height']):
            raise ValueError('사진 크기 불일치')
        x, y, w, h = json.loads(row['bbox'])
        padding = max(8, int(max(w, h) * 0.12))
        image = image.crop((max(0, x - padding), max(0, y - padding), min(image.width, x + w + padding), min(image.height, y + h + padding)))
        return letterbox(image, size)

def cache_path(cache, row, size):
    key = hashlib.sha256((PREPROCESS_VERSION + str(size) + row['sha'] + row['bbox']).encode()).hexdigest()
    return Path(cache) / key[:2] / (key + '.jpg')

def image_for(row, size, cache=None):
    if not cache:
        return crop_original(row, size)
    path = cache_path(cache, row, size)
    if path.exists():
        with Image.open(path) as image:
            image.load()
            if image.size != (size, size):
                raise ValueError('캐시 사진 크기 불일치')
            return image.convert('RGB')
    image = crop_original(row, size)
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < 1024 ** 3:
        raise ValueError('사진 축소 캐시용 여유 공간이 부족합니다.')
    temp = path.with_name(path.name + f'.{os.getpid()}.part')
    image.save(temp, format='JPEG', quality=95, subsampling=0)
    try:
        if os.name == 'nt':
            os.rename(temp, path)
        else:
            os.link(temp, path)
            temp.unlink()
    except FileExistsError:
        temp.unlink(missing_ok=True)
    with Image.open(path) as cached:
        cached.load()
        return cached.convert('RGB')

def augment(image, size, rng):
    angle = rng.uniform(-180, 180)
    color = tuple((int(v) for v in np.asarray(image)[0, 0, :]))
    image = letterbox(image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=color), size)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.25))
    if rng.random() < 0.1:
        image = ImageOps.grayscale(image).convert('RGB')
    if rng.random() < 0.12:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.45)))
    return image

def tensor(image):
    value = pil_to_tensor(image).float() / 255
    return (value - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]

class Pills(Dataset):

    def __init__(self, rows, tokenizer, size=160, cache=None, training=False, seed=20261109):
        self.rows = rows
        self.tokenizer = tokenizer
        self.size = size
        self.cache = cache
        self.training = training
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = image_for(row, self.size, self.cache)
        if self.training:
            key = f"{self.seed}|{self.epoch}|{row['sample_id']}"
            rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
            image = augment(image, self.size, rng)
        return (tensor(image), torch.tensor(self.tokenizer.encode(row['text']), dtype=torch.long), row['sample_id'], row['text'])

def collate(batch):
    images, tokens, ids, texts = zip(*batch)
    length = max((t.size(0) for t in tokens))
    targets = torch.full((len(batch), length), PAD, dtype=torch.long)
    for n, value in enumerate(tokens):
        targets[n, :value.size(0)] = value
    return (torch.stack(images), targets, list(ids), list(texts))
