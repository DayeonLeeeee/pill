"""Reusable local high-resolution reader. No label, lexicon or medicine lookup."""
from pathlib import Path
import json, hashlib
import torch
from PIL import Image
from .model import Config, Tokenizer, build_model
from .data import letterbox, tensor
from .tta import choose, options_for
from .preprocess import fitted, encode95, attention_box, focus_raw
from .decoding import minimum_edit_risk, attentive_greedy

def load_model(path, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    cfg = Config(**state['model_config'])
    m = build_model(cfg).to(device)
    m.load_state_dict(state['model'])
    m.eval()
    m.network.decode_ar = True
    m.network.refine_iters = 0
    return (state, m)

def crop_pill(image, bbox=None):
    im = image.convert('RGB')
    if bbox is not None:
        x, y, w, h = bbox
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > im.width or (y + h > im.height):
            raise ValueError('잘못된 XYWH 알약 좌표')
        pad = max(8, int(max(w, h) * 0.12))
        im = im.crop((max(0, x - pad), max(0, y - pad), min(im.width, x + w + pad), min(im.height, y + h + pad)))
    return im

@torch.inference_mode()
def primary_options(model, images, device):
    inp = torch.stack([tensor(fitted(im, 320)) for im in images]).to(device)
    with torch.autocast(torch.device(device).type, dtype=torch.float16, enabled=torch.device(device).type == 'cuda'):
        return options_for(model, inp, Tokenizer())

@torch.inference_mode()
def auxiliary_options(model, images, device):
    bases = [encode95(letterbox(im, 224)) for im in images]
    inp = torch.stack([tensor(im) for im in bases]).to(device)
    pools = [[] for _ in images]
    maps = []
    with torch.autocast(torch.device(device).type, dtype=torch.float16, enabled=torch.device(device).type == 'cuda'):
        for k in range(4):
            ps, heat = attentive_greedy(model, torch.rot90(inp, k, (-2, -1)).contiguous())
            maps.append(heat)
            for pool, p in zip(pools, ps):
                pool.append(dict(p, angle=k * 90))
        zooms = []
        boxes = []
        for i, (im, pool) in enumerate(zip(images, pools)):
            p = choose(pool, 'vote_score')
            k = int(p['angle'] // 90)
            box = attention_box(maps[k][i], len(p['text']))
            zoom, native = focus_raw(im, box, k)
            zooms.append(fitted(zoom, 224))
            boxes.append(native)
        zs = options_for(model, torch.stack([tensor(im) for im in zooms]).to(device), Tokenizer())
    return [dict(legacy=pool, zoom=z, roi=box) for pool, z, box in zip(pools, zs, boxes)]

def select(primary, policy, aux=None):
    if policy == 'primary_vote':
        return choose(primary, 'vote_score')
    if policy == 'primary_mbr':
        return minimum_edit_risk(primary)
    if aux is None:
        raise ValueError('보조 모델 필요')
    if policy == 'legacy_mbr':
        return minimum_edit_risk(primary + aux['legacy'])
    if policy == 'focus_mbr':
        return minimum_edit_risk(primary + aux['zoom'])
    raise ValueError('Unknown policy')

class Reader:

    def __init__(self, model, device='auto', auxiliary=None, policy=None):
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.state, self.model = load_model(model, self.device)
        if self.state.get('format') != 'pill-ocr-filled-checkpoint-v1' or self.model.config.image_size != 320:
            raise ValueError('320px 확대 전처리용 모델이 필요합니다.')
        self.policy = policy or self.state['inference_policy']
        self.aux = None
        if self.policy in ['legacy_mbr', 'focus_mbr']:
            path = Path(auxiliary) if auxiliary else Path(model).resolve().parent / self.state['auxiliary_model']
            if self.state.get('auxiliary_sha256') and hashlib.sha256(path.read_bytes()).hexdigest() != self.state['auxiliary_sha256']:
                raise ValueError('선택된 보조 모델과 해시가 다릅니다.')
            self.aux_state, self.aux = load_model(path, self.device)
            if self.aux.config.image_size != 224:
                raise ValueError('224px 보조 모델 필요')

    def read(self, images):
        ps = primary_options(self.model, images, self.device)
        aux = auxiliary_options(self.aux, images, self.device) if self.aux is not None else [None] * len(images)
        result = []
        for p, a in zip(ps, aux):
            chosen = select(p, self.policy, a)
            result.append(dict(text=chosen['text'], raw_sequence_score=chosen['score'], confidence_is_calibrated=False, terminated=chosen['terminated'], inference_policy=self.policy, device=self.device, teacher_used=False, estimated_edit_risk=chosen.get('estimated_edit_risk'), attention_roi_is_human_confirmed=False, production_ready=False))
        return result
