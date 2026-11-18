# OCR 8개 읽기(4방향 x roi/whole) 중 MBR 대신 점수 최대 선택
import json
from pathlib import Path

import torch
from PIL import Image

from pill_models.ocr import reader as ocr_reader
from pill_models.ocr.data import tensor
from pill_models.ocr.model import Tokenizer
from pill_models.ocr.portable_data import amp_context
from pill_models.ocr.tta import options_for
from pill_models.search import edit_distance
from verdict import norm


@torch.inference_mode()
def read_best(self, path, bbox):
    path = Path(path).resolve()
    with Image.open(path) as im:
        w, h = im.size
    row = dict(path=str(path), sha=ocr_reader.sha(path), width=w, height=h, bbox=json.dumps(list(bbox)))
    whole, roi, _ = ocr_reader.crop_views(row)
    inputs = torch.stack([tensor(ocr_reader.fit(roi, 320)), tensor(ocr_reader.fit(whole, 320))]).to(self.device)
    with amp_context(self.device):
        roi_reads, whole_reads = options_for(self.model, inputs, Tokenizer())
    for r in roi_reads:
        r["view"] = "roi"
    for r in whole_reads:
        r["view"] = "whole"
    reads = roi_reads + whole_reads
    best = max(reads, key=lambda r: r["score"])
    q = norm(best["text"])
    agree = {r["angle"] for r in reads if q and 1 - edit_distance(q, norm(r["text"])) / max(len(q), len(norm(r["text"])), 1) >= 0.8}
    self.last = {"text": best["text"], "raw_sequence_score": best["score"], "angle": best["angle"], "view": best["view"],
                 "agreeing_angles": len(agree),
                 "candidates": [{k: r[k] for k in ("text", "score", "angle", "view")} for r in reads]}
    return self.last


def use_max_score(model):
    model.ocr_reader.read = read_best.__get__(model.ocr_reader)
