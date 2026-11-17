# 각인 OCR 방향 처리
# 원래 패키지는 원본·90·180·270도 x (각인 영역, 알약 전체) 8개 읽기 중 '서로 가장 비슷한 것'(MBR)을 고른다.
# 뒤집힌 알약에서 이게 틀리는 경우가 있어서, 8개 중 확률(점수)이 가장 높은 읽기를 쓰도록 바꾼다.
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
    # 다른 각도에서도 비슷하게 읽혔는지 (한 각도에서만 나온 높은 점수는 지어낸 글자일 수 있음)
    q = norm(best["text"])
    agree = {r["angle"] for r in reads if q and 1 - edit_distance(q, norm(r["text"])) / max(len(q), len(norm(r["text"])), 1) >= 0.8}
    self.last = {"text": best["text"], "raw_sequence_score": best["score"], "angle": best["angle"], "view": best["view"],
                 "agreeing_angles": len(agree),
                 "candidates": [{k: r[k] for k in ("text", "score", "angle", "view")} for r in reads]}
    return self.last


def use_max_score(model):
    # PillRecognizer 안의 OCR만 바꾼다. 패키지 코드는 그대로 둔다.
    model.ocr_reader.read = read_best.__get__(model.ocr_reader)
