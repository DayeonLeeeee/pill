"""Independent v12 PARSeq reader. No labels, Qwen or original PC paths required."""
from .common import *
import torch
from .runtime import load_model, select
from .data import tensor
from .model import Tokenizer
from .tta import options_for
from .portable_data import choose_device, amp_context

class Reader:

    def __init__(self, weights=None, device='auto', threads=6):
        torch.set_num_threads(threads)
        self.device = choose_device(device)
        path = Path(weights).resolve() if weights else ROOT / 'weights/parseq_v12.pt'
        if not weights:
            expected = read(ROOT / 'weights/weights.json')['parseq_v12.pt']
            if sha(path) != expected:
                raise ValueError('Bundled model SHA-256 mismatch')
        self.state, self.model = load_model(path, str(self.device))
        if self.model.config.image_size != 320 or self.model.config.architecture != 'parseq_wide_v6':
            raise ValueError('Expected the 320px wide PARSeq checkpoint')
        self.weights_sha256 = sha(path)

    @torch.inference_mode()
    def read(self, path, bbox=None):
        path = Path(path).resolve()
        with Image.open(path) as im:
            w, h = im.size
        bbox = list(bbox) if bbox is not None else [0, 0, w, h]
        if len(bbox) != 4:
            raise ValueError('bbox requires x,y,width,height')
        x, y, bw, bh = bbox
        if not all((math.isfinite(v) for v in bbox)) or min(x, y) < 0 or min(bw, bh) <= 0 or (x + bw > w) or (y + bh > h):
            raise ValueError('Invalid original-image XYWH pill bbox')
        row = dict(path=str(path), sha=sha(path), width=w, height=h, bbox=json.dumps(bbox))
        whole, roi, roi_box = crop_views(row)
        inputs = torch.stack([tensor(fit(roi, 320)), tensor(fit(whole, 320))]).to(self.device)
        with amp_context(self.device):
            pools = options_for(self.model, inputs, Tokenizer())
        result = select(pools[0] + pools[1], 'primary_mbr')
        return dict(text=result['text'], model='PARSeq v12', weights_sha256=self.weights_sha256, device=str(self.device), input_pill_bbox_xywh=bbox, automatic_ROI_within_tight_crop_xyxy=roi_box, ROI_is_GT_verified=False, orientation_candidates=8, raw_sequence_score=result['score'], score_is_calibrated=False, teacher_used=False, GT_or_medicine_lookup_used=False)
