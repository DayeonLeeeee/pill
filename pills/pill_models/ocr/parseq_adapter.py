"""PARSeq square pretrained OCR, adapted to reviewed pill crops and exact spaces/case.

Upstream network is Apache-2.0 by Darwin Bautista; see vendor_parseq/NOTICE.txt.
"""
from pathlib import Path
import json, math
import torch
from torch import nn
from torch.nn import functional as F
from .model import Config, VOCAB, PAD, BOS, EOS, UNK
from .vendor_parseq.model import PARSeq
from .vendor_parseq.tokenizer import Tokenizer as UpstreamTokenizer

class PillPARSeq(nn.Module):

    def __init__(self, config, pretrained=False):
        if (config.d_model, config.heads, config.layers) not in [(384, 12, 1), (768, 24, 1)] or config.image_size % 16:
            raise ValueError('지원 구조는 (d_model, decoder heads, decoder layers)=(384,12,1) 또는 (768,24,1), 입력 크기는 16의 배수입니다.')
        super().__init__()
        self.config = config
        self.tokenizer = UpstreamTokenizer(config.vocab)
        self.network = PARSeq(len(self.tokenizer), config.max_length, (config.image_size, config.image_size), (16, 16), config.d_model, config.d_model // 64, 4, 12, config.heads, 4, config.layers, True, 0, config.dropout)
        to_native = [self.tokenizer.pad_id, self.tokenizer.bos_id, self.tokenizer.eos_id, self.tokenizer.pad_id] + [self.tokenizer._stoi[c] for c in config.vocab]
        to_ours = [EOS] + [4 + config.vocab.index(c) for c in config.vocab]
        self.register_buffer('to_native', torch.tensor(to_native), persistent=False)
        self.register_buffer('to_ours', torch.tensor(to_ours), persistent=False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225])[None, :, None, None], persistent=False)
        if pretrained:
            self.load_ocr_pretrained()

    @property
    def encoder(self):
        return self.network.encoder

    def load_ocr_pretrained(self):
        description = json.loads((Path(__file__).parent / 'vendor_parseq/pretraining.json').read_text('utf-8'))
        original = torch.hub.load_state_dict_from_url(description['weights_url'], map_location='cpu', check_hash=True, weights_only=True)
        source_tokenizer = UpstreamTokenizer(description['charset'])
        target = self.network.state_dict()
        if set(original) != set(target):
            raise ValueError('PARSeq upstream parameter names do not match the adapter')
        with torch.no_grad():
            for key, dst in target.items():
                src = original[key]
                if key == 'encoder.pos_embed':
                    side = math.isqrt(src.size(1))
                    size = self.config.image_size // 16
                    if side * side != src.size(1):
                        raise ValueError('Expected square pretrained image positions')
                    target[key] = F.interpolate(src.reshape(1, side, side, 384).permute(0, 3, 1, 2), size=(size, size), mode='bicubic', align_corners=False).permute(0, 2, 3, 1).reshape(1, size * size, 384)
                elif key == 'pos_queries':
                    dst[:, :src.size(1)].copy_(src)
                elif key in ('head.weight', 'head.bias', 'text_embed.embedding.weight'):
                    if key.startswith('head.'):
                        dst[0].copy_(src[0])
                        for c in self.config.vocab:
                            if c in source_tokenizer._stoi:
                                dst[self.tokenizer._stoi[c]].copy_(src[source_tokenizer._stoi[c]])
                        if key == 'head.bias':
                            dst[self.tokenizer._stoi[' ']] = -2.0
                    else:
                        for symbol, index in self.tokenizer._stoi.items():
                            if symbol in source_tokenizer._stoi:
                                dst[index].copy_(src[source_tokenizer._stoi[symbol]])
                else:
                    if src.shape != dst.shape:
                        raise ValueError('Unexpected pretrained shape ' + key)
                    target[key] = src
            self.network.load_state_dict(target, strict=True)

    def native_images(self, images):
        return (images * self.std + self.mean) * 2 - 1

    def encode(self, images):
        return self.network.encode(self.native_images(images))

    def forward(self, images, tokens):
        native = self.to_native[tokens]
        length = native.size(1)
        mask = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool), 1)
        padding = native.eq(self.tokenizer.pad_id) | native.eq(self.tokenizer.eos_id)
        memory = self.encode(images)
        values = self.network.head(self.network.decode(native, memory, mask, padding, tgt_query_mask=mask))
        result = values.new_full((*values.shape[:-1], len(self.config.vocab) + 4), -10000.0)
        result[:, :, self.to_ours] = values
        return result

    @torch.inference_mode()
    def predict(self, images, max_length=None):
        logits = self.network.forward(self.tokenizer, self.native_images(images), max_length)
        probabilities = logits.float().softmax(-1)
        scores, native = probabilities.max(-1)
        positions = torch.arange(native.size(1), device=native.device)[None, :].expand_as(native)
        first_eos = torch.where(native.eq(self.tokenizer.eos_id), positions, native.size(1)).min(1).values
        active = positions <= first_eos[:, None]
        confidence = (scores.clamp_min(1e-09).log().mul(active).sum(1) / active.sum(1).clamp_min(1)).exp()
        return (self.to_ours[native], confidence)
