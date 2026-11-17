"""Square pill image -> 2D CNN features -> autoregressive character sequence."""
import math, string
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
VOCAB = ''.join((chr(n) for n in range(32, 127)))
PAD, BOS, EOS, UNK = (0, 1, 2, 3)

class Tokenizer:

    def __init__(self, vocab=VOCAB):
        self.vocab = vocab
        self.ids = {c: n + 4 for n, c in enumerate(vocab)}

    def encode(self, text):
        if any((c not in self.ids for c in text)):
            raise ValueError('지원되지 않는 문자')
        return [BOS] + [self.ids[c] for c in text] + [EOS]

    def decode(self, ids):
        result = []
        for token in ids:
            token = int(token)
            if token == EOS:
                break
            if token >= 4:
                result.append(self.vocab[token - 4])
        return ''.join(result)

    def __len__(self):
        return len(self.vocab) + 4

@dataclass
class Config:
    image_size: int = 160
    d_model: int = 192
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    max_length: int = 32
    vocab: str = VOCAB
    architecture: str = 'resnet18_layer3_2d_transformer_v1'

def position_2d(h, w, dim, device, dtype):
    if dim % 4:
        raise ValueError('dimension must be divisible by four')
    bands = torch.exp(torch.arange(0, dim // 2, 2, device=device, dtype=torch.float32) * (-math.log(10000) / (dim // 2)))
    y = torch.arange(h, device=device, dtype=torch.float32)[:, None] * bands[None, :]
    x = torch.arange(w, device=device, dtype=torch.float32)[:, None] * bands[None, :]
    py = torch.cat((y.sin(), y.cos()), dim=1)[:, None, :].expand(h, w, dim // 2)
    px = torch.cat((x.sin(), x.cos()), dim=1)[None, :, :].expand(h, w, dim // 2)
    return torch.cat((py, px), dim=-1).reshape(1, h * w, dim).to(dtype)

class PillOCR(nn.Module):

    def __init__(self, config=None, pretrained=False):
        super().__init__()
        self.config = config or Config()
        c = self.config
        if c.max_length < 2:
            raise ValueError('max length too short')
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1, backbone.layer2, backbone.layer3)
        self.projection = nn.Conv2d(256, c.d_model, 1)
        self.embedding = nn.Embedding(len(c.vocab) + 4, c.d_model, padding_idx=PAD)
        self.token_position = nn.Embedding(c.max_length + 2, c.d_model)
        layer = nn.TransformerDecoderLayer(c.d_model, c.heads, 4 * c.d_model, c.dropout, activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, c.layers, norm=nn.LayerNorm(c.d_model))
        self.output = nn.Linear(c.d_model, len(c.vocab) + 4)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[PAD].zero_()
        nn.init.normal_(self.token_position.weight, std=0.02)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode(self, images):
        z = self.projection(self.encoder(images))
        h, w = z.shape[-2:]
        return z.flatten(2).transpose(1, 2) + position_2d(h, w, self.config.d_model, z.device, z.dtype)

    def decode_tokens(self, memory, tokens):
        if tokens.size(1) > self.config.max_length + 1:
            raise ValueError('정답을 조용히 자르지 않습니다.')
        pos = torch.arange(tokens.size(1), device=tokens.device)
        target = self.embedding(tokens) * math.sqrt(self.config.d_model) + self.token_position(pos)[None, :, :]
        causal = torch.triu(torch.ones(tokens.size(1), tokens.size(1), device=tokens.device, dtype=torch.bool), diagonal=1)
        return self.output(self.decoder(target, memory, tgt_mask=causal, tgt_key_padding_mask=tokens.eq(PAD)))

    def forward(self, images, tokens):
        return self.decode_tokens(self.encode(images), tokens)

    @torch.inference_mode()
    def predict(self, images, max_length=None):
        memory = self.encode(images)
        n = images.size(0)
        tokens = torch.full((n, 1), BOS, device=images.device, dtype=torch.long)
        done = torch.zeros(n, device=images.device, dtype=torch.bool)
        scores = []
        counts = torch.zeros(n, device=images.device)
        for _ in range(min(max_length or self.config.max_length, self.config.max_length) + 1):
            logits = self.decode_tokens(memory, tokens)[:, -1, :].float()
            logits[:, [PAD, BOS, UNK]] = -torch.inf
            probabilities = logits.softmax(-1)
            score, next_token = probabilities.max(-1)
            counts += (~done).float()
            next_token = torch.where(done, torch.full_like(next_token, EOS), next_token)
            scores.append(torch.where(done, torch.ones_like(score), score))
            tokens = torch.cat((tokens, next_token[:, None]), dim=1)
            done |= next_token.eq(EOS)
            if done.all():
                break
        confidence = (torch.stack(scores, dim=1).clamp_min(1e-09).log().sum(1) / counts.clamp_min(1)).exp()
        return (tokens[:, 1:], confidence)

def build_model(config, pretrained=False):
    if config.architecture in ('parseq_square_v1', 'parseq_wide_v6'):
        from .parseq_adapter import PillPARSeq
        return PillPARSeq(config, pretrained)
    if config.architecture != 'resnet18_layer3_2d_transformer_v1':
        raise ValueError('알 수 없는 OCR 모델 구조')
    return PillOCR(config, pretrained)
