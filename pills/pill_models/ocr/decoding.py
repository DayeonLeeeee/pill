"""Lexicon-free normalized voting, N-best edit-risk decoding, and attention ROI."""
from collections import Counter
import math
import torch
from .model import Tokenizer, EOS
from .tta import choose
from .cer_metrics import normalize, distance

def normalized_vote(options):
    counts = Counter((normalize(p['text'], True) for p in options))
    best = max(counts.values())
    keys = {k for k, v in counts.items() if v == best}
    return dict(max((p for p in options if normalize(p['text'], True) in keys), key=lambda p: p['score']))

def minimum_edit_risk(options, temperature=1.0, uniform=False):
    if uniform:
        weights = [1 / len(options)] * len(options)
    else:
        values = [p.get('log_weight', math.log(max(1e-09, p['score']))) for p in options]
        m = max(values)
        weights = [math.exp((v - m) / temperature) for v in values]
        total = sum(weights)
        weights = [v / total for v in weights]
    texts = [normalize(p['text'], True) for p in options]
    risks = [sum((w * distance(h, r) / max(1, len(r)) for r, w in zip(texts, weights))) for h in texts]
    index = min(range(len(options)), key=lambda i: (risks[i], -options[i]['score']))
    return dict(options[index], estimated_edit_risk=risks[index])

@torch.inference_mode()
def attentive_greedy(model, images):
    attentions = []

    def hook(module, args, output):
        attentions.append(output[1].detach())
    handle = model.network.decoder.layers[-1].cross_attn.register_forward_hook(hook)
    try:
        ids, scores = model.predict(images)
    finally:
        handle.remove()
    weights = torch.cat(attentions, dim=1)
    tok = Tokenizer()
    result = [dict(text=tok.decode(p), score=s, terminated=bool(p.eq(EOS).any())) for p, s in zip(ids.cpu(), scores.cpu().tolist())]
    return (result, weights)

@torch.inference_mode()
def beam_candidates(model, images, width=5):
    net = model.network
    memory = model.encode(images)
    b = len(images)
    v = net.head.out_features
    device = images.device
    bos = model.tokenizer.bos_id
    eos = model.tokenizer.eos_id
    native = torch.full((b, width, 1), bos, dtype=torch.long, device=device)
    logsum = torch.full((b, width), -torch.inf, device=device)
    logsum[:, 0] = 0
    done = torch.zeros((b, width), dtype=torch.bool, device=device)
    length = torch.zeros((b, width), device=device)
    expanded = memory.repeat_interleave(width, 0)
    forced = torch.zeros_like(done)
    max_chars = model.config.max_length
    nar = net.head(net.decode(native[:, 0, :], memory, tgt_query=net.pos_queries[:, :max_chars + 1].expand(b, -1, -1))).float().log_softmax(-1)
    for step in range(max_chars + 1):
        context = native.flatten(0, 1)
        query = net.pos_queries[:, step:step + 1].expand(b * width, -1, -1)
        logits = net.head(net.decode(context, expanded, tgt_query=query))[:, 0, :].float().log_softmax(-1).reshape(b, width, v)
        logits = torch.where(done[:, :, None], torch.full_like(logits, -torch.inf), logits)
        logits[:, :, eos] = torch.where(done, torch.zeros_like(logsum), logits[:, :, eos])
        if step == max_chars:
            keep = logits[:, :, eos].clone()
            logits.fill_(-torch.inf)
            logits[:, :, eos] = keep
        sums = logsum[:, :, None] + logits
        next_length = torch.where(done, length, torch.full_like(length, step + 1))
        ranks = sums / next_length[:, :, None].clamp_min(1)
        _, indices = ranks.flatten(1).topk(width, dim=1)
        parents = indices // v
        tokens = indices % v
        old_done = done.gather(1, parents)
        forced = forced.gather(1, parents) | (step == max_chars) & ~old_done
        native = torch.cat((native.gather(1, parents[:, :, None].expand(-1, -1, native.size(2))), tokens[:, :, None]), 2)
        logsum = sums.flatten(1).gather(1, indices)
        length = next_length.gather(1, parents)
        done = old_done | tokens.eq(eos)
        if done.all():
            break
    output = []
    tok = Tokenizer()
    for i in range(b):
        choices = []
        for j in range(width):
            ids = native[i, j, 1:]
            n = int(length[i, j])
            ids = ids[:n]
            text = tok.decode(model.to_ours[ids])
            ar = float(logsum[i, j] / max(1, n))
            nar_score = float(nar[i, torch.arange(n, device=device), ids].mean())
            choices.append(dict(text=text, score=math.exp(ar), ar_mean=ar, nar_mean=nar_score, ar_sum=float(logsum[i, j]), terminated=bool(done[i, j] and (not forced[i, j]))))
        output.append(choices)
    return output
