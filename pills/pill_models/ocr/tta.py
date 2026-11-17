"""Deterministic rotation consensus. No lexicon, item ID, or expected text is used."""
from collections import Counter, defaultdict
import torch
from .model import EOS
POLICIES = ['none', 'max_score', 'vote_original', 'vote_score', 'character_vote']

def choose(options, policy):
    if policy == 'none':
        return dict(options[0])
    if policy == 'max_score':
        return dict(max(options, key=lambda p: p['score']))
    if policy in ['vote_original', 'vote_score']:
        counts = Counter((p['text'] for p in options))
        n = max(counts.values())
        tied = {t for t, v in counts.items() if v == n}
        if policy == 'vote_original' and options[0]['text'] in tied:
            return dict(options[0])
        return dict(max((p for p in options if p['text'] in tied), key=lambda p: p['score']))
    if policy == 'character_vote':
        counts = Counter((len(p['text']) for p in options))
        n = max(counts.values())
        lengths = {k for k, v in counts.items() if v == n}
        length = len(options[0]['text']) if len(options[0]['text']) in lengths else len(max((p for p in options if len(p['text']) in lengths), key=lambda p: p['score'])['text'])
        group = [p for p in options if len(p['text']) == length]
        anchor = options[0] if len(options[0]['text']) == length else max(group, key=lambda p: p['score'])
        result = []
        for i in range(length):
            freq = Counter((p['text'][i] for p in group))
            n = max(freq.values())
            chars = {c for c, v in freq.items() if v == n}
            result.append(anchor['text'][i] if anchor['text'][i] in chars else max((p for p in group if p['text'][i] in chars), key=lambda p: p['score'])['text'][i])
        return dict(text=''.join(result), score=None, terminated=all((p['terminated'] for p in group)), angle=None)
    raise ValueError('Unknown inference policy')

@torch.inference_mode()
def options_for(model, images, tokenizer, rotations=True):
    results = [[] for _ in range(len(images))]
    for turns in range(4) if rotations else [0]:
        batch = images if turns == 0 else torch.rot90(images, turns, (-2, -1)).contiguous()
        ids, scores = model.predict(batch)
        for entries, p, score in zip(results, ids.cpu(), scores.cpu().tolist()):
            entries.append(dict(text=tokenizer.decode(p), score=score, terminated=bool(p.eq(EOS).any()), angle=turns * 90))
    return results

@torch.inference_mode()
def recognize(model, images, tokenizer, policy='none'):
    results = []
    for options in options_for(model, images, tokenizer, rotations=policy != 'none'):
        result = choose(options, policy)
        result.update(original_score=options[0]['score'], original_text=options[0]['text'])
        results.append(result)
    return results
