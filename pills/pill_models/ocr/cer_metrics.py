"""Character edit metrics. Whitespace may be ignored; case/punctuation are retained."""

def normalize(text, ignore_whitespace=False):
    return ''.join(text.split()) if ignore_whitespace else text

def distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(prediction, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]

def summarize(records, ignore_whitespace=False):
    total = errors = characters = exact = blank = blank_fp = 0
    similarity = 0.0
    for p in records:
        ref = normalize(p['expected'], ignore_whitespace)
        pred = normalize(p['predicted'], ignore_whitespace)
        d = distance(ref, pred)
        total += 1
        errors += d
        characters += len(ref)
        exact += ref == pred
        similarity += 1 - d / max(1, len(ref), len(pred))
        if not ref:
            blank += 1
            blank_fp += bool(pred)
    return dict(samples=total, character_edits=errors, reference_characters=characters, cer=errors / characters if characters else None, mean_normalized_edit_similarity=similarity / max(1, total), exact_match_reference_only=exact / max(1, total), blank_reference_samples=blank, blank_false_positive_samples=blank_fp, blank_false_positive_rate=blank_fp / blank if blank else None, whitespace_ignored=ignore_whitespace, case_sensitive=True, punctuation_preserved=True)

def report(records, training_texts=()):
    known = {normalize(t, True) for t in training_texts}
    unseen = [p for p in records if normalize(p['expected'], True) not in known]
    return dict(primary=summarize(records, True), raw=summarize(records, False), unseen_whitespace_normalized_training=summarize(unseen, True), definition='CER=(substitutions+deletions+insertions)/reference characters; lower is better. Similarity is mean 1-edit_distance/max(reference_length,prediction_length,1).')
