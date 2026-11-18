import re

from pill_models.labels import COLOR_SETS, COLORS, SHAPES
from pill_models.search import edit_distance

TOP_K = 5
IMPRINT_MIN = 0.5  # TODO 검증셋으로 다시

NOTES = ("십자분할선", "분할선", "마크")
LOOKALIKE = str.maketrans("ODQILSBZG", "000115826")
LINE_NAME = {"": "없음", "-": "-", "+": "+", "기타": "기타"}
ATTR_NAME = {"color": "색", "shape": "모양", "line": "분할선", "imprint": "각인"}


def norm(s):
    return re.sub("[^A-Z0-9]", "", s.upper())


def read_face(row, side):
    # kind: text / partial(글자+로고) / logo / blank
    text, extra = "", False
    for p in row["print_" + side]:
        extra |= "마크" in p
        for w in NOTES:
            p = p.replace(w, " ")
        text += norm(p)
        extra |= bool(re.search(r"[^\sA-Za-z0-9\-./()\[\],+·]", p))
    if text:
        return text, "partial" if extra else "text"
    return "", "logo" if extra else "blank"


def similarity(q, ref, kind):
    if kind == "logo":
        return None
    if kind == "blank":
        return 1.0 if len(q) <= 1 else 0.0
    a, b = norm(q).translate(LOOKALIKE), ref.translate(LOOKALIKE)
    if not a:
        return 0.0
    sim = 1 - edit_distance(a, b) / max(len(a), len(b))
    if len(a) >= 2 and (a in b or (b in a and (len(b) >= 2 or kind == "partial"))):
        sim = max(sim, 0.9)
    return sim


def match_imprint(ocr, row):
    q = norm(ocr)
    best, face, logo = None, "front", False
    for side in ("front", "back"):
        s = similarity(q, *read_face(row, side))
        if s is None:
            logo = True
        elif best is None or s > best:
            best, face = s, side
    ok = logo or (best is not None and best >= IMPRINT_MIN)
    return {"ok": ok, "pred": q, "similarity": best, "face": face,
            "catalog": {s: [t for t in row["print_" + s] if t] for s in ("front", "back")}}


def imprint_conflict(ocr, row):
    q = norm(ocr)
    if not q:
        both = all(read_face(row, s)[1] == "text" for s in ("front", "back"))
        return both, "각인을 읽지 못함(후보는 양면 모두 글자 각인)"
    m = match_imprint(ocr, row)
    if m["ok"]:
        return False, ""
    return True, f"읽은 각인 '{q}'이(가) 1순위 후보와 다름(유사도 {m['similarity']:.2f})"


def compare(pred, row):
    probs = pred["probabilities"]
    colors = set(pred["color"]["colors"])
    db_colors = {c.strip() for k in ("color_class1", "color_class2")
                 for c in (row[k] or "").split(",") if c.strip() in COLORS}
    color_score = 0.5 if not db_colors else sum(
        p * len({COLORS[i] for i in COLOR_SETS[k]} & db_colors) / len({COLORS[i] for i in COLOR_SETS[k]} | db_colors)
        for k, p in enumerate(probs["color"]) if p > 1e-4)
    imprint = match_imprint(pred["ocr"]["text"], row)
    line = pred["score_line"]["label"]
    db_line = LINE_NAME.get(row["line_" + imprint["face"]], "?")
    line_id = {"없음": 0, "-": 1, "+": 2}.get(db_line)
    line_score = probs["score_line"][line_id] if line_id is not None else 0.5
    shape = pred["shape"]["label"]
    shape_score = probs["shape"][SHAPES.index(row["drug_shape"])] if row["drug_shape"] in SHAPES else 0.5
    return {
        "color": {"ok": colors == db_colors, "pred": "+".join(sorted(colors)), "catalog": "+".join(sorted(db_colors)) or "?",
                  "score": round(color_score, 3)},
        "shape": {"ok": shape == row["drug_shape"], "pred": shape, "catalog": row["drug_shape"], "score": round(shape_score, 3)},
        "line": {"ok": line == db_line, "pred": line, "catalog": db_line, "score": round(line_score, 3)},
        "imprint": {**imprint, "score": None if imprint["similarity"] is None else round(imprint["similarity"], 3)},
    }


SHAPE_CONF_MIN = 0.5


def imprint_candidates(q, catalog):
    # DB 각인이 짧게 적힌 약이 있어서 (GS5YZ300 vs GS5YZ) 포함되면 0.85
    a = q.translate(LOOKALIKE)
    sims = {}
    for item, row in catalog.items():
        for side in ("front", "back"):
            ref = read_face(row, side)[0].translate(LOOKALIKE)
            if not ref:
                continue
            s = 1 - edit_distance(a, ref) / max(len(a), len(ref))
            if min(len(a), len(ref)) >= 4 and (ref in a or a in ref):
                s = max(s, 0.85)
            if s > sims.get(item, (-1, ""))[0]:
                sims[item] = (s, side)
    return sims


def imprint_first(pred, catalog, ranks):
    q = norm(pred["ocr"]["text"])
    if not q or pred["ocr"].get("agreeing_angles", 0) < 2:
        return None
    sims = imprint_candidates(q, catalog)
    best = max(s for s, _ in sims.values())
    if best < 0.6:
        return None
    group = [i for i, (s, _) in sims.items() if s >= best - 0.1]
    group.sort(key=lambda i: (-round(sims[i][0], 2), -ranks[i][1], ranks[i][0]))
    return group, sims


def judge(pred, catalog, rx_items, bbox_ok=True):
    if not bbox_ok:
        return "판단 보류", "알약 위치를 찾지 못함", [], []

    search = pred["search"]
    ranks = {c["item"]: (n, round(c["score"], 4)) for n, c in enumerate(search["candidates"], 1)}
    baseline = [c["item"] for c in search["candidates"][:TOP_K]]

    def rows(items):
        return [{"item": i, "name": catalog[i]["name"], "rank": ranks[i][0], "score": ranks[i][1],
                 "in_rx": i in rx_items, "match": compare(pred, catalog[i])} for i in items]

    found = imprint_first(pred, catalog, ranks)
    if found:
        group, sims = found
        top = rows((group + [i for i in baseline if i not in group])[:TOP_K])
        first = top[0]
        q = norm(pred["ocr"]["text"])
        reasons = [f"각인 '{q}'(으)로 거른 후보 {len(group)}개" + ("" if len(group) > 1 else f": [{first['name']}] (점수 검색으로는 {first['rank']}위)")]
        if len(group) > 1:
            second = top[1]
            if round(sims[first["item"]][0], 2) > round(sims[second["item"]][0], 2):
                reasons.append(f"각인이 가장 잘 맞음 (2순위 [{second['name']}])")
            else:
                gap = first["score"] - second["score"]
                threshold = search["margin_threshold"]
                if gap < threshold:
                    reasons.append(f"각인이 같은 후보끼리 점수 차 {gap:.3f} < 기준 {threshold:.3f}")
                    if any(t["in_rx"] for t in top):
                        reasons.append("후보 중 처방약: " + ", ".join(t["name"] for t in top if t["in_rx"]))
                    return "판단 보류", "후보끼리 구분이 어려움", reasons, top
                reasons.append(f"각인이 같은 후보 중 점수가 가장 높음 (2순위 [{second['name']}], 차 {gap:.3f})")
        # OCR이 다른 약 각인을 지어내는 경우 (YH CNT 등)
        shape = first["match"]["shape"]
        if not shape["ok"] and shape["pred"] != "기타" and pred["shape"]["confidence"] >= SHAPE_CONF_MIN:
            reasons.append(f"그런데 모양이 다름 (사진 {shape['pred']}, DB {shape['catalog']}) - 각인을 잘못 읽었을 수 있음")
            return "판단 보류", "각인과 모양이 맞지 않음", reasons, top
        wrong = [ATTR_NAME[k] for k in ("color", "line") if not first["match"][k]["ok"]]
        if wrong:
            reasons.append(", ".join(wrong) + "은(는) DB와 다르게 읽힘 (조명·배경 영향일 수 있음)")
        if first["in_rx"]:
            return "일치 후보", f"처방약 [{first['name']}]과 일치", reasons, top
        reasons.append(f"[{first['name']}]은 오늘 처방에 없음")
        return "불일치 의심", "처방에 없는 약으로 보임", reasons, top

    # OCR 못 믿으면 패키지 검색 그대로
    top = rows(baseline)
    first = top[0]
    rx_found = ", ".join(t["name"] for t in top if t["in_rx"])
    reasons = ["각인을 믿을 수 없어 점수 검색 사용"]

    if search["withheld"]:
        msg = f"1·2순위 점수 차 {search['score_margin']:.3f} < 기준 {search['margin_threshold']:.3f}"
        if search["top_score_tie_count"] > 1:
            msg += f" (1위 동점 {search['top_score_tie_count']}개)"
        reasons.append(msg)
        if rx_found:
            reasons.append("후보 중 처방약: " + rx_found)
        return "판단 보류", "후보끼리 구분이 어려움", reasons, top

    conflict, why = imprint_conflict(pred["ocr"]["text"], catalog[first["item"]])
    if conflict:
        return "판단 보류", "각인이 후보와 불일치", [why], top

    same = [ATTR_NAME[k] for k in ATTR_NAME if first["match"][k]["ok"]]
    reasons.append("1순위 후보와 같은 속성: " + (", ".join(same) or "없음"))
    if first["in_rx"]:
        return "일치 후보", f"처방약 [{first['name']}]과 일치", reasons, top

    reasons.append(f"[{first['name']}]은 오늘 처방에 없음")
    if rx_found:
        reasons.append("하위 후보 중 처방약: " + rx_found)
    return "불일치 의심", "처방에 없는 약으로 보임", reasons, top
