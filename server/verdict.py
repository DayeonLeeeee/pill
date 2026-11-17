# 처방 대조 판정
# 1) 각인을 믿을 수 있으면: 각인으로 거르고 색 -> 분할선 -> 모양 순으로 고른다
# 2) 아니면: 패키지의 점수 검색 (네 속성 같은 비중, 점수 차가 기준보다 작으면 보류)
# 1순위가 처방에 있으면 일치 후보, 없으면 불일치 의심
import re

from pill_models.labels import COLOR_SETS, COLORS, SHAPES
from pill_models.search import edit_distance

TOP_K = 5
IMPRINT_MIN = 0.5  # 임시값. 검증셋으로 다시 정할 것

# DB 각인 정답에 섞여 있는 설명어
NOTES = ("십자분할선", "분할선", "마크")
# OCR이 자주 헷갈리는 글자
LOOKALIKE = str.maketrans("ODQILSBZG", "000115826")
LINE_NAME = {"": "없음", "-": "-", "+": "+", "기타": "기타"}
ATTR_NAME = {"color": "색", "shape": "모양", "line": "분할선", "imprint": "각인"}


def norm(s):
    return re.sub("[^A-Z0-9]", "", s.upper())


def read_face(row, side):
    # 반환: (영문·숫자만 남긴 각인, 면 종류)
    # 종류: text 글자만 / partial 글자+로고·한글 / logo 로고·한글만 / blank 없음
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
        return None  # 로고는 OCR이 못 읽음
    if kind == "blank":
        return 1.0 if len(q) <= 1 else 0.0  # 한 글자는 잡음으로 봄
    a, b = norm(q).translate(LOOKALIKE), ref.translate(LOOKALIKE)
    if not a:
        return 0.0
    sim = 1 - edit_distance(a, b) / max(len(a), len(b))
    # 일부만 읽혀도 인정
    if len(a) >= 2 and (a in b or (b in a and (len(b) >= 2 or kind == "partial"))):
        sim = max(sim, 0.9)
    return sim


def match_imprint(ocr, row):
    # 어느 면을 찍었는지 모르니 더 잘 맞는 면을 쓴다
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
    colors = set(pred["color"]["colors"])
    db_colors = {c.strip() for k in ("color_class1", "color_class2")
                 for c in (row[k] or "").split(",") if c.strip() in COLORS}
    imprint = match_imprint(pred["ocr"]["text"], row)
    line = pred["score_line"]["label"]
    db_line = LINE_NAME.get(row["line_" + imprint["face"]], "?")
    shape = pred["shape"]["label"]
    return {
        "color": {"ok": colors == db_colors, "pred": "+".join(sorted(colors)), "catalog": "+".join(sorted(db_colors)) or "?"},
        "shape": {"ok": shape == row["drug_shape"], "pred": shape, "catalog": row["drug_shape"]},
        "line": {"ok": line == db_line, "pred": line, "catalog": db_line},
        "imprint": imprint,
    }


# 비슷한 색은 부분 점수 (0.5). 짝은 서로 헷갈리기 쉬운 이웃 색
NEAR_COLORS = [
    ("하양", "회색"), ("하양", "노랑"), ("노랑", "주황"), ("노랑", "연두"), ("주황", "빨강"), ("주황", "갈색"),
    ("분홍", "빨강"), ("분홍", "자주"), ("빨강", "갈색"), ("연두", "초록"), ("초록", "청록"), ("청록", "파랑"),
    ("파랑", "남색"), ("남색", "보라"), ("남색", "검정"), ("보라", "자주"), ("회색", "검정"),
]
NEAR = {frozenset(p) for p in NEAR_COLORS}
CONF_MIN = 0.5  # 이 속성으로 1·2위가 갈릴 때, 모델 확신도가 이보다 낮으면 보류


def color_sim(a, b):
    return 1.0 if a == b else 0.5 if frozenset((a, b)) in NEAR else 0.0


def color_set_sim(pred, db):
    if not db:
        return 0.5  # DB에 색 정보 없음
    total = sum(max(color_sim(a, b) for b in db) for a in pred) + sum(max(color_sim(a, b) for a in pred) for b in db)
    return total / (len(pred) + len(db))


def db_colors(row):
    return {c.strip() for k in ("color_class1", "color_class2") for c in (row[k] or "").split(",") if c.strip() in COLORS}


def attr_scores(pred, row, face):
    probs = pred["probabilities"]
    colors = db_colors(row)
    color = sum(p * color_set_sim({COLORS[i] for i in COLOR_SETS[k]}, colors) for k, p in enumerate(probs["color"]) if p > 1e-4)
    line_id = {"": 0, "-": 1, "+": 2}.get(row["line_" + face])
    line = probs["score_line"][line_id] if line_id is not None else probs["score_line"][1] + probs["score_line"][2]
    shape = probs["shape"][SHAPES.index(row["drug_shape"])] if row["drug_shape"] in SHAPES else 0.5
    return round(color, 2), round(line, 2), round(shape, 2)


def imprint_candidates(q, catalog):
    # 편집거리로 비교. 빈 면은 보지 않는다.
    # DB 각인이 실제보다 짧게 적힌 경우가 있어서(예: 실제 GS 5YZ 300, DB엔 GS5YZ) 4글자 이상이 통째로
    # 들어 있으면 0.85를 준다. 완전히 맞는 약(1.0)이 있으면 0.85는 거르는 범위(최고 -0.1) 밖이라 끼지 않는다.
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


def imprint_first(pred, catalog):
    """각인으로 거른 뒤 색 -> 분할선 -> 모양 순으로 고른다. OCR을 못 믿을 땐 None (기존 점수 검색 사용)."""
    q = norm(pred["ocr"]["text"])
    if not q or pred["ocr"].get("agreeing_angles", 0) < 2:
        return None
    sims = imprint_candidates(q, catalog)
    best = max(s for s, _ in sims.values())
    if best < 0.6:
        return None
    group = [i for i, (s, _) in sims.items() if s >= best - 0.1]
    # 확신도가 낮은 속성은 순서 정하는 데 쓰지 않는다 (모두 같은 값으로)
    use = [conf >= CONF_MIN for _, conf in attr_confidence(pred)]
    keyed = sorted(((tuple(s if u else 0 for s, u in zip(attr_scores(pred, catalog[i], sims[i][1]), use)), sims[i][0], i)
                    for i in group), reverse=True)
    return [i for _, _, i in keyed], {i: k for k, _, i in keyed}, [name for (name, _), u in zip(attr_confidence(pred), use) if not u]


def attr_confidence(pred):
    return [("색", pred["color"]["confidence"]), ("분할선", max(pred["probabilities"]["score_line"])),
            ("모양", pred["shape"]["confidence"])]


def judge(pred, catalog, rx_items, bbox_ok=True):
    if not bbox_ok:
        return "판단 보류", "알약 위치를 찾지 못함", [], []

    search = pred["search"]
    ranks = {c["item"]: (n, round(c["score"], 4)) for n, c in enumerate(search["candidates"], 1)}
    baseline = [c["item"] for c in search["candidates"][:TOP_K]]

    def rows(items):
        return [{"item": i, "name": catalog[i]["name"], "rank": ranks[i][0], "score": ranks[i][1],
                 "in_rx": i in rx_items, "match": compare(pred, catalog[i])} for i in items]

    found = imprint_first(pred, catalog)
    if found:
        group, keys, skipped = found
        top = rows((group + [i for i in baseline if i not in group])[:TOP_K])
        first = top[0]
        q = norm(pred["ocr"]["text"])
        reasons = [f"각인 '{q}'(으)로 거른 후보 {len(group)}개" + ("" if len(group) > 1 else f": [{first['name']}] (점수 검색으로는 {first['rank']}위)")]
        if skipped and len(group) > 1:
            reasons.append(", ".join(skipped) + f"은(는) 확신도가 {CONF_MIN:.0%} 미만이라 쓰지 않음")
        if len(group) > 1:
            second = group[1]
            diff = next((n for n in range(3) if keys[group[0]][n] != keys[second][n]), None)
            if diff is None:
                ties = sum(keys[i] == keys[group[0]] for i in group)
                reasons.append(f"남은 속성까지 같은 약이 {ties}개라 사진만으로 구별 안 됨")
                if any(t["in_rx"] for t in top):
                    reasons.append("후보 중 처방약: " + ", ".join(t["name"] for t in top if t["in_rx"]))
                return "판단 보류", "후보끼리 구분이 어려움", reasons, top
            reasons.append(f"{attr_confidence(pred)[diff][0]}(으)로 1순위 결정 (2순위 [{catalog[second]['name']}])")
        wrong = [ATTR_NAME[k] for k in ("color", "shape", "line") if not first["match"][k]["ok"]]
        if wrong:
            reasons.append(", ".join(wrong) + "은(는) DB와 다르게 읽힘 (조명·배경 영향일 수 있음)")
        if first["in_rx"]:
            return "일치 후보", f"처방약 [{first['name']}]과 일치", reasons, top
        reasons.append(f"[{first['name']}]은 오늘 처방에 없음")
        return "불일치 의심", "처방에 없는 약으로 보임", reasons, top

    # 각인을 못 믿을 때: 기존 점수 검색 (색·모양·분할선·각인 같은 비중)
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
