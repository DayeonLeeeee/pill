# 처방 대조 판정
# 위치 실패 / 검색 보류 / 각인 충돌 -> 판단 보류, 1순위가 처방에 있으면 일치 후보, 아니면 불일치 의심
import re

from pill_models.labels import COLORS
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


imprint_faces = None


def unique_imprint(ocr, catalog):
    # 읽은 각인(4글자 이상)이 DB에서 딱 한 약과만 맞으면 그 약을 돌려준다
    global imprint_faces
    if imprint_faces is None:
        imprint_faces = [(item, text.translate(LOOKALIKE)) for item, row in catalog.items()
                         for text in (read_face(row, "front")[0], read_face(row, "back")[0]) if len(text) >= 4]
    q = norm(ocr).translate(LOOKALIKE)
    if len(q) < 4:
        return None
    hits = {item for item, text in imprint_faces
            if abs(len(text) - len(q)) <= len(q) // 10 and 1 - edit_distance(q, text) / max(len(q), len(text)) >= 0.9}
    return hits.pop() if len(hits) == 1 else None


def judge(pred, catalog, rx_items, bbox_ok=True):
    if not bbox_ok:
        return "판단 보류", "알약 위치를 찾지 못함", [], []

    search = pred["search"]
    ranks = {c["item"]: (n, round(c["score"], 4)) for n, c in enumerate(search["candidates"], 1)}
    order = [c["item"] for c in search["candidates"][:TOP_K]]
    # 각인이 한 약을 가리키면 색·모양 점수보다 각인을 믿는다 (실제 사진은 조명 때문에 색이 자주 틀림)
    only = unique_imprint(pred["ocr"]["text"], catalog)
    # OCR이 그럴듯한 각인을 지어내는 경우가 있어서, 두 각도 이상에서 비슷하게 읽히고 모양도 맞을 때만 믿는다
    if only and (pred["ocr"].get("agreeing_angles", 0) < 2 or pred["shape"]["label"] != catalog[only]["drug_shape"]):
        only = None
    if only:
        order = [only] + [i for i in order if i != only][:len(order) - 1]
    top = [{"item": i, "name": catalog[i]["name"], "rank": ranks[i][0], "score": ranks[i][1],
            "in_rx": i in rx_items, "match": compare(pred, catalog[i])}
           for i in order]
    first = top[0]
    rx_found = ", ".join(t["name"] for t in top if t["in_rx"])
    reasons = []

    if only:
        reasons.append(f"각인 '{norm(pred['ocr']['text'])}'이(가) DB에서 [{first['name']}] 하나와만 일치 (점수 검색으로는 {first['rank']}위)")
        wrong = [ATTR_NAME[k] for k in ("color", "shape", "line") if not first["match"][k]["ok"]]
        if wrong:
            reasons.append(", ".join(wrong) + "은(는) DB와 다르게 읽힘 (조명·배경 영향일 수 있음)")
        if first["in_rx"]:
            return "일치 후보", f"처방약 [{first['name']}]과 일치", reasons, top
        reasons.append(f"[{first['name']}]은 오늘 처방에 없음")
        return "불일치 의심", "처방에 없는 약으로 보임", reasons, top

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
