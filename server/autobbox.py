import cv2
import numpy as np


def find_pill(rgb):
    H, W = rgb.shape[:2]
    scale = 512 / max(H, W)
    img = cv2.resize(rgb, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # 손에 올려 찍은 사진은 피부 사이에 뚫린 구멍을 찾는 게 제일 정확함
    best = find_on_skin(img)
    if best is not None:
        return to_original(best, scale, W, H)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]

    b = max(4, min(h, w) // 25)
    edge = np.concatenate([lab[:b], lab[-b:], lab[:, :b].transpose(1, 0, 2), lab[:, -b:].transpose(1, 0, 2)], axis=None)
    bg = np.median(edge.reshape(-1, 3), axis=0)

    d = lab - bg
    # 그림자 덜 잡게
    dl = np.where(d[..., 0] > 0, 0.5, 0.2) * d[..., 0]
    dist = np.sqrt(dl ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    n, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    best, best_score = None, 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not 0.002 * h * w < area < 0.6 * h * w:
            continue
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        off = np.hypot(centers[i][0] - w / 2, centers[i][1] - h / 2) / np.hypot(w / 2, h / 2)
        score = area * (1 - 0.7 * off)
        if score > best_score:
            best, best_score = (x, y, bw, bh), score
    if best is None:
        # 흰 알약 + 흰 바탕 + 조명 얼룩이면 색 차이로는 못 찾음.
        # 손바닥 위처럼 바탕도 밝으면 밝기로는 못 찾고 윤곽으로만 잡혀서 둘 다 보고 고름
        found = [c for c in (find_bright_blob(img), find_by_edges(img)) if c]
        if found:
            best = max(found, key=lambda c: c[1])[0]
    if best is None:
        return None
    return to_original(best, scale, W, H)


def to_original(box, scale, W, H):
    x, y, bw, bh = (round(v / scale) for v in box)
    pad = round(0.06 * max(bw, bh))
    x, y = max(0, x - pad), max(0, y - pad)
    bw, bh = min(bw + 2 * pad, W - x), min(bh + 2 * pad, H - y)
    if min(bw, bh) < 16:
        return None
    return [x, y, bw, bh]


def find_on_skin(img):
    h, w = img.shape[:2]
    skin = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb), (40, 135, 80), (255, 180, 135))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, k)
    if skin.mean() / 255 < 0.15:
        return None
    # 나무 책상도 피부색으로 잡힘. 나뭇결이 있어서 피부보다 훨씬 거칠다
    Y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if np.median(np.abs(Y - cv2.GaussianBlur(Y, (0, 0), 3))[skin > 0]) > 1.5:
        return None
    other = cv2.morphologyEx(cv2.bitwise_not(skin), cv2.MORPH_OPEN, k)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(other)
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    best, best_score = None, 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        if not 0.002 * h * w < area < 0.25 * h * w or max(bw, bh) > 3 * min(bw, bh):
            continue
        blob = (labels == i).astype(np.uint8)
        contour = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        solidity = area / max(cv2.contourArea(cv2.convexHull(contour)), 1)
        # 둘레가 거의 다 피부여야 손 위에 놓인 것
        ring = cv2.dilate(blob, ring_k) - blob
        if solidity < 0.8 or skin[ring > 0].mean() / 255 < 0.7:
            continue
        off = np.hypot(centers[i][0] - w / 2, centers[i][1] - h / 2) / np.hypot(w / 2, h / 2)
        score = area * solidity * (1 - off) ** 2
        if score > best_score:
            best, best_score = (x, y, bw, bh), score
    return best


def find_bright_blob(img):
    # 주변보다 밝은 덩어리 (조명 얼룩은 큰 블러로 빼냄)
    h, w = img.shape[:2]
    L = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    res = L - cv2.GaussianBlur(L, (0, 0), min(h, w) / 10)
    mad = np.median(np.abs(res - np.median(res))) + 1e-6
    mask = (res > max(6, 4 * mad)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        area = cv2.contourArea(cv2.convexHull(c))
        if not 0.001 * h * w < area < 0.25 * h * w or max(bw, bh) > 4 * min(bw, bh):
            continue
        # 글자·그림자 때문에 윤곽이 울퉁불퉁할 수 있어서 solidity는 느슨하게
        solidity = cv2.contourArea(c) / max(area, 1)
        if area / (bw * bh) < 0.6 or solidity < 0.4:
            continue
        # 가운데에서 멀면 크게 깎음 (가장자리 물건 경계가 잘 잡힘)
        off = np.hypot(x + bw / 2 - w / 2, y + bh / 2 - h / 2) / np.hypot(w / 2, h / 2)
        score = area * solidity * (1 - off) ** 2
        if score > best_score:
            best, best_score = (x, y, bw, bh), score
    return best and (best, best_score)


def find_by_edges(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0]
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    hi = max(8, np.percentile(grad, 97))
    edges = cv2.Canny(gray, hi * 0.5, hi)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if not 0.002 * h * w < area < 0.6 * h * w or max(bw, bh) > 4 * min(bw, bh):
            continue
        # 조명 얼룩은 테두리가 흐릿함. 알약 테두리는 또렷함
        ring = np.zeros_like(gray)
        cv2.drawContours(ring, [hull], -1, 255, 3)
        if grad[ring > 0].mean() < 5:
            continue
        # 알약은 볼록하고 꽉 찬 모양
        fill = area / (bw * bh)
        if fill < 0.6:
            continue
        solidity = cv2.contourArea(c) / max(area, 1)
        off = np.hypot(x + bw / 2 - w / 2, y + bh / 2 - h / 2) / np.hypot(w / 2, h / 2)
        score = area * solidity * (1 - off) ** 2
        if score > best_score:
            best, best_score = (x, y, bw, bh), score
    return best and (best, best_score)


def snap_to_pill(rgb, bbox):
    # 상자가 알약보다 1.3배쯤 커지면 shape 모델이 '기타'로 읽음
    H, W = rgb.shape[:2]
    x, y, w, h = bbox
    m = round(0.1 * max(w, h))
    X0, Y0 = max(0, x - m), max(0, y - m)
    X1, Y1 = min(W, x + w + m), min(H, y + h + m)
    found = find_pill(rgb[Y0:Y1, X0:X1])
    if not found:
        return bbox
    fx, fy, fw, fh = found
    if fw * fh < 0.15 * w * h:
        return bbox
    return [X0 + fx, Y0 + fy, fw, fh]
