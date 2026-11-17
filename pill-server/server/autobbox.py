# 단색 배경 위 알약 한 알의 위치(x, y, w, h)를 찾는다. 못 찾으면 None.
import cv2
import numpy as np


def find_pill(rgb):
    H, W = rgb.shape[:2]
    scale = 512 / max(H, W)
    img = cv2.resize(rgb, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]

    # 테두리 색을 배경으로 본다
    b = max(4, min(h, w) // 25)
    edge = np.concatenate([lab[:b], lab[-b:], lab[:, :b].transpose(1, 0, 2), lab[:, -b:].transpose(1, 0, 2)], axis=None)
    bg = np.median(edge.reshape(-1, 3), axis=0)

    d = lab - bg
    # 그림자(배경보다 어둡고 색은 비슷)를 덜 잡도록 어두워지는 쪽은 약하게
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
        # 크고 가운데에 가까운 덩어리
        off = np.hypot(centers[i][0] - w / 2, centers[i][1] - h / 2) / np.hypot(w / 2, h / 2)
        score = area * (1 - 0.7 * off)
        if score > best_score:
            best, best_score = (x, y, bw, bh), score
    if best is None:
        return None

    x, y, bw, bh = (round(v / scale) for v in best)
    pad = round(0.06 * max(bw, bh))  # 학습 때 박스처럼 여백을 조금
    x, y = max(0, x - pad), max(0, y - pad)
    bw, bh = min(bw + 2 * pad, W - x), min(bh + 2 * pad, H - y)
    if min(bw, bh) < 16:
        return None
    return [x, y, bw, bh]
