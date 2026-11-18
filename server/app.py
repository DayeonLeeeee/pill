# 실행: uvicorn app:app --port 8000
import asyncio
import io
import json
import os
import re
import sqlite3
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

HERE = Path(__file__).parent
BUNDLE = Path(os.environ.get("BUNDLE_ROOT", HERE.parent / "pills")).resolve()
DATA = Path(os.environ.get("DATA_DIR", HERE / "data"))
IMAGES = DATA / "images"
IMAGES.mkdir(parents=True, exist_ok=True)
KST = timezone(timedelta(hours=9))

sys.path.insert(0, str(BUNDLE))
from autobbox import find_pill  # noqa: E402
from ocr_rotation import use_max_score  # noqa: E402
from verdict import judge  # noqa: E402

config = json.loads((HERE / "demo_config.json").read_text("utf8"))
with open(BUNDLE / "database/catalog.jsonl", encoding="utf8") as f:
    catalog = {r["item"]: r for r in map(json.loads, f)}

model = None
model_error = None
model_lock = threading.Lock()  # 모델 하나를 요청끼리 나눠 씀


def load_model():
    global model, model_error
    try:
        from pill_models import PillRecognizer
        m = PillRecognizer(BUNDLE, device=os.environ.get("DEVICE", "auto"), threads=int(os.environ.get("TORCH_THREADS", 2)))
        use_max_score(m)
        # 첫 요청이 느리지 않게 한 번 돌려 둔다 (DB 검색도 이때 로드됨)
        ex = json.loads((BUNDLE / "examples/inputs.json").read_text("utf8"))[0]
        m.predict(BUNDLE / "examples" / ex["image"], bbox=ex["bbox"], search=True)
        model = m
    except Exception as e:
        model_error = repr(e)
        print("model load failed:", model_error)


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=load_model, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


def db():
    con = sqlite3.connect(DATA / "logs.sqlite3")
    con.row_factory = sqlite3.Row
    return con


with db() as con:
    con.execute("""create table if not exists logs (
        id integer primary key autoincrement, created_at text, patient_id text, target_item text,
        image_id text, bbox text, verdict text, reason text, top1_item text, top1_name text,
        status text, result_json text, actions text default '[]')""")
    con.execute("create table if not exists patients (id text primary key, name text)")
    con.execute("create table if not exists rx (patient_id text, item text, label text, position integer, primary key (patient_id, item))")
    # 처음 켤 때만 demo_config.json의 환자·처방을 넣는다
    if not con.execute("select 1 from patients").fetchone():
        for p in config["patients"]:
            con.execute("insert into patients values (?, ?)", (p["id"], p["name"]))
            for n, r in enumerate(p["rx"]):
                con.execute("insert into rx values (?, ?, ?, ?)", (p["id"], r["item"], r["label"], n))

START_STATUS = {"일치 후보": "담당자 확인 대기", "불일치 의심": "투약 중지 · 약사 확인 요청", "판단 보류": "약사 확인 대기"}
ACTIONS = {
    "administered": "투약 완료",
    "stopped": "투약 중지",
    "resolved_match": "처리 완료 · 처방약 맞음",
    "resolved_mismatch": "처리 완료 · 처방약 아님(회수)",
    "retake": "처리 완료 · 재촬영 요청",
}


def now():
    return datetime.now(KST).isoformat(timespec="seconds")


def log_dict(row):
    d = dict(row)
    d["bbox"] = json.loads(d["bbox"])
    d["actions"] = json.loads(d["actions"])
    d["result"] = json.loads(d.pop("result_json"))
    d["patient_name"] = patient_name(d["patient_id"])
    return d


def patient_name(patient_id):
    with db() as con:
        row = con.execute("select name from patients where id = ?", (patient_id,)).fetchone()
    return row["name"] if row else f"(삭제된 환자 {patient_id})"


def rx_items(patient_id):
    with db() as con:
        return [r["item"] for r in con.execute("select item from rx where patient_id = ? order by position", (patient_id,))]


def drug_info(item):
    c = catalog[item]
    return {
        "item": item, "name": c["name"],
        "color": "+".join(x for x in (c["color_class1"], c["color_class2"]) if x),
        "shape": c["drug_shape"],
        "imprint": " / ".join(x for x in ("".join(c["print_front"]), "".join(c["print_back"])) if x),
    }


def image_path(image_id):
    path = IMAGES / f"{Path(image_id).name}.png"
    if not path.exists():
        raise HTTPException(404, "이미지 없음")
    return path


def save_image(data):
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise HTTPException(400, "이미지를 열 수 없습니다")
    image_id = uuid.uuid4().hex
    im.save(IMAGES / f"{image_id}.png")
    return image_id, im


@app.get("/api/health")
def health():
    return {"ready": model is not None, "error": model_error}


@app.get("/api/patients")
def get_patients():
    with db() as con:
        people = con.execute("select * from patients order by id").fetchall()
        rx = con.execute("select * from rx order by position").fetchall()
    return [{"id": p["id"], "name": p["name"],
             "rx": [{**drug_info(r["item"]), "label": r["label"]} for r in rx if r["patient_id"] == p["id"]]}
            for p in people]


class PatientIn(BaseModel):
    name: str


@app.post("/api/patients")
def add_patient(p: PatientIn):
    if not p.name.strip():
        raise HTTPException(400, "이름을 입력하세요")
    with db() as con:
        ids = [r["id"] for r in con.execute("select id from patients")]
        n = max([int(i[1:]) for i in ids if i[1:].isdigit()] + [0]) + 1
        new_id = f"P{n:02d}"
        con.execute("insert into patients values (?, ?)", (new_id, p.name.strip()))
    return {"id": new_id}


@app.put("/api/patients/{patient_id}")
def rename_patient(patient_id: str, p: PatientIn):
    with db() as con:
        if not con.execute("update patients set name = ? where id = ?", (p.name.strip(), patient_id)).rowcount:
            raise HTTPException(404, "환자 없음")
    return {"ok": True}


@app.delete("/api/patients/{patient_id}")
def delete_patient(patient_id: str):
    # 확인 기록은 남긴다
    with db() as con:
        con.execute("delete from rx where patient_id = ?", (patient_id,))
        if not con.execute("delete from patients where id = ?", (patient_id,)).rowcount:
            raise HTTPException(404, "환자 없음")
    return {"ok": True}


class RxIn(BaseModel):
    item: str
    label: str = ""


@app.post("/api/patients/{patient_id}/rx")
def add_rx(patient_id: str, r: RxIn):
    if r.item not in catalog:
        raise HTTPException(404, "DB에 없는 품목코드")
    with db() as con:
        if not con.execute("select 1 from patients where id = ?", (patient_id,)).fetchone():
            raise HTTPException(404, "환자 없음")
        pos = con.execute("select coalesce(max(position), -1) + 1 from rx where patient_id = ?", (patient_id,)).fetchone()[0]
        con.execute("insert or replace into rx values (?, ?, ?, ?)", (patient_id, r.item, r.label.strip(), pos))
    return {"ok": True}


@app.put("/api/patients/{patient_id}/rx/{item}")
def edit_rx(patient_id: str, item: str, r: RxIn):
    with db() as con:
        if not con.execute("update rx set label = ? where patient_id = ? and item = ?", (r.label.strip(), patient_id, item)).rowcount:
            raise HTTPException(404, "처방 없음")
    return {"ok": True}


@app.delete("/api/patients/{patient_id}/rx/{item}")
def delete_rx(patient_id: str, item: str):
    with db() as con:
        con.execute("delete from rx where patient_id = ? and item = ?", (patient_id, item))
    return {"ok": True}


@app.get("/api/drugs")
def search_drugs(q: str):
    # 처방 추가용: 이름, 품목코드, 각인으로 찾기
    q = q.strip()
    if not q:
        return []
    key = re.sub("[^A-Z0-9]", "", q.upper())
    hits = []
    for item, c in catalog.items():
        imprint = re.sub("[^A-Z0-9]", "", "".join(c["print_front"] + c["print_back"]).upper())
        if q in c["name"] or q.upper() == item or (len(key) >= 2 and key in imprint):
            hits.append(drug_info(item))
            if len(hits) == 30:
                break
    return hits


@app.post("/api/upload")
async def upload(image: UploadFile = File(...)):
    image_id, im = save_image(await image.read())
    return {"image_id": image_id, "width": im.width, "height": im.height, "bbox": find_pill(np.asarray(im))}


@app.get("/api/images/{image_id}")
def get_image(image_id: str):
    return FileResponse(image_path(image_id))


def run_model(path, bbox):
    with model_lock:
        pred = model.predict(path, bbox=bbox, search=True, top_k=len(catalog), include_probabilities=True)  # 전체 순위 (각인으로 끌어올린 약의 점수도 보이게)
        pred["ocr"]["agreeing_angles"] = model.ocr_reader.last["agreeing_angles"]
        return pred


@app.post("/api/predict")
async def predict(
    patient_id: str = Form(...),
    image_id: str | None = Form(None),
    bbox: str | None = Form(None),
    target_item: str | None = Form(None),
    image: UploadFile | None = File(None),
):
    if model is None:
        raise HTTPException(503, model_error or "모델 로딩 중입니다. 잠시 후 다시 시도하세요.")
    with db() as con:
        if not con.execute("select 1 from patients where id = ?", (patient_id,)).fetchone():
            raise HTTPException(404, "환자 없음")
    if image:
        image_id, im = save_image(await image.read())
    elif image_id:
        im = Image.open(image_path(image_id)).convert("RGB")
    else:
        raise HTTPException(400, "image 또는 image_id가 필요합니다")

    box = json.loads(bbox) if bbox else find_pill(np.asarray(im))
    rx = rx_items(patient_id)
    pred = None
    if box:
        try:
            pred = await asyncio.to_thread(run_model, image_path(image_id), [int(v) for v in box])
        except ValueError as e:
            print("predict failed:", e)

    if pred:
        verdict, reason, reasons, top = judge(pred, catalog, rx)
        attrs = {
            "color": pred["color"]["top3"],
            "shape": pred["shape"]["top3"],
            "line": {"label": pred["score_line"]["label"], "presence_probability": pred["score_line"]["presence_probability"]},
            "imprint": {"text": pred["ocr"]["text"]},
        }
        s = pred["search"]
        search = {"withheld": s["withheld"], "score_margin": s["score_margin"], "margin_threshold": s["margin_threshold"]}
    else:
        verdict, reason, reasons, top = judge(None, catalog, rx, bbox_ok=False)
        attrs = search = None

    result = {"verdict": verdict, "reason": reason, "reasons": reasons, "bbox": box,
              "attrs": attrs, "top_k": top, "search": search, "target_item": target_item}
    with db() as con:
        cur = con.execute(
            "insert into logs (created_at, patient_id, target_item, image_id, bbox, verdict, reason, top1_item, top1_name, status, result_json)"
            " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now(), patient_id, target_item, image_id, json.dumps(box), verdict, reason,
             top[0]["item"] if top else None, top[0]["name"] if top else None,
             START_STATUS[verdict], json.dumps(result, ensure_ascii=False)))
    return {**result, "log_id": cur.lastrowid, "image_id": image_id}


@app.get("/api/logs")
def logs(patient_id: str | None = None):
    with db() as con:
        if patient_id:
            rows = con.execute("select * from logs where patient_id = ? order by id desc limit 100", (patient_id,))
        else:
            rows = con.execute("select * from logs order by id desc limit 100")
        return [log_dict(r) for r in rows]


@app.get("/api/review_queue")
def review_queue():
    # 판단 보류·불일치 의심 중 약사가 아직 처리 안 한 것
    with db() as con:
        rows = con.execute("select * from logs where verdict in ('판단 보류', '불일치 의심')"
                           " and status not like '처리 완료%' order by id desc")
        return [log_dict(r) for r in rows]


class Action(BaseModel):
    action: str
    actor: str = ""
    note: str = ""


@app.post("/api/logs/{log_id}/action")
def log_action(log_id: int, a: Action):
    if a.action not in ACTIONS:
        raise HTTPException(400, "알 수 없는 action")
    with db() as con:
        row = con.execute("select actions from logs where id = ?", (log_id,)).fetchone()
        if not row:
            raise HTTPException(404)
        history = json.loads(row["actions"]) + [{"at": now(), **a.model_dump()}]
        con.execute("update logs set status = ?, actions = ? where id = ?",
                    (ACTIONS[a.action], json.dumps(history, ensure_ascii=False), log_id))
        return log_dict(con.execute("select * from logs where id = ?", (log_id,)).fetchone())


@app.get("/api/demo_cases")
def demo_cases():
    return [{**c, "image_url": f"/api/demo_image/{i}"} for i, c in enumerate(config["stage_cases"])]


@app.get("/api/demo_image/{i}")
def demo_image(i: int):
    if not 0 <= i < len(config["stage_cases"]):
        raise HTTPException(404)
    return FileResponse(BUNDLE / config["stage_cases"][i]["example_image"])


@app.middleware("http")
async def no_cache(request, call_next):
    # 화면 파일을 브라우저가 예전 것으로 들고 있지 않게 (서버만 새 버전이면 화면이 깨짐)
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/", StaticFiles(directory=HERE / "static", html=True))
