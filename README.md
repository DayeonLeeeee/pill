# pills 데모 서버

모델 패키지(pills/, 받은 PillModels4 폴더 이름만 바꾼 것)를 불러 쓰는 FastAPI 서버. 화면(static/index.html)도 같이 띄운다.

```
pills/            저장소
├─ pills/         모델 패키지 (weights 포함)
├─ server/
│  ├─ app.py
│  ├─ verdict.py         처방 대조 판정
│  ├─ autobbox.py        알약 위치 찾기 (단색 배경 가정, 화면에서 드래그로 수정 가능)
│  ├─ demo_config.json   환자, 처방(품목코드), 무대 사례
│  ├─ fetch_weights.py   배포할 때 가중치 받기
│  └─ static/index.html
└─ Dockerfile
```

## 노트북에서

```bat
cd pills\server
..\pills\.venv\Scripts\python.exe -m pip install fastapi uvicorn python-multipart opencv-python-headless
..\pills\.venv\Scripts\python.exe -m uvicorn app:app --port 8000
```

http://localhost:8000 . 오른쪽 위가 "AI 준비됨"이 되면 시작 (10~20초).

## Railway

- `pills/weights/`, `server/data/`는 .gitignore에 넣고 올린다.
- 가중치 4개는 GitHub Release 같은 곳에 올리고 환경변수 `WEIGHTS_URL_BASE`에 주소를 넣는다.
- 메모리 2GB 정도 필요.
- 기록(SQLite)과 사진은 `data/`에 저장돼서 재배포하면 지워진다. 남기려면 Volume을 `/app/data`에 연결.

## API

| | 경로 | |
|---|---|---|
| GET | /api/health | 모델 준비 여부 |
| GET | /api/patients | 환자와 처방 |
| POST | /api/upload | 사진 저장 + 자동 bbox |
| POST | /api/predict | form: patient_id, image_id 또는 image, bbox(생략하면 자동), target_item |
| GET | /api/logs?patient_id= | 확인 기록 |
| GET | /api/review_queue | 약사 확인 목록 |
| POST | /api/logs/{id}/action | {action, actor, note} — administered, stopped, resolved_match, resolved_mismatch, retake |
| GET | /api/demo_cases | 무대 사례 |

## 판정

1. 위치 못 찾음 → 판단 보류
2. DB 검색 보류 (1·2위 점수 차 < 0.0667, 패키지 기준) → 판단 보류
3. 각인이 1순위와 다름 (유사도 < 0.5, 임시값) → 판단 보류
4. 1순위가 처방에 있으면 일치 후보, 없으면 불일치 의심

각인 비교는 DB 정답에서 '분할선', '마크', 빈칸을 지우고, O/0 같은 헷갈리는 글자는 같게 본다. 로고만 있는 면은 OCR이 못 읽으니 충돌로 치지 않는다.
