# 경구약제 색상·외형·각인·분할선 추론 패키지

네 모델의 실제 가중치, 추론 코드, 예제 7장과 선택적 4,525품목 DB 검색이 포함되어 있습니다. 학습용 전체 데이터나 학습 코드는 포함하지 않습니다.

## Windows에서 바로 실행

1. ZIP을 새 폴더에 완전히 풉니다. ZIP 내부에서 직접 실행하지 마세요.
2. Python 3.12 64-bit를 설치합니다.
3. CPU는 `01_setup_cpu.cmd`, NVIDIA GPU는 `02_setup_nvidia.cmd`를 실행합니다. 처음 설치할 때 인터넷이 필요하며, 패키지 폴더의 `.venv`에만 설치합니다. 시스템 Python 패키지를 변경하지 않습니다.
4. CPU는 `03_run_example_cpu.cmd`, GPU는 `04_run_example_nvidia.cmd`를 실행합니다.
5. 예제 결과는 `results/example_cpu.jsonl` 또는 `results/example_cuda.jsonl`에 저장됩니다.

가중치는 이미 모두 포함되어 있습니다. 의존성 설치 후 추론할 때 모델 다운로드, Qwen, 원래 컴퓨터의 폴더, 학습 GT가 필요하지 않습니다. 예제는 사용법 검사용이며 성능 평가 자료가 아닙니다.

설치 버전은 실제 검증 환경인 PyTorch 2.10.0 / torchvision 0.25.0 / timm 1.0.24 / NumPy 2.3.5 / Pillow 13.0.0으로 고정했습니다. NVIDIA 설치는 CUDA 12.8용 공식 휠을 사용합니다. 공식 설치 명령 참고: https://pytorch.org/get-started/previous-versions/

## 자신의 사진 하나

전체 사진에는 원본 이미지 픽셀 좌표의 알약 BBOX가 필요합니다. **X Y WIDTH HEIGHT** 순서이며 X1 Y1 X2 Y2가 아닙니다. 여러 알약 사진은 한 알약씩 BBOX를 지정해 호출하세요.

```bat
.venv\Scripts\python.exe predict.py --image "my_photo.png" --bbox 100 200 300 180 --device auto --search --output results/my_photo.jsonl
```

이미 알약 주변만 잘라낸 사진이라면:

```bat
.venv\Scripts\python.exe predict.py --image "my_crop.png" --cropped --device cpu
```

`--search`를 생략하면 네 속성만 반환합니다. 자동 알약 검출은 포함하지 않습니다. `--device cuda`는 CUDA 사용 불가 시 오류를 내며, `auto`는 사용 가능한 CUDA 또는 CPU를 선택합니다.

## Python 코드에서 사용

압축을 푼 폴더를 작업 폴더로 열거나, 그 폴더를 Python 모듈 경로에 추가하세요. 실행 예제는 `example_usage.py`입니다.

```python
from pill_models import PillRecognizer

model = PillRecognizer(device="auto")  # 한 번 로드하고 재사용
result = model.predict("my_photo.png", bbox=[100, 200, 300, 180], search=True)
print(result["color"])
print(result["shape"])
print(result["ocr"]["text"])
print(result["score_line"])
print(result["search"])
```

## 여러 사진

JSON 배열 또는 JSONL에 다음처럼 적습니다. 상대 이미지 경로는 입력 manifest가 있는 폴더를 기준으로 해석합니다. `--data-root`로 기준 폴더를 별도 지정할 수 있습니다.

```json
{"image":"images/a.png","bbox":[100,200,300,180]}
{"image":"crops/b.png","cropped":true}
```

```bat
.venv\Scripts\python.exe predict.py --inputs inputs.jsonl --device auto --search --output results/batch.jsonl
```

모델은 배치 파일 전체에서 한 번만 로드합니다. 사진은 순서대로 한 장씩 추론하므로 단일 사진 API와 같은 호출 방식입니다.

## 모델과 출력

| 모델 | 입력과 구조 | 출력 |
|---|---|---|
| 색상 v4 | RGB 224, 독립 계층 ResNet18 | 단색·두 색 조합 120개 Softmax |
| 외형 v4 | Grayscale 224, 독립 계층 ResNet18 | 모양 11개 Softmax |
| 각인 v12 | 알약 전체+자동 ROI, 320, 각 4방향, PARSeq MBR | 각인 문자열 |
| 분할선 확장 v1 | RGB 384, ConvNeXt-Tiny, 검증 선택 epoch 1 | 없음 / - / +, 유무 임계값 0.52 |

분할선 `기타` 유형은 학습 자료가 없어 미지원입니다. `score_line.other_supported`가 false인 이유입니다. OCR의 `raw_sequence_score`와 각 속성 Softmax는 보정된 정답 확률을 의미하지 않습니다.

## 선택 기능: DB 검색

현재 ablation의 기준 검색을 그대로 제공합니다. 색상·외형·각인·분할선 점수를 동일 가중치로 결합하고, 각인과 분할선은 같은 카탈로그 면에 맞춥니다. 두 면 중 어느 쪽인지 정답을 추론에 주지 않습니다.

`search.candidates`는 후보 순위이고 `search.withheld`가 true이면 보류입니다. 해시 순서로 동점 후보 하나가 먼저 표시될 수 있지만 그것이 식별 근거는 아닙니다. 점수와 margin은 보정된 정답 확률이 아닙니다.

기본 DB와 임계값에서는 기존 외부 4,100장 Top-1 69.41%, 보류율 49.54%, 수락 정확도 99.13%였습니다. 별도 기존 진단 4,050장에서는 같은 기준의 수락 정확도가 61.29%로 낮았습니다. 이를 임의의 새 사진에서도 보장되는 수치로 해석하지 마세요. 원래 자료는 새로운 맹검 평가가 아닙니다.

기호 `/`, `-`를 이용한 동점 보정 실험은 이 패키지에 기본 적용하지 않았습니다. `database/search_policy.json`에 명시된 기준판입니다. DB를 바꾸면 임계값을 다시 검증해야 하므로 DB 해시가 달라지면 검색 모듈은 오류를 냅니다.

## 무결성 검사와 실제 모델 검사

```bat
py -3.12 verify_package.py
.venv\Scripts\python.exe self_test.py --device cpu --output results/self_test_cpu.json
.venv\Scripts\python.exe self_test.py --device cuda --output results/self_test_gpu.json
```

가중치와 파일 해시, 실제 네 모델 추론, DB 검색, 잘못된 BBOX 거부, 네트워크 다운로드 없이 추론하는 동작을 검사합니다. 제작 측 실행 기록은 `verification/`에 있습니다. CPU/GPU에서 부동소수점 연산 차이로 아주 작은 확률 차이는 생길 수 있습니다.

## 파일 구성

- `weights/`: 실제 네 모델 가중치
- `pill_models/`: 독립된 Python 추론 라이브러리. 다른 프로그램의 common/model/data 모듈과 충돌하지 않도록 OCR 코드를 패키지 안으로 분리했습니다.
- `predict.py`, `example_usage.py`: CLI와 Python API 예제
- `examples/`: 첨부받은 PARSeq 패키지의 예제 이미지 7장과 좌표
- `database/`: 원본 카탈로그 메타데이터와 기준 검색 설정
- `가중치_위치와_버전.md`, `model_manifest.json`: 원본 위치·모델 버전·SHA-256
- `provenance/`, `THIRD_PARTY_NOTICES_OCR.md`, `pill_models/ocr/vendor_parseq/LICENSE`: 출처 및 PARSeq 고지

제작 시 기존 코드·모델·GT를 변경하지 않고 복사본으로 구성했습니다. 원본 절대경로는 출처 문서에만 기록하며 실행에는 사용하지 않습니다.
