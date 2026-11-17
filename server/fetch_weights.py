# weights/에 가중치가 없으면 WEIGHTS_URL_BASE에서 받는다 (GitHub에는 100MB 넘는 파일을 못 올려서)
# 예: WEIGHTS_URL_BASE=https://github.com/<계정>/<저장소>/releases/download/v1
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

bundle = Path(os.environ.get("BUNDLE_ROOT", Path(__file__).parent.parent / "pills"))
models = json.loads((bundle / "model_manifest.json").read_text("utf8"))["models"]
base = os.environ.get("WEIGHTS_URL_BASE", "").rstrip("/")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


for name, m in models.items():
    path = bundle / m["package_path"]
    if path.exists() and sha256(path) == m["sha256"]:
        continue
    if not base:
        sys.exit(f"{path}가 없고 WEIGHTS_URL_BASE도 비어 있음")
    print("downloading", name)
    path.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(f"{base}/{name}", path)
    if sha256(path) != m["sha256"]:
        path.unlink()
        sys.exit(f"{name} 해시가 다름")
