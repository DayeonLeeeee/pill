from pathlib import Path
import json
from pill_models import PillRecognizer

root=Path(__file__).resolve().parent
example=json.loads((root/'examples/inputs.json').read_text('utf8'))[0]

# Load once. Subsequent predict calls reuse all resident models.
model=PillRecognizer(device='auto')
result=model.predict(root/'examples'/example['image'],bbox=example['bbox'],search=True)
print(json.dumps(result,ensure_ascii=False,indent=2))

# For your own already-cropped image:
# result=model.predict('my_pill_crop.png',cropped=True,search=False)
