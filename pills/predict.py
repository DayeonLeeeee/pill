"""Single image or manifest batch. This script can be run from any working directory."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description='색상·외형·각인·분할선 추론. 알약 검출은 포함하지 않습니다.')
    inp=p.add_mutually_exclusive_group(required=True);inp.add_argument('--image');inp.add_argument('--inputs',help='JSON list or JSONL of image + bbox records')
    b=p.add_mutually_exclusive_group();b.add_argument('--bbox',nargs=4,type=int,metavar=('X','Y','W','H'));b.add_argument('--cropped',action='store_true')
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda']);p.add_argument('--search',action='store_true');p.add_argument('--top-k',type=int,default=5);p.add_argument('--output');p.add_argument('--data-root');p.add_argument('--probabilities',action='store_true');p.add_argument('--threads',type=int,default=8)
    args=p.parse_args()
    if args.image and args.bbox is None and not args.cropped:p.error('전체 사진은 --bbox X Y W H, 이미 잘라낸 사진은 --cropped가 필요합니다.')
    if args.inputs and (args.bbox is not None or args.cropped):p.error('배치 입력은 각 행에 bbox 또는 cropped:true를 지정하세요.')
    from pill_models import PillRecognizer
    model=PillRecognizer(ROOT,device=args.device,threads=args.threads)
    if args.image:records=[dict(image=args.image,bbox=args.bbox,cropped=args.cropped)];base=Path.cwd()
    else:
        manifest=Path(args.inputs).resolve();text=manifest.read_text('utf-8-sig');records=json.loads(text) if text.lstrip().startswith('[') else [json.loads(l) for l in text.splitlines() if l.strip()];base=Path(args.data_root).resolve() if args.data_root else manifest.parent
    if not records:raise ValueError('Input manifest is empty')
    output=Path(args.output).resolve() if args.output else None
    if output:output.parent.mkdir(parents=True,exist_ok=True)
    f=output.open('w',encoding='utf8') if output else None
    try:
        for r in records:
            path=Path(r['image']);path=path if path.is_absolute() else base/path
            result=model.predict(path,bbox=r.get('bbox'),cropped=r.get('cropped',False),search=args.search,top_k=args.top_k,include_probabilities=args.probabilities)
            line=json.dumps(result,ensure_ascii=False)
            if f:f.write(line+'\n');f.flush()
            else:print(line,flush=True)
    finally:
        if f:f.close()

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf8')
    main()
