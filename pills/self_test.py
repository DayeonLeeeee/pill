"""All four real models, offline guard, namespaced imports, and input validation."""
from pathlib import Path
import argparse,json,sys,types,socket,time,math,hashlib
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--output');args=p.parse_args()
    # Integration with applications that already have common/model/data modules must work.
    sentinels={name:types.ModuleType(name) for name in ['common','model','data','runtime','labels']}
    for name,module in sentinels.items():sys.modules[name]=module
    def no_network(*a,**k):raise AssertionError('Inference must not access the network')
    socket.create_connection=no_network
    import torch
    torch.hub.load_state_dict_from_url=no_network
    from pill_models import PillRecognizer
    from pill_models.search import edit_distance
    assert edit_distance('ABC','ADC')==1 and edit_distance('AB','BA')==2 and edit_distance('','A')==1
    start=time.perf_counter();model=PillRecognizer(ROOT,device=args.device);loaded=time.perf_counter()
    examples=json.loads((ROOT/'examples/inputs.json').read_text('utf8'));results=[]
    for row in examples[:2]:
        path=ROOT/'examples'/row['image'];result=model.predict(path,bbox=row['bbox'],search=True,include_probabilities=True)
        assert len(result['probabilities']['color'])==120 and len(result['probabilities']['shape'])==11 and len(result['probabilities']['score_line'])==3
        for value in result['probabilities'].values():assert abs(sum(value)-1)<1e-4 and all(math.isfinite(x) for x in value)
        assert result['ocr']['orientation_candidates']==8 and result['score_line']['other_supported'] is False
        assert result['search']['catalog_items']==4525 and len(result['search']['candidates'])==5
        assert result['search']['withheld']==(result['search']['score_margin']<result['search']['margin_threshold'])
        results.append(result)
    testpath=ROOT/'examples'/examples[0]['image']
    for kwargs in [dict(),dict(bbox=[0,0,0,10]),dict(bbox=[0,0,1,1]),dict(bbox=[-1,0,2,2]),dict(bbox=examples[0]['bbox'],cropped=True)]:
        try:model.predict(testpath,**kwargs)
        except ValueError:pass
        else:raise AssertionError('Invalid crop request was silently accepted')
    assert all(sys.modules[name] is module for name,module in sentinels.items())
    report=dict(passed=True,device=str(model.device),real_images=len(results),all_four_models_executed=True,optional_DB_executed=True,
        no_network_download=True,namespace_collision_check=True,invalid_bbox_rejection=True,
        load_seconds=loaded-start,total_seconds=time.perf_counter()-start,results=results)
    if args.output:
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=False))
if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf8')
    main()
