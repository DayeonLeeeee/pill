"""Reproduces the frozen study baseline. Experimental tie corrections are not enabled."""
from pathlib import Path
from itertools import combinations
import hashlib,json,re
import numpy as np
from .labels import COLORS,COLOR_SETS,SHAPES

def normalize(text):return re.sub('[^A-Z0-9]','',str(text).upper())
def edit_distance(a,b):
    if not a:return len(b)
    eqs={}
    for i,ch in enumerate(a):eqs[ch]=eqs.get(ch,0)|(1<<i)
    pv=~0;mv=0;score=len(a);last=1<<(len(a)-1)
    for ch in b:
        eq=eqs.get(ch,0);xv=eq|mv;xh=(((eq&pv)+pv)^pv)|eq;ph=mv|~(xh|pv);mh=pv&xh
        if ph&last:score+=1
        elif mh&last:score-=1
        ph=(ph<<1)|1;mh<<=1;pv=mh|~(xv|ph);mv=ph&xv
    return score

class CatalogSearch:
    def __init__(self,root):
        root=Path(root);path=root/'database/catalog.jsonl';self.policy=json.loads((root/'database/search_policy.json').read_text('utf8'))
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=self.policy['calibrated_for_catalog_sha256']:raise ValueError('Catalog changed; bundled margin calibration no longer applies. Recalibrate before using acceptance decisions.')
        with path.open(encoding='utf8') as f:self.catalog=[json.loads(l) for l in f]
        self.order=np.array(sorted(range(len(self.catalog)),key=lambda i:hashlib.sha256((self.catalog[i]['item']+'|'+str(self.policy['seed'])).encode()).hexdigest()))
        self.color=np.full((120,len(self.catalog)),.5,dtype=np.float32);self.shape=[];self.lines=[];self.refs=sorted({normalize(t) for r in self.catalog for side in ['front','back'] for t in r['print_'+side]}|{''});ri={t:i for i,t in enumerate(self.refs)};self.refids=[]
        for j,r in enumerate(self.catalog):
            color={COLORS.index(t.strip()) for k in ['color_class1','color_class2'] for t in str(r[k] or '').split(',') if t.strip() in COLORS}
            if color:
                for k,c in enumerate(COLOR_SETS):c=set(c);self.color[k,j]=len(c&color)/len(c|color)
            self.shape.append(SHAPES.index(r['drug_shape']) if r['drug_shape'] in SHAPES else -1)
            self.lines.append([{'':0,'-':1,'+':2,'기타':3}.get(r['line_'+side],-1) for side in ['front','back']])
            self.refids.append([[ri[normalize(t)] for t in r['print_'+side]] or [ri['']] for side in ['front','back']])
        self.shape=np.array(self.shape);self.lines=np.array(self.lines)
    def search(self,color,shape,line,text,top_k=5):
        if not isinstance(top_k,int) or not 1<=top_k<=len(self.catalog):raise ValueError('top_k must be an integer within the catalog size')
        q=normalize(text);sim=np.array([1-edit_distance(q,t)/max(len(q),len(t)) if q and t else 0 for t in self.refs],dtype=np.float32)
        cs=np.asarray(color,dtype=np.float32)@self.color;ss=np.full(len(self.catalog),.5,dtype=np.float32);known=self.shape>=0;ss[known]=np.asarray(shape)[self.shape[known]]
        face=np.empty((len(self.catalog),2),dtype=np.float32);l=np.asarray(line,dtype=np.float32)
        for f in [0,1]:
            ts=np.array([sim[self.refids[j][f]].max() for j in range(len(self.catalog))],dtype=np.float32);ls=np.full(len(self.catalog),.5,dtype=np.float32);ids=self.lines[:,f]
            for k in [0,1,2]:ls[ids==k]=l[k]
            ls[ids==3]=l[1:].sum();face[:,f]=ts+ls
        scores=np.round((cs+ss+face.max(1))/4,self.policy['score_round_decimals']);ranked=self.order[np.argsort(-scores[self.order],kind='stable')];margin=float(scores[ranked[0]]-scores[ranked[1]]);threshold=float(self.policy['margin_threshold']);ties=int((scores==scores[ranked[0]]).sum())
        return dict(catalog_items=len(self.catalog),policy=self.policy['version'],withheld=bool(margin<threshold),
            score_margin=margin,margin_threshold=threshold,top_score_tie_count=ties,
            reason='margin_below_validation_threshold' if margin<threshold else 'passes_bundled_validation_threshold',
            calibrated_probability=False,
            candidates=[dict(item=self.catalog[i]['item'],name=self.catalog[i]['name'],score=float(scores[i])) for i in ranked[:top_k]])
