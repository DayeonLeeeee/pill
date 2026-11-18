from pathlib import Path
from contextlib import nullcontext
import hashlib,json,math
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import convnext_tiny
from .appearance_models import ColorClassifier,LargeColorClassifier,ShapeClassifier
from .labels import COLORS,COLOR_NAMES,COLOR_SETS,SHAPES
from .ocr.reader import Reader
from .ocr.portable_data import choose_device

class ScoreModel(nn.Module):
    def __init__(self):
        super().__init__();net=convnext_tiny(weights=None,stochastic_depth_prob=.1);net.classifier=nn.Identity()
        self.backbone=net;self.norm=nn.LayerNorm(768);self.presence=nn.Linear(768,1);self.score_type=nn.Linear(768,2)
    def forward(self,x):
        z=self.norm(self.backbone(x).flatten(1));return self.presence(z).flatten(),self.score_type(z)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def fit(im,size):
    a=np.asarray(im);bg=tuple(int(v) for v in a[[0,0,-1,-1],[0,-1,0,-1]].mean(0));scale=size/max(im.size)
    im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(size,size),bg);canvas.paste(im,((size-im.width)//2,(size-im.height)//2));return canvas

def appearance_inputs(im,bbox):
    x,y,w,h=bbox;pad=max(8,int(max(w,h)*.12))
    wide=im.crop((max(0,x-pad),max(0,y-pad),min(im.width,x+w+pad),min(im.height,y+h+pad)))
    wide=fit(wide,320).resize((224,224),Image.Resampling.BILINEAR)
    c=torch.from_numpy(np.asarray(wide).copy()).permute(2,0,1).float()/127.5-1
    s=torch.from_numpy(np.asarray(wide.convert('L')).copy())[None].float()/127.5-1
    pad=max(4,round(max(w,h)*.03));tight=im.crop((max(0,x-pad),max(0,y-pad),min(im.width,x+w+pad),min(im.height,y+h+pad)))
    tight.thumbnail((384,384),Image.Resampling.LANCZOS)
    l=torch.from_numpy(np.asarray(fit(tight,384)).copy()).permute(2,0,1).float()/255
    l=(l-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    return c,s,l

def large_color_input(im,bbox):
    # 색 모델 v2: 같은 넓은 자르기, 320 그대로, ImageNet 정규화
    x,y,w,h=bbox;pad=max(8,int(max(w,h)*.12))
    wide=fit(im.crop((max(0,x-pad),max(0,y-pad),min(im.width,x+w+pad),min(im.height,y+h+pad))),320)
    c=torch.from_numpy(np.asarray(wide).copy()).permute(2,0,1).float()/255
    return (c-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]

def top(prob,names,n=3):
    order=np.argsort(-prob,kind='stable')[:n]
    return [dict(label=names[i],class_index=int(i),probability=float(prob[i])) for i in order]

class PillRecognizer:
    """Load once, predict many. Coordinates are original-image XYWH, not XYXY."""
    def __init__(self,bundle_root=None,device='auto',threads=8,verify=True):
        self.root=Path(bundle_root).resolve() if bundle_root else Path(__file__).resolve().parents[1]
        self.device=choose_device(device);torch.set_num_threads(int(threads))
        self.manifest=json.loads((self.root/'model_manifest.json').read_text('utf8'));self.database=None
        if verify:
            for name,entry in self.manifest['models'].items():
                path=self.root/entry['package_path']
                if sha(path)!=entry['sha256']:raise ValueError('Weight SHA-256 mismatch: '+name)
        color_state=torch.load(self.root/'weights/color.pt',map_location='cpu',weights_only=True)
        self.large_color=color_state.get('architecture')=='resnet50'
        self.color_model=LargeColorClassifier(120) if self.large_color else ColorClassifier(120)
        self.color_model.load_state_dict(color_state['model']);self.color_model.to(self.device).eval()
        self.shape_model=ShapeClassifier(11);self.score_model=ScoreModel()
        for model,name in [(self.shape_model,'shape.pt'),(self.score_model,'score.pt')]:
            state=torch.load(self.root/'weights'/name,map_location='cpu',weights_only=True);model.load_state_dict(state['model']);model.to(self.device).eval()
        self.ocr_reader=Reader(weights=self.root/'weights/parseq_v12.pt',device=str(self.device),threads=int(threads))

    def _amp(self):
        if self.device.type=='cpu':return nullcontext()
        return torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    @torch.inference_mode()
    def predict(self,image,bbox=None,*,cropped=False,search=False,top_k=5,include_probabilities=False):
        image=Path(image).resolve()
        with Image.open(image) as opened:
            im=opened.convert('RGB');width,height=im.size
        if bbox is None:
            if not cropped:raise ValueError('Provide bbox=[x,y,width,height], or cropped=True for an already-cropped pill image. Automatic detection is not included.')
            bbox=[0,0,width,height]
        elif cropped:raise ValueError('Choose bbox or cropped=True, not both')
        if len(bbox)!=4 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in bbox):raise ValueError('bbox must contain four finite XYWH numbers')
        if any(float(v)!=int(v) for v in bbox):raise ValueError('bbox coordinates must be integer pixels')
        bbox=list(map(int,bbox));x,y,w,h=bbox
        if min(x,y)<0 or min(w,h)<16 or x+w>width or y+h>height:raise ValueError('bbox must stay inside the original image and have width/height of at least 16 pixels')
        c,s,l=appearance_inputs(im,bbox)
        if self.large_color:c=large_color_input(im,bbox)
        with self._amp():
            cp=self.color_model(c[None].to(self.device)).float().softmax(-1)[0].cpu().numpy()
            sp=self.shape_model(s[None].to(self.device)).float().softmax(-1)[0].cpu().numpy()
            presence,typ=self.score_model(l[None].to(self.device));p=float(presence.float().sigmoid()[0]);tp=typ.float().softmax(-1)[0].cpu().numpy()
        lp=np.array([1-p,p*tp[0],p*tp[1]],dtype=np.float32)
        ci=int(cp.argmax());si=int(sp.argmax());positive=p>=self.manifest['score_presence_threshold'];ti=int(tp.argmax())
        ocr=self.ocr_reader.read(image,bbox)
        result=dict(image=str(image),bbox_xywh=bbox,automatic_detection=False,device=str(self.device),
            color=dict(label=COLOR_NAMES[ci],colors=[COLORS[j] for j in COLOR_SETS[ci]],class_index=ci,confidence=float(cp[ci]),top3=top(cp,COLOR_NAMES)),
            shape=dict(label=SHAPES[si],class_index=si,confidence=float(sp[si]),top3=top(sp,SHAPES)),
            ocr=dict(text=ocr['text'],raw_sequence_score=ocr['raw_sequence_score'],score_is_calibrated=False,orientation_candidates=8),
            score_line=dict(present=positive,presence_probability=p,presence_threshold=.52,label=['-','+'][ti] if positive else '없음',
                conditional_type_probabilities=dict(zip(['-','+'],map(float,tp))),other_supported=False))
        if include_probabilities:result['probabilities']=dict(color=cp.tolist(),shape=sp.tolist(),score_line=lp.tolist())
        if search:
            if self.database is None:
                from .search import CatalogSearch
                self.database=CatalogSearch(self.root)
            result['search']=self.database.search(cp,sp,lp,ocr['text'],top_k=top_k)
        return result
