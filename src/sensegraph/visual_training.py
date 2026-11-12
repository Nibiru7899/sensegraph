"""Supervised visual targets, losses and image-only decoding for SenseGraph-Net."""
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.ops import batched_nms
from .io import read_jsonl
from .perception import TYPE_CLASS

# Dense head order -> historical evaluator class indices.
BOX_TYPES = (1, 4, 5, 3, 6)
KEY_TYPES = (2, 7)
MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]


def preprocess(path, size=640):
    with Image.open(path) as source:
        image = source.convert('RGB'); w, h = image.size
        scale = min(size/w, size/h); nw, nh = round(w*scale), round(h*scale)
        left, top = (size-nw)//2, (size-nh)//2
        canvas = Image.new('RGB', (size, size), 'white')
        canvas.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
        rgb = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float()/255
    transform = torch.tensor([nw/w, nh/h, left, top, w, h], dtype=torch.float32)
    return (rgb-MEAN)/STD, transform


def transform_boxes(boxes, affine, inverse=False):
    sx, sy, left, top = affine[:4]
    scale = boxes.new_tensor([sx, sy, sx, sy]); shift = boxes.new_tensor([left, top, left, top])
    return (boxes-shift)/scale if inverse else boxes*scale+shift


def targets(record, affine, size=640):
    side=size//4
    valid=torch.zeros(side,side,dtype=torch.bool)
    sx,sy,l,t,w,h=affine.tolist()
    yy,xx=torch.meshgrid(torch.arange(side)*4+2,torch.arange(side)*4+2,indexing='ij')
    valid=(xx>=l)&(xx<l+w*sx)&(yy>=t)&(yy<t+h*sy)
    result={'valid':valid,'unsupported':0,'invalid':0,'counts':{},'collisions':{}}
    entries={'box':{},'key':{}}
    for node in record['nodes']:
        label=TYPE_CLASS.get(node['type'])
        if label is None:
            result['unsupported']+=1; continue
        b=torch.tensor(node['bbox'],dtype=torch.float32)
        if not torch.isfinite(b).all() or (b[:2]<0).any() or (b[2:] > torch.tensor([record['width'],record['height']])).any() or (b[2:]<=b[:2]).any():
            result['invalid']+=1; continue
        b=transform_boxes(b,affine)/4; center=(b[:2]+b[2:])/2; cell=center.floor().long()
        branch='box' if label in BOX_TYPES else 'key'; labels=BOX_TYPES if branch=='box' else KEY_TYPES
        c=labels.index(label); x,y=cell.tolist(); key=(c,y,x)
        entries[branch].setdefault(key,[]).append((b[2:]-b[:2],center-cell))
        result['counts'][str(label)]=result['counts'].get(str(label),0)+1
    for branch, labels in [('box',BOX_TYPES),('key',KEY_TYPES)]:
        heat=torch.zeros(len(labels),side,side); extent=torch.zeros(2*len(labels),side,side); offset=extent.clone()
        regression=torch.zeros(len(labels),side,side,dtype=torch.bool)
        for (c,y,x), objects in entries[branch].items():
            if len(objects)>1:
                name=str(labels[c]); result['collisions'][name]=result['collisions'].get(name,0)+len(objects)
            wh,off=objects[0]
            radius=1 if branch=='key' else max(1,int(float(wh.min())/12))
            # Gaussian truncated at 3r; exact center=1, matching the design contract.
            a=max(0,x-3*radius); b=min(side,x+3*radius+1); u=max(0,y-3*radius); v=min(side,y+3*radius+1)
            gy,gx=torch.meshgrid(torch.arange(u,v),torch.arange(a,b),indexing='ij')
            gaussian=torch.exp(-((gx-x).square()+(gy-y).square())/(2*(radius/3)**2))
            heat[c,u:v,a:b]=torch.maximum(heat[c,u:v,a:b],gaussian)
            if len(objects)==1:
                regression[c,y,x]=True; extent[2*c:2*c+2,y,x]=wh; offset[2*c:2*c+2,y,x]=off
        result[branch+'_heat']=heat; result[branch+'_extent']=extent; result[branch+'_offset']=offset; result[branch+'_mask']=regression
    return result


class VisualDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, size=640):
        if split not in ('train','val'):
            raise ValueError('Protected test split is not a training/development input')
        self.root=Path(root); self.records=read_jsonl(self.root/f'data/synthetic/v3/{split}.jsonl'); self.size=size

    def __len__(self):
        return len(self.records)

    def __getitem__(self,i):
        record=self.records[i]; image,affine=preprocess(self.root/record['image_path'],self.size)
        return image,targets(record,affine,self.size),affine,record


def detection_loss(output, target):
    losses={}
    for branch,hm,sz,off in [('box','center','size','offset'),('key','keypoint','key_extent','key_offset')]:
        prediction=output[hm].float()[0]; y=target[branch+'_heat'].to(prediction.device)
        valid=target['valid'].to(prediction.device)[None]
        p=prediction.sigmoid().clamp(1e-5,1-1e-5); positive=(y==1)&valid
        negative=(y<1)&valid
        focal=-(1-p).square()*p.log()*positive-(1-y).pow(4)*p.square()*(1-p).log()*negative
        losses[hm]=focal.sum()/positive.sum().clamp_min(1)
        mask=target[branch+'_mask'].to(prediction.device).repeat_interleave(2,0)
        for key,activation,suffix,weight in [(sz,F.softplus,'extent',.1),(off,torch.sigmoid,'offset',1.)]:
            pred=activation(output[key].float()[0]); truth=target[branch+'_'+suffix].to(pred.device)
            losses[key]=weight*(F.smooth_l1_loss(pred[mask],truth[mask],beta=1) if suffix=='extent' else F.l1_loss(pred[mask],truth[mask])) if mask.any() else pred.sum()*0
    # No segmentation target or loss exists for box-only annotations.
    return sum(losses.values()),losses


@torch.no_grad()
def decode(output, affine, config):
    device=output['center'].device; boxes=[]; scores=[]; labels=[]; centers=[]
    sx,sy,l,t,w,h=affine.to(device)
    for names, hm, sz, off in [(BOX_TYPES,'center','size','offset'),(KEY_TYPES,'keypoint','key_extent','key_offset')]:
        heat=output[hm][0].float().sigmoid(); peak=heat==F.max_pool2d(heat[None],3,1,1)[0]
        c,y,x=torch.where(peak&(heat>=config['decode_threshold']))
        for ci,yi,xi in zip(c,y,x):
            extent=4*F.softplus(output[sz][0,2*ci:2*ci+2,yi,xi].float())
            center=4*(torch.stack((xi,yi)).float()+output[off][0,2*ci:2*ci+2,yi,xi].float().sigmoid())
            if not (l<=center[0]<l+w*sx and t<=center[1]<t+h*sy):
                continue
            box=torch.cat((center-extent/2,center+extent/2))
            box[0::2].clamp_(l,l+w*sx); box[1::2].clamp_(t,t+h*sy)
            if (box[2:]<=box[:2]).any(): continue
            boxes.append(box); centers.append(center); scores.append(heat[ci,yi,xi]); labels.append(names[int(ci)])
    if not boxes:
        return {'boxes':torch.empty(0,4),'scores':torch.empty(0),'labels':torch.empty(0,dtype=torch.long),'centers':torch.empty(0,2)}
    boxes=torch.stack(boxes); scores=torch.stack(scores); labels=torch.tensor(labels,device=device)
    order=torch.argsort(scores,descending=True,stable=True)[:config['max_proposals']]
    keep=order[batched_nms(boxes[order],scores[order],labels[order],config['nms_iou'])]
    native=transform_boxes(boxes[keep],affine.to(device),inverse=True)
    native[:,0::2].clamp_(0,w); native[:,1::2].clamp_(0,h)
    ctr=(torch.stack(centers)[keep]-torch.stack((l,t)))/torch.stack((sx,sy))
    return {'boxes':native.cpu(),'scores':scores[keep].cpu(),'labels':labels[keep].cpu(),'centers':ctr.cpu()}
