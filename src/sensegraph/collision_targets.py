"""Order-invariant Gaussian union at collisions; never mutates frozen v1/v2 targets."""
import torch
from .visual_training import VisualDataset,preprocess,targets,transform_boxes,BOX_TYPES,KEY_TYPES
from .perception import TYPE_CLASS


def invariant_targets(record,affine,size=640):
    result=targets(record,affine,size)
    if not result['collisions']: return result
    for node in record['nodes']:
        label=TYPE_CLASS.get(node['type'])
        if label is None: continue
        branch='box' if label in BOX_TYPES else 'key'; types=BOX_TYPES if branch=='box' else KEY_TYPES
        box=torch.tensor(node['bbox'],dtype=torch.float32)
        if not torch.isfinite(box).all() or (box[:2]<0).any() or (box[2:] > torch.tensor([record['width'],record['height']])).any() or (box[2:]<=box[:2]).any(): continue
        resized=transform_boxes(box,affine)/4; x,y=((resized[:2]+resized[2:])/2).floor().long().tolist(); c=types.index(label)
        if result[branch+'_heat'][c,y,x]==1 and not result[branch+'_mask'][c,y,x]:
            individual=targets({**record,'nodes':[node]},affine,size)
            result[branch+'_heat']=torch.maximum(result[branch+'_heat'],individual[branch+'_heat'])
    return result


class CollisionInvariantDataset(VisualDataset):
    def __getitem__(self,index):
        record=self.records[index]; image,affine=preprocess(self.root/record['image_path'],self.size)
        return image,invariant_targets(record,affine,self.size),affine,record
