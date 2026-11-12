"""CPU-only audit of frozen backbone, unused heads and restored trained inference."""
from pathlib import Path
import sys,random,argparse
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,write_json,sha256
from sensegraph.neural_spec import SenseGraphNet
from sensegraph.visual_training import preprocess
from sensegraph.full_visual_training import decode_maps


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',choices=['v2','v3'],default='v2'); args=parser.parse_args()
    torch.set_num_threads(1); d=ROOT/f'experiments/visual_training/{args.run}'; cfg=read_json(ROOT/('configs/corrected_visual_training.json' if args.run=='v3' else 'configs/full_visual_training.json'))
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed']); np.random.seed(cfg['seed'])
    fresh=SenseGraphNet().eval(); reference=fresh.state_dict()
    selected=torch.load(d/'best.pt',weights_only=True,map_location='cpu'); weights=selected['model']
    unused=[k for k in weights if k.startswith(('graph.','relations.','questions.','reasoner.','ocr.','node_fuse.','heads.segmentation.'))]
    unchanged_unused=all(torch.equal(weights[k],reference[k]) for k in unused)
    pretrained=torch.load(ROOT/cfg['weights_path'],weights_only=True,map_location='cpu')
    early=[k for k in pretrained if k.startswith(('conv1.','bn1.','layer1.','layer2.'))]
    unchanged_early=all(torch.equal(weights['backbone.'+k],pretrained[k]) for k in early)
    bn=[k for k in pretrained if 'running_' in k or 'num_batches_tracked' in k]
    unchanged_bn=all(torch.equal(weights['backbone.'+k],pretrained[k]) for k in bn)
    assert unchanged_unused and unchanged_early and unchanged_bn
    fresh.load_state_dict(weights,strict=True); fresh.eval(); image,affine=preprocess(ROOT/'data/synthetic/v3/images/synthetic-01400.png')
    with torch.inference_mode(): first=fresh.encode_image(image[None])[1]; a=decode_maps(first,affine,cfg)
    fresh.load_state_dict(torch.load(d/'best.pt',weights_only=True,map_location='cpu')['model'])
    with torch.inference_mode(): second=fresh.encode_image(image[None])[1]; b=decode_maps(second,affine,cfg)
    exact=all(torch.equal(first[k],second[k]) for k in first) and all(torch.equal(a[k],b[k]) for k in a)
    assert exact and torch.isfinite(a['boxes']).all() and (a['boxes'][:,2:]>a['boxes'][:,:2]).all()
    write_json(d/'checkpoint_audit.json',{'selected_stage':selected['stage'],'selected_epoch':selected['epoch'],'sha256':sha256(d/'best.pt'),
                'unused_graph_qa_ocr_and_segmentation_tensors_checked':len(unused),'unused_unchanged':unchanged_unused,
                'early_backbone_tensors_checked':len(early),'early_backbone_unchanged':unchanged_early,
                'bn_stat_tensors_checked':len(bn),'all_bn_statistics_unchanged':unchanged_bn,'reloaded_cpu_inference_exact':exact,
                'positive_finite_native_boxes':True,'image_only':True,'gpu_execution':False})
    print('Checkpoint, frozen tensors and CPU reload audit passed.')


if __name__=='__main__': main()
