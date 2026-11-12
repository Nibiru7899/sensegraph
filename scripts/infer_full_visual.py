"""Image-only inference from a local trained full-perception checkpoint; CPU default."""
import argparse
from pathlib import Path
import sys
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.neural_spec import SenseGraphNet
from sensegraph.visual_training import preprocess
from sensegraph.full_visual_training import decode_maps
from sensegraph.visual_experiment import overlay
from sensegraph.io import write_json,sha256
from sensegraph.neural_spec import CropCTCRecognizer
from sensegraph.ocr_inference import recognize_regions


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--image',required=True); parser.add_argument('--checkpoint',default='experiments/visual_training/release/selected.pt'); parser.add_argument('--output',required=True); parser.add_argument('--device',default='cpu',choices=['cpu']); parser.add_argument('--with-ocr',action='store_true'); args=parser.parse_args()
    torch.set_num_threads(1)
    path=ROOT/args.checkpoint; saved=torch.load(path,map_location='cpu',weights_only=True)
    model=SenseGraphNet().to(args.device).eval(); model.load_state_dict(saved['model'],strict=True)
    image,affine=preprocess(ROOT/args.image,640)
    with torch.inference_mode(): _,maps=model.encode_image(image[None].to(args.device)); pred=decode_maps(maps,affine,saved['config'])
    output=ROOT/args.output
    result={'image':args.image,'checkpoint_sha256':sha256(path),'stage':saved['stage'],'epoch':saved['epoch'],'predictions':{k:v.tolist() for k,v in pred.items()},'oracle_annotations_read':False}
    if args.with_ocr:
        ocr_path=ROOT/'experiments/visual_training/release/ocr.pt'
        recognizer=CropCTCRecognizer().eval(); recognizer.load_state_dict(torch.load(ocr_path,map_location='cpu',weights_only=True)['model'],strict=True)
        indices=torch.where(((pred['labels']==4)|(pred['labels']==5))&(pred['scores']>=.25))[0]
        with Image.open(ROOT/args.image) as raw:
            words=recognize_regions(recognizer,raw.convert('RGB'),pred['boxes'][indices].tolist(),device='cpu')
        result['ocr']={'checkpoint_sha256':sha256(ocr_path),'scope':'ASCII synthetic chart crop recognizer; uncalibrated strings, no verified numerical extraction',
                       'regions':[{'proposal_index':int(i),'text':word} for i,word in zip(indices,words)]}
    write_json(output,result)
    overlay(ROOT,{'image_path':args.image},pred,output.with_suffix('.png'))
    print(output)


if __name__=='__main__': main()
