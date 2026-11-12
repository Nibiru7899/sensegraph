"""CPU-only architecture audit: no downloads, optimizer, fitting, or CUDA access."""
from pathlib import Path
import sys
import json
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sensegraph.neural_spec import SenseGraphNet
from sensegraph.io import write_json,sha256


def main():
    torch.set_num_threads(1); torch.manual_seed(42)
    model=SenseGraphNet().eval()
    counts={name:sum(p.numel() for p in module.parameters()) for name,module in model.named_children()}
    stages={}
    for stage in ('warmup','perception','ocr','structural','reasoning','joint'):
        model.set_stage(stage); stages[stage]=sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    with torch.no_grad():
        pyramid,dense=model.encode_image(torch.zeros(1,3,640,640))
        ocr=model.ocr(torch.zeros(2,1,32,128))
    mlp=SenseGraphNet('mlp')
    result={'status':'random-weight CPU tensor verification only','torch':torch.__version__,
            'component_parameters':counts,'total_parameters':sum(p.numel() for p in model.parameters()),
            'stage_trainable_parameters_segmentation_disabled':stages,
            'mlp_graph_parameters':sum(p.numel() for p in mlp.graph.parameters()),
            'gat_graph_parameters':counts['graph'],
            'pyramid_shapes':{k:list(v.shape) for k,v in pyramid.items()},
            'dense_shapes':{k:list(v.shape) for k,v in dense.items()},'ocr_shape':list(ocr.shape),
            'source_sha256':sha256(ROOT/'src/sensegraph/neural_spec.py'),
            'downloads':0,'gpu_seconds':0,'training_executed':False,'vram_measured':False}
    write_json(ROOT/'experiments/neural_architecture/audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
