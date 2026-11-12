"""Post-selection GPU repeatability/precision audit using TRAIN and VAL only."""
from pathlib import Path
import sys,time
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,write_json,sha256
from sensegraph.neural_spec import SenseGraphNet
from sensegraph.visual_training import VisualDataset
from sensegraph.collision_targets import CollisionInvariantDataset
from sensegraph.full_visual_training import evaluate,decode_maps
from sensegraph.relation_training import seed_everything
from sensegraph.quality_budget import QualitySession


def flags():
    return {'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_benchmark':torch.backends.cudnn.benchmark}


def main():
    torch.set_num_threads(1); base=ROOT/'experiments/visual_training'; d=base/'postselection_verification'; checkpoint=base/'release/selected.pt'
    if (d/'attempt.json').exists(): raise RuntimeError('Post-selection verification already attempted; preserve it')
    assert read_json(base/'real_partial/results.json')['complete']
    selected=read_json(base/'independent_holdout/selection.json'); saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
    assert sha256(checkpoint)==selected['candidates'][selected['selected']]['sha256']
    write_json(d/'attempt.json',{'time_unix':time.time(),'checkpoint_sha256':sha256(checkpoint),'script_sha256':sha256(Path(__file__)),'no_holdout_inference_or_model_selection':True})
    results={}; val=CollisionInvariantDataset(ROOT,'val',640); image,_,affine,_=VisualDataset(ROOT,'train',640)[0]
    with QualitySession(d/'gpu_budget.json',base/'real_partial/gpu_budget.json',240,min_seconds=180) as session:
        model=SenseGraphNet().cuda().eval(); model.load_state_dict(saved['model'],strict=True); pixels=image[None].cuda()
        for mode in ['audit_defaults','training_fp32']:
            if mode=='training_fp32': seed_everything(saved['config']['seed'])
            torch.cuda.reset_peak_memory_stats(); started=time.monotonic()
            with torch.inference_mode(): a=model.encode_image(pixels)[1]; first=decode_maps(a,affine,saved['config']); b=model.encode_image(pixels)[1]; second=decode_maps(b,affine,saved['config'])
            repeated=all(torch.equal(a[k],b[k]) for k in a) and all(torch.equal(first[k],second[k]) for k in first)
            model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda')['model'])
            with torch.inference_mode(): c=model.encode_image(pixels)[1]; restored=decode_maps(c,affine,saved['config'])
            reloaded=all(torch.equal(a[k],c[k]) for k in a) and all(torch.equal(first[k],restored[k]) for k in first)
            del a,b,c
            metrics=evaluate(model,val,saved['config'],d,mode)
            results[mode]={'flags':flags(),'training_image_repeat_exact':repeated,'training_image_reload_exact':reloaded,'validation':metrics,'seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            print({'mode':mode,'validation_ap50':metrics['ap50']['macro_ap'],'GPU_repeat_exact':repeated,'GPU_reload_exact':reloaded},flush=True)
    frozen=selected['candidates'][selected['selected']]['validation_ap50']; strict=results['training_fp32']['validation']['ap50']['macro_ap']
    write_json(d/'results.json',{'complete':True,'checks':results,'original_validation_ap50':frozen,'strict_validation_ap50_exactly_reproduced':strict==frozen,
        'limits':'One TRAIN image repeated/reloaded; two complete VAL evaluations, no new test inference, tuning or reselection. Exact on this case/host does not prove cross-device determinism. Audit permits TF32 convolutions by default; individual kernels not traced. First independent-holdout results remain unchanged.'})


if __name__=='__main__': main()
