"""Selective official-source acquisition; conservative transfer accounting."""
from pathlib import Path
import io,json,zipfile,urllib.request,urllib.parse,time,re
from collections import Counter
import numpy as np
from PIL import Image
from .io import read_json,read_jsonl,write_json,write_jsonl,sha256
from .acquisition import Downloader,RemoteZipReader,archive_open,BudgetExceeded
from .real_mapping import CHARTQA_FULL_REVISION
from .domain_mapping import chartqa_partial
from .synthetic import fingerprint
from .domain_data import annotation_overlay


class DomainDownloader(Downloader):
    def __init__(self,root,cfg):
        super().__init__(root,cfg['download_ceiling']); self.cfg=cfg
        self.initial=read_json(Path(root)/cfg['experiment']/'initial.json')['download_initial']
    @property
    def remaining(self): return min(super().remaining,self.cfg['new_download_limit']-(self.ledger['transferred_bytes']-self.initial))
    def range(self,url,start,end):
        cache=self.root/self.cfg['real_output']/'range_cache'/f'{start}-{end}.bin'
        if cache.exists():
            value=cache.read_bytes()
            if len(value)!=end-start+1: raise IOError('Corrupt cached ZIP range')
            return value
        # ZIP headers/names/payloads otherwise require three TLS roundtrips each.
        # Small aligned blocks preserve selective acquisition and are fully charged.
        size=65536; chunks=[]
        for first in range(start//size*size,end//size*size+1,size):
            last=min(first+size-1,875370872-1)
            value=self._range_block(url,first,last)
            chunks.append(value[max(start-first,0):min(end-first+1,len(value))])
        return b''.join(chunks)
    def _range_block(self,url,start,end):
        cache=self.root/self.cfg['real_output']/'range_cache'/f'{start}-{end}.bin'
        if cache.exists():
            value=cache.read_bytes()
            if len(value)!=end-start+1: raise IOError('Corrupt cached ZIP range')
            return value
        size=end-start+1; self.charge(size) # Failed range attempts remain conservatively charged.
        separator='&' if '?' in url else '?'; fresh=url+separator+'domain_request='+str(time.time_ns())
        request=urllib.request.Request(fresh,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
        with archive_open(request,timeout=25) as response:
            if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'): raise IOError('Selective source did not honor range')
            value=response.read(size)
            if len(value)!=size: raise IOError('Truncated source range')
        cache.parent.mkdir(parents=True,exist_ok=True); temporary=cache.with_suffix('.tmp'); temporary.write_bytes(value); temporary.replace(cache)
        return value


def dhash(path):
    with Image.open(path) as image: values=np.asarray(image.convert('L').resize((9,8),Image.Resampling.BILINEAR))
    result=0
    for bit in (values[:,1:]>values[:,:-1]).flat: result=(result<<1)|int(bit)
    return result


def source_keys(annotation):
    info=annotation.get('general_figure_info',{}); title=info.get('title',{}).get('text','')
    title=' '.join(str(title).casefold().split())
    table=[{'x':m.get('x'),'y':m.get('y')} for m in annotation.get('models',[])]
    return {'title':title,'table_hash':fingerprint(table) if table and all(m['x'] is not None and m['y'] is not None for m in table) else None}


def partition_candidates(records,cfg):
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def join(i,j): parent[find(j)]=find(i)
    for i,a in enumerate(records):
        for j,b in enumerate(records[:i]):
            if a['image_sha256']==b['image_sha256'] or (a['source_keys']['table_hash'] and a['source_keys']['table_hash']==b['source_keys']['table_hash']) or (a['source_keys']['title'] and a['source_keys']['title']==b['source_keys']['title']) or (int(a['dhash'],16)^int(b['dhash'],16)).bit_count()<=4: join(i,j)
    groups={}
    for i,r in enumerate(records): groups.setdefault(find(i),[]).append(r)
    representatives=[]
    for group in groups.values():
        representative=min(group,key=lambda r:fingerprint([cfg['real_seed'],r['id']]))
        representative={**representative,'source_group':fingerprint(sorted(r['id'] for r in group)),
                        'group_members':[r['id'] for r in group],'near_prior':any(r['near_prior'] for r in group)}
        representatives.append(representative)
    result={s:[] for s in ['train','val','final']}; selected=[]
    # Final/validation use only groups without unresolved near-overlap with history.
    for kind in ['h_bar','v_bar','line']:
        pool=sorted([r for r in representatives if r['chart_type']==kind],key=lambda r:fingerprint([cfg['real_seed'],r['id']]))
        for split in ['final','val','train']:
            eligible=[r for r in pool if split=='train' or not r['near_prior']]; needed=cfg['real_counts_per_type'][split]
            if len(eligible)<needed: return None
            chosen=eligible[:needed]
            for r in chosen: result[split].append({**r,'split':split})
            ids={r['id'] for r in chosen}; pool=[r for r in pool if r['id'] not in ids]
    return result


def acquire_real(root,cfg,downloader):
    root=Path(root); output=root/cfg['real_output']; d=root/cfg['experiment']; manifest=output/'manifest.json'
    if manifest.exists(): return read_json(manifest)
    prior_paths=set()
    for base in [root/'data/raw/chartqa',root/'data/real_annotations/chartqa/train/png']:
        prior_paths.update(base.rglob('*.png'))
    for base in [root/'data/synthetic/v2',root/'data/synthetic/v3',root/'data/synthetic/perception_holdout_v1']:
        for file in base.glob('*.jsonl'):
            prior_paths.update(root/r['image_path'] for r in read_jsonl(file))
    prior_images={sha256(p):str(p.relative_to(root)) for p in prior_paths}; prior_hashes=[(dhash(p),str(p.relative_to(root))) for p in prior_paths]
    prior_ids={p.stem for p in (root/'data/raw/chartqa').rglob('*.png')}; prior_title=set(); prior_tables=set()
    for p in (root/'data/real_annotations/chartqa/train/annotations').glob('*.json'):
        key=source_keys(read_json(p)); prior_title.add(key['title']); prior_tables.add(key['table_hash'])
    url=f'https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/{CHARTQA_FULL_REVISION}/ChartQA%20Dataset.zip'
    reader=RemoteZipReader(downloader,url)
    if reader.size!=875370872: raise RuntimeError('Pinned ChartQA archive size changed')
    candidates=[]; excluded=[]
    progress=output/'candidate_progress_v2.json'
    if progress.exists():
        old=read_json(progress); candidates=old['candidates']; excluded=old['excluded']
    processed={r['id'] for r in candidates}|{r['id'] for r in excluded}
    with zipfile.ZipFile(reader) as archive:
        names=[n for n in archive.namelist() if n.startswith('ChartQA Dataset/train/annotations/') and n.endswith('.json')]
        names.sort(key=lambda n:fingerprint([cfg['real_seed'],Path(n).stem]))
        for name in names[:cfg['max_candidate_annotations']]:
            identity=Path(name).stem
            if identity in processed: continue
            if identity in prior_ids:
                excluded.append({'id':identity,'reason':'known prior identity'}); processed.add(identity); continue
            annotation_path=output/'raw/annotations'/f'{identity}.json'; annotation_path.parent.mkdir(parents=True,exist_ok=True)
            if not annotation_path.exists(): annotation_path.write_bytes(archive.read(name))
            annotation=read_json(annotation_path); kind=annotation.get('type'); keys=source_keys(annotation)
            try:
                if kind not in ['h_bar','v_bar','line']: raise ValueError('unsupported chart type')
                if (keys['table_hash'] and keys['table_hash'] in prior_tables) or (keys['title'] and keys['title'] in prior_title): raise ValueError('prior title/table identity')
                if sum(r['chart_type']==kind for r in candidates)>=100:
                    raise ValueError('Candidate family pool already contains100 images; preserve acquired pool and prioritize other families')
                image_name=f'ChartQA Dataset/train/png/{identity}.png'; image_path=output/'raw/png'/f'{identity}.png'
                if not image_path.exists(): image_path.parent.mkdir(parents=True,exist_ok=True); image_path.write_bytes(archive.read(image_name))
                digest=sha256(image_path)
                if digest in prior_images: raise ValueError('prior exact image identity')
                with Image.open(image_path) as image: record=chartqa_partial(annotation,image.size); width,height=image.size
                phash=dhash(image_path); near=[path for value,path in prior_hashes if (phash^value).bit_count()<=4]
                record.update({'schema_version':'domain-real-partial-2','id':identity,'source':'ChartQA official train','split':'candidate','annotation_coverage':'partial',
                    'width':width,'height':height,'image_path':image_path.relative_to(root).as_posix(),'image_sha256':digest,'source_annotation_sha256':sha256(annotation_path),
                    'source_keys':keys,'dhash':hex(phash),'near_prior':near,'source_revision':CHARTQA_FULL_REVISION,
                    'source_members':{'annotation':name,'image':image_name},'source_crc32':{'annotation':archive.getinfo(name).CRC,'image':archive.getinfo(image_name).CRC},
                    'class_coverage':{k:'partial' for k in ['bar','text','legend_item','line_segment']},'training_eligible':False})
                candidates.append(record)
            except (ValueError,KeyError) as error: excluded.append({'id':identity,'reason':str(error),'annotation_path':annotation_path.relative_to(root).as_posix()})
            processed.add(identity); write_json(progress,{'candidates':candidates,'excluded':excluded})
            partition=partition_candidates(candidates,cfg)
            if partition is not None:
                for split,rows in partition.items(): write_jsonl(output/f'{split}.jsonl',rows)
                all_rows=[r for rows in partition.values() for r in rows]
                for split,rows in partition.items():
                    for record in rows: annotation_overlay(root,record,d/'real_annotation_overlays'/split/f"{record['id']}.png")
                result={'complete':True,'images':len(all_rows),'split_counts':{s:dict(Counter(r['chart_type'] for r in rows)) for s,rows in partition.items()},
                    'revision':CHARTQA_FULL_REVISION,'source_split':'train','archive':reader.metadata,'candidate_count':len(candidates),'excluded':excluded,
                    'manifest_hashes':{s:sha256(output/f'{s}.jsonl') for s in partition},'prior_images_checked':len(prior_paths),'source_group_overlap':0,
                    'license':'GPL-3.0 author metadata; underlying third-party chart rights retained','coverage':'partial agent-audited source labels; no exhaustive real AP/precision claim',
                    'independence_limits':'Grouped by title, arrays, image identity and dHash<=4 on acquired candidates only. Publisher/pretraining overlap cannot be excluded.'}
                write_json(manifest,result); return result
            if len(processed)%25==0: print({'candidate_annotations':len(processed),'compatible_images':len(candidates),'download_bytes':downloader.ledger['transferred_bytes']},flush=True)
    write_json(output/'acquisition_blocker.json',{'reason':'Insufficient independent compatible source groups within candidate cap','candidates':len(candidates),'examined':len(processed),'excluded':excluded})
    raise RuntimeError('Real source inventory incomplete; see acquisition_blocker.json')


def acquire_ocr(root,cfg,downloader):
    output=Path(root)/'data/models/domain_trocr'; manifest=output/'source_manifest.json'
    if manifest.exists(): return read_json(manifest)
    revision=cfg['ocr_revision']; prefix=f"https://huggingface.co/{cfg['ocr_model']}/resolve/{revision}/"
    assets=['README.md','config.json','generation_config.json','preprocessor_config.json','sentencepiece.bpe.model','special_tokens_map.json','tokenizer_config.json']
    for name in assets: downloader.fetch(prefix+name,output.relative_to(root)/name)
    upstream='https://raw.githubusercontent.com/microsoft/unilm/master/'
    downloader.fetch(upstream+'trocr/README.md',output.relative_to(root)/'upstream_README.md')
    downloader.fetch(upstream+'LICENSE',output.relative_to(root)/'upstream_LICENSE')
    text=(output/'upstream_README.md').read_text(encoding='utf-8'); license_text=(output/'upstream_LICENSE').read_text(encoding='utf-8')
    # Project-level official declaration is retained separately from missing HF metadata.
    permitted='This project is licensed under' in text and 'trocr-small-printed.pt' in text and 'MIT License' in license_text
    result={'model':cfg['ocr_model'],'revision':revision,'HF_model_card_license':'not separately declared','upstream_project_license':'MIT' if 'MIT License' in license_text else 'unverified',
        'permission_basis':'Official TrOCR project README lists the printed model and applies its root project license; separate HF weight-specific declaration absent.',
        'permission_resolved':permitted,'redistribution':'No weights/data redistributed by this study; upstream notices retained'}
    if permitted:
        downloader.fetch(prefix+'model.safetensors',output.relative_to(root)/'model.safetensors',cfg['ocr_safetensors_sha256'],cfg['ocr_safetensors_bytes'])
        result['weights_sha256']=sha256(output/'model.safetensors')
    else: result['blocker']='Applicable official permission not resolved; no weights acquired and no substitute'
    result['files']={p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in output.iterdir() if p.is_file()}; write_json(manifest,result); return result


def acquire(root,cfg):
    downloader=DomainDownloader(root,cfg); result={}
    # Independent branches preserve exact blockers without erasing completed transfers.
    for name,fn in [('real',acquire_real),('ocr',acquire_ocr)]:
        try: result[name]=fn(root,cfg,downloader)
        except Exception as error:
            result[name]={'complete':False,'blocker':f'{type(error).__name__}: {error}'}
            write_json(Path(root)/cfg['experiment']/f'{name}_acquisition_failure.json',result[name])
    write_json(Path(root)/cfg['experiment']/'acquisition.json',result)
    return {'real_complete':result['real'].get('complete',False),'ocr_permission_resolved':result['ocr'].get('permission_resolved',False),'details':result}
