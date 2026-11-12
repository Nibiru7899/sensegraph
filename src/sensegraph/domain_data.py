"""New perception-only renderer, independent validators and manifest readers."""
from pathlib import Path
from collections import Counter
import math
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.markers import MarkerStyle
from matplotlib.font_manager import findfont
import torch
from .io import read_json,read_jsonl,write_json,write_jsonl,sha256
from .synthetic import fingerprint
from .visual_training import preprocess,TYPE_CLASS
from .collision_targets import invariant_targets

FAMILIES=('vertical_bar','horizontal_bar','vertical_grouped','horizontal_grouped','line')
FONTS={'train':['Georgia','Tahoma','Segoe UI','Trebuchet MS','Verdana'],
       'val':['Cambria','Consolas','Palatino Linotype']}
WORDS={'train':['Harbor','Valley','Meadow','Riverside','Pinecrest','Westfield','Ashford','Oakridge','Seabrook','Hillview','Lakeside','Brookhaven'],
       'val':['Copper','Silver','Amber','Cobalt','Violet','Pearl','Sienna','Jade','Ivory','Crimson','Indigo','Cedar']}
MEASURES={'train':['Water use','Energy demand','Operating cost','Net exports','Employment','Sales revenue','Cargo volume','Housing supply','Visitor arrivals','Crop yield','Fuel demand','Service hours','Output growth','Unit sales'],
          'val':['Research spending','Passenger traffic','Waste recovery','Forest coverage','Loan balances','Library visits','Freight capacity','Retail turnover','Solar production','Rainfall totals','Maintenance costs','Rental demand','Battery capacity','Market share']}


def make_domain_spec(cfg,index,split):
    seed=int(fingerprint([cfg['synthetic_seed'],split,index])[:16],16); rng=np.random.default_rng(seed)
    family=FAMILIES[index%5]; horizontal=family.startswith('horizontal'); grouped='grouped' in family
    ns=int(rng.integers(2,4)) if grouped or family=='line' else 1; n=int(rng.integers(3,13))
    low,high=[(-100,100),(0,1),(-1000,-10),(10000,50000),(0,100)][int(rng.integers(5))]
    values=np.round(rng.uniform(low,high,(ns,n)),4)
    if index%10==0: values[0,0]=0.
    font=FONTS[split][(index//5)%len(FONTS[split])]
    measure=MEASURES[split][index%14]; place=WORDS[split][(index//14)%12]; year=2000+index//168
    title=(f'{measure} in {place}, {year}' if split=='train' else f'{place}: {measure.lower()} during {year}')
    categories=[WORDS[split][i] for i in range(n)]
    if index%4==0: categories=[f'{v} district' for v in categories]
    series=[('Observed','Forecast','Benchmark')[i] if split=='train' else ('Current','Planned','Historical')[i] for i in range(ns)]
    palette=[matplotlib.colors.to_hex(matplotlib.colormaps['tab20'](int(rng.integers(20)))) for _ in range(ns)]
    layout=[float(rng.uniform(.16,.25)),float(rng.uniform(.21,.29)),float(rng.uniform(.65,.72)),float(rng.uniform(.52,.61))]
    if layout[0]+layout[2]>.96: layout[2]=.96-layout[0]
    if layout[1]+layout[3]>.87: layout[3]=.87-layout[1]
    style={'font':font,'font_size':int(rng.integers(6,12)),'title_size':int(rng.integers(10,17)),
           'background':['#ffffff','#faf8f1','#f0f4f7','#f7f5f5'][int(rng.integers(4))],
           'grid':bool(rng.integers(2)),'grid_style':['-',':','--'][int(rng.integers(3))],
           'layout':layout,'rotation':int(rng.choice([0,15,30,45,90])),'palette':palette,
           'linewidth':float(rng.choice([.4,.8,1.6,2.8])),'marker':str(rng.choice(['none','o','s','D','x'])),
           'markersize':float(rng.choice([2.5,4.,6.])), 'legend':str(rng.choice(['bottom','top','inside','right','direct'])),
           'legend_frame':bool(rng.integers(2)), 'legend_columns':int(rng.integers(1,ns+1)),
           'per_bar_colors':bool(rng.integers(2)), 'tick_format':'percent' if index%7==0 else 'number'}
    if np.abs(values).max()>100: style['tick_format']='number'
    data={'categories':categories,'series':series,'values':values.tolist()}
    return {'id':f'domain-{split}-{index:05d}','split':split,'family':family,'horizontal':horizontal,
            'chart_type':'line' if family=='line' else 'grouped_bar' if grouped else 'bar',
            'seed':seed,'style':style,'style_hash':fingerprint(style),'data':data,'data_hash':fingerprint(data),
            'numeric_hash':fingerprint(data['values']),'title':title,'numeric_label':measure+' (%)' if style['tick_format']=='percent' else measure+' (units)',
            'categorical_label':'Reporting district' if split=='train' else 'Regional category',
            'resolution':[[480,360],[640,480],[850,600],[1000,750]][int(rng.integers(4))],'dpi':int(rng.choice([80,100,120]))}


def render_domain(spec,destination,root):
    w,h=spec['resolution']; dpi=spec['dpi']; st=spec['style']; vals=np.asarray(spec['data']['values']); ns,n=vals.shape
    with matplotlib.rc_context({'font.family':st['font'],'font.size':st['font_size'],'axes.unicode_minus':False}):
        fig=Figure(figsize=(w/dpi,h/dpi),dpi=dpi,facecolor=st['background']); canvas=FigureCanvasAgg(fig)
        ax=fig.add_axes(st['layout'],facecolor=st['background']); ax.spines[['top','right']].set_visible(False)
        lo=min(0.,float(vals.min())); hi=max(0.,float(vals.max())); span=max(hi-lo,.01); limits=(lo-.15*span,hi+.20*span)
        ticks=np.linspace(*limits,5)
        def label(v):
            if st['tick_format']=='percent': return f'{v:.2f}%'
            return f'{v:.3f}' if max(abs(limits[0]),abs(limits[1]))<2 else f'{v:.1f}' if max(abs(limits[0]),abs(limits[1]))<10000 else f'{v:.2e}'
        if spec['horizontal']:
            ax.set_xlim(*limits); ax.set_xticks(ticks,[label(v) for v in ticks]); ax.set_ylim(-.65,n-.35)
            ax.set_yticks(range(n),spec['data']['categories'],rotation=st['rotation']); ax.set_xlabel(spec['numeric_label']); ax.set_ylabel(spec['categorical_label'])
        else:
            ax.set_ylim(*limits); ax.set_yticks(ticks,[label(v) for v in ticks]); ax.set_xlim(-.65,n-.35)
            ax.set_xticks(range(n),spec['data']['categories'],rotation=st['rotation']); ax.set_xlabel(spec['categorical_label']); ax.set_ylabel(spec['numeric_label'])
        ax.set_title(spec['title'],fontsize=st['title_size'])
        if st['grid']: ax.grid(True,axis='x' if spec['horizontal'] else 'y',alpha=.25,linestyle=st['grid_style'])
        ax.set_axisbelow(True); artists=[]; known_text={}; rng=np.random.default_rng(spec['seed'])
        for s in range(ns):
            color=st['palette'][s]; xs=np.arange(n)+(s-(ns-1)/2)*.72/ns
            if spec['chart_type']!='line':
                colors=[matplotlib.colors.to_hex(matplotlib.colormaps['tab20'](int(rng.integers(20)))) for _ in range(n)] if st['per_bar_colors'] else color
                bars=ax.barh(xs,vals[s],height=.72/ns*.94,color=colors,edgecolor=colors,linewidth=.8) if spec['horizontal'] else ax.bar(xs,vals[s],width=.72/ns*.94,color=colors,edgecolor=colors,linewidth=.8)
                for j,artist in enumerate(bars):
                    xy=[float(vals[s,j]),float(xs[j])] if spec['horizontal'] else [float(xs[j]),float(vals[s,j])]
                    artists.append((f'mark:{s}:{j}','bar',artist,.4*dpi/72,{'series':s,'category':j,'data_xy':xy,'numeric_value':float(vals[s,j])}))
            else:
                for j in range(n-1):
                    artist,=ax.plot([j,j+1],vals[s,j:j+2],color=color,linewidth=st['linewidth'])
                    artists.append((f'segment:{s}:{j}','line_segment',artist,st['linewidth']*.5*dpi/72,{'series':s,'category':j,'data_endpoints':[[j,float(vals[s,j])],[j+1,float(vals[s,j+1])]]}))
                if st['marker']!='none':
                    for j in range(n):
                        artist,=ax.plot(j,vals[s,j],marker=st['marker'],markersize=st['markersize'],markeredgewidth=.6,color=color,linestyle='none')
                        artists.append((f'mark:{s}:{j}','point',artist,.3*dpi/72,{'series':s,'category':j,'data_xy':[j,float(vals[s,j])],'numeric_value':float(vals[s,j])}))
            if st['legend']=='direct' and spec['chart_type']=='line':
                text=ax.text(n-1,float(vals[s,-1]),spec['data']['series'][s],color=color,fontsize=max(6,st['font_size']-1)); known_text[id(text)]=('legend_item',f'legend:{s}')
        if not(st['legend']=='direct' and spec['chart_type']=='line'):
            handles=[Line2D([],[],color=st['palette'][s],lw=3,label=name) for s,name in enumerate(spec['data']['series'])]
            kwargs={'handles':handles,'ncol':st['legend_columns'],'fontsize':max(6,st['font_size']-1),'frameon':st['legend_frame']}
            if st['legend']=='inside': legend=ax.legend(loc='upper right',**kwargs)
            elif st['legend']=='top': legend=fig.legend(loc='upper center',bbox_to_anchor=(.55,.99),**kwargs)
            elif st['legend']=='right': legend=fig.legend(loc='center right',bbox_to_anchor=(.99,.5),**kwargs)
            else: legend=fig.legend(loc='lower center',bbox_to_anchor=(.55,.01),**kwargs)
            for s,text in enumerate(legend.get_texts()): known_text[id(text)]=('legend_item',f'legend:{s}')
        fig.text(.02,.04,'Source: regional reporting office' if spec['split']=='train' else 'Notes: quarterly administrative records',fontsize=6)
        canvas.draw(); actual=canvas.get_width_height()
        if actual!=(w,h): raise ValueError(f'Unexpected canvas {actual} != {(w,h)}')
        renderer=canvas.get_renderer(); nodes=[]; unavailable=[]
        def add(identity,kind,artist,pad=0,attrs=None):
            extent=artist.get_window_extent(renderer); raw=[float(extent.x0-pad),float(h-extent.y1-pad),float(extent.x1+pad),float(h-extent.y0+pad)]
            if kind=='point':
                marker=MarkerStyle(st['marker']); extent=marker.get_path().transformed(marker.get_transform()).get_extents()
                x,y=attrs['anchor_px']; factor=st['markersize']*dpi/72
                raw=[float(x+extent.x0*factor-pad),float(y-extent.y1*factor-pad),float(x+extent.x1*factor+pad),float(y-extent.y0*factor+pad)]
            box=[max(0.,raw[0]),max(0.,raw[1]),min(float(w),raw[2]),min(float(h),raw[3])]
            if not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
                unavailable.append({'id':identity,'type':kind,'reason':'no positive visible renderer extent','renderer_bbox':raw}); return
            nodes.append({'id':identity,'type':kind,'bbox':box,'renderer_bbox':raw,'stroke_padding_pixels':pad,**(attrs or {})})
        for axis,spine in [('x','bottom'),('y','left')]:
            add(f'axis:{axis}','axis',ax.spines[spine],.5*ax.spines[spine].get_linewidth()*dpi/72,{'axis':axis})
            source=ax.xaxis if axis=='x' else ax.yaxis
            known_text[id(source.label)]=('axis_label',f'axis_label:{axis}')
            for j,tick in enumerate(source.get_major_ticks()):
                if tick.tick1line.get_visible(): add(f'tick:{axis}:{j}','tick',tick.tick1line,.5*tick.tick1line.get_markeredgewidth()*dpi/72,{'axis':axis,'axis_value':float(tick.get_loc())})
                known_text[id(tick.label1)]=('tick_label',f'tick_label:{axis}:{j}')
        known_text[id(ax.title)]=('title','title')
        seen=set()
        for text in fig.findobj(match=Text):
            if id(text) in seen or not text.get_visible() or not text.get_text(): continue
            seen.add(id(text)); kind,identity=known_text.get(id(text),('text',f'text:{len(seen)}'))
            add(identity,kind,text,attrs={'text':text.get_text(),'rotation':float(text.get_rotation()),'font_size_pt':float(text.get_fontsize())})
        matrix=np.array([[1,0,0],[0,-1,h],[0,0,1]])@ax.transData.get_affine().get_matrix()
        for identity,kind,artist,pad,attrs in artists:
            if 'data_xy' in attrs: attrs['anchor_px']=(matrix@[*attrs['data_xy'],1])[:2].tolist()
            add(identity,kind,artist,pad,attrs)
        texts=[v for v in nodes if 'text' in v]
        for node in texts:
            a=node['bbox']; overlap=any(other is not node and min(a[2],other['bbox'][2])>max(a[0],other['bbox'][0]) and min(a[3],other['bbox'][3])>max(a[1],other['bbox'][1]) for other in texts)
            node['recognition_supervision']='ambiguous_overlap_or_clip' if overlap or node['bbox']!=node['renderer_bbox'] else 'trusted'
        destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()).convert('RGB').save(destination)
        record={'schema_version':'domain-perception-1','source':'synthetic','annotation_coverage':'complete_supported_classes','id':spec['id'],'split':spec['split'],
                'chart_type':spec['chart_type'],'family':spec['family'],'orientation':'horizontal' if spec['horizontal'] else 'vertical',
                'width':w,'height':h,'spec':spec,'image_path':destination.relative_to(root).as_posix(),'image_sha256':sha256(destination),
                'font':{'family':st['font'],'sha256':sha256(findfont(st['font'],fallback_to_default=False))},
                'nodes':nodes,'unavailable_artists':unavailable,'axes':{'data_to_pixel':matrix.tolist(),'numeric_axis':'x' if spec['horizontal'] else 'y',
                'categorical_axis':'y' if spec['horizontal'] else 'x','numeric_limits':list(limits),'numeric_ticks':ticks.tolist(),
                'plot_bbox':[float(ax.bbox.x0),float(h-ax.bbox.y1),float(ax.bbox.x1),float(h-ax.bbox.y0)]},
                'questions':None,'evidence':None,'segmentation_masks':None,'notes':'Perception supervision only; no inferred QA or stroke masks'}
        fig.clear(); return record


def validate_domain(record,root=None):
    errors=[]; w,h=record['width'],record['height']; ids=[n['id'] for n in record['nodes']]
    if len(ids)!=len(set(ids)): errors.append('duplicate node ids')
    matrix=np.asarray(record['axes']['data_to_pixel']); inverse=np.linalg.inv(matrix)
    for node in record['nodes']:
        b=node['bbox']
        if node['type'] not in TYPE_CLASS or len(b)!=4 or not np.isfinite(b).all() or not(0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h): errors.append('invalid physical bbox/type')
        if 'anchor_px' in node:
            expected=(matrix@[*node['data_xy'],1])[:2]
            if not np.allclose(expected,node['anchor_px'],atol=1e-6) or not np.allclose((inverse@[*node['anchor_px'],1])[:2],node['data_xy'],atol=1e-6): errors.append('transform mismatch')
            value_axis=0 if record['orientation']=='horizontal' else 1
            if node['numeric_value']!=node['data_xy'][value_axis]: errors.append('numeric axis mismatch')
            if node['type']=='bar':
                zero=list(node['data_xy']); zero[value_axis]=0; base=(matrix@[*zero,1])[:2]
                if not(b[0]-1e-5<=base[0]<=b[2]+1e-5 and b[1]-1e-5<=base[1]<=b[3]+1e-5): errors.append('bar baseline mismatch')
                count=len(record['spec']['data']['series']); half=.72/count*.94/2; xy=node['data_xy']; cat=1-value_axis
                first=list(xy); second=list(xy); first[value_axis]=0; first[cat]-=half; second[cat]+=half
                corners=np.asarray([(matrix@[*v,1])[:2] for v in [first,second]]); pad=node['stroke_padding_pixels']
                expected=[max(0.,corners[:,0].min()-pad),max(0.,corners[:,1].min()-pad),min(w,corners[:,0].max()+pad),min(h,corners[:,1].max()+pad)]
                if not np.allclose(b,expected,atol=1e-5): errors.append('bar renderer geometry mismatch')
        raw=node['renderer_bbox']; clipped=[max(0.,raw[0]),max(0.,raw[1]),min(float(w),raw[2]),min(float(h),raw[3])]
        if not np.allclose(b,clipped,atol=1e-8): errors.append('renderer extent clipping mismatch')
        if node['type']=='line_segment':
            ends=np.asarray([(matrix@[*v,1])[:2] for v in node['data_endpoints']]); pad=node['stroke_padding_pixels']
            expected=[max(0.,ends[:,0].min()-pad),max(0.,ends[:,1].min()-pad),min(w,ends[:,0].max()+pad),min(h,ends[:,1].max()+pad)]
            if not np.allclose(b,expected,atol=1e-5): errors.append('segment renderer geometry mismatch')
    if record['spec']['chart_type']=='line' and record['spec']['style']['marker']=='none' and any(n['type']=='point' for n in record['nodes']): errors.append('unpainted point supervised')
    if record['questions'] is not None or record['segmentation_masks'] is not None: errors.append('unexpected fabricated supervision')
    if root is not None:
        with Image.open(Path(root)/record['image_path']) as image:
            if image.size!=(w,h): errors.append('resolution mismatch')
        if sha256(Path(root)/record['image_path'])!=record['image_sha256']: errors.append('image hash mismatch')
    return sorted(set(errors))


def annotation_overlay(root,record,path):
    with Image.open(Path(root)/record['image_path']) as raw: image=raw.convert('RGB')
    draw=ImageDraw.Draw(image); colors={'bar':'red','point':'red','line_segment':'orange','axis':'green','tick':'cyan'}
    for node in record['nodes']:
        if node.get('bbox') is not None: draw.rectangle(node['bbox'],outline=colors.get(node['type'],'blue'),width=1)
    Path(path).parent.mkdir(parents=True,exist_ok=True); image.save(path)


def prepare_synthetic(root,cfg,limit=None):
    root=Path(root); output=root/cfg['synthetic_output']; records=[]; total=cfg['train_count']+cfg['val_count']; count=min(limit or total,total)
    if limit and limit<total and (output/'manifest.json').exists() and read_json(output/'manifest.json')['complete']:
        raise ValueError('Full inventory exists; bounded preflight requires a fresh namespace, not truncated manifests')
    for global_index in range(count):
        split='train' if global_index<cfg['train_count'] else 'val'; index=global_index if split=='train' else global_index-cfg['train_count']
        spec=make_domain_spec(cfg,index,split); target=output/'annotations'/f"{spec['id']}.json"
        if target.exists():
            record=read_json(target)
            if record['spec']!=spec: raise ValueError('Existing generated spec changed')
        else:
            record=render_domain(spec,output/'images'/f"{spec['id']}.png",root); write_json(target,record)
        errors=validate_domain(record,root)
        if errors: raise ValueError(f"{record['id']}: {errors}")
        records.append(record)
        if global_index<15: annotation_overlay(root,record,root/cfg['experiment']/'synthetic_overlays'/output.name/f"{record['id']}.png")
    for split in ['train','val']: write_jsonl(output/f'{split}.jsonl',[r for r in records if r['split']==split])
    report={'records':len(records),'complete':count==total,'invalid':0,'families':dict(Counter(r['family'] for r in records)),
            'font_family_cells':dict(Counter(f"{r['split']}:{r['family']}:{r['font']['family']}" for r in records)),
            'unique_titles':len({r['spec']['title'] for r in records}),'ambiguous_text':sum(n.get('recognition_supervision')=='ambiguous_overlap_or_clip' for r in records for n in r['nodes']),
            'config_sha256':fingerprint(cfg),'generation_source_sha256':sha256(Path(__file__))}
    write_json(output/'manifest.json',report); return report


class DomainDataset:
    def __init__(self,root,manifest,split):
        self.root=Path(root); self.size=640; self.records=read_jsonl(self.root/manifest)
        if split not in ('train','val') or any(r['split']!=split or r.get('annotation_coverage')!='complete_supported_classes' for r in self.records): raise ValueError('Only complete synthetic train/val records may enter fitting')
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        record=self.records[i]; image,affine=preprocess(self.root/record['image_path'],640)
        return image,invariant_targets(record,affine,640),affine,record


def validate_inventory(root,cfg):
    root=Path(root); records=read_jsonl(root/cfg['synthetic_output']/'train.jsonl')+read_jsonl(root/cfg['synthetic_output']/'val.jsonl')
    if len(records)!=1700: raise ValueError('Incomplete synthetic inventory')
    per_split={}; failures=[]
    for split in ['train','val']:
        counts=Counter(); collisions=Counter()
        for r in [v for v in records if v['split']==split]:
            failures.extend((r['id'],e) for e in validate_domain(r,root)); _,a=preprocess(root/r['image_path'],640); target=invariant_targets(r,a)
            counts.update(target['counts']); collisions.update(target['collisions'])
        fractions={k:collisions[k]/v for k,v in counts.items()}; per_split[split]={'counts':dict(counts),'collisions':dict(collisions),'collision_fractions':fractions}
        if any(v>cfg['collision_limit'] for v in fractions.values()): failures.append((split,'collision gate exceeded'))
    for key in ['title','numeric_hash','style_hash']:
        a={r['spec'][key] for r in records if r['split']=='train'}; b={r['spec'][key] for r in records if r['split']=='val'}
        if a&b: failures.append((key,'cross-split identity overlap'))
    prior_numbers=set(); prior_images=set()
    for version in ['v2','v3','perception_holdout_v1']:
        for p in (root/'data/synthetic'/version).glob('*.jsonl'):
            for r in read_jsonl(p): prior_numbers.add(fingerprint(r['spec']['data']['values'])); prior_images.add(r['image_sha256'])
    if any(r['spec']['numeric_hash'] in prior_numbers or r['image_sha256'] in prior_images for r in records): failures.append(('history','prior numeric/image identity overlap'))
    if len({r['image_sha256'] for r in records})!=len(records): failures.append(('images','exact duplicate'))
    result={'valid':not failures,'failures':failures,'records':len(records),'splits':per_split,'prior_numeric_configurations_checked':len(prior_numbers),'prior_images_checked':len(prior_images)}
    write_json(root/cfg['experiment']/'synthetic_validation.json',result); return result
