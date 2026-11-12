"""Partial ChartQA pixel mapping: missing text never invalidates a valid box."""
import math


def supported_record(record):
    """Pre-inference conservative coverage policy; raw nodes are never deleted."""
    kept=[]; exclusions=[]
    multi_bar=record.get('chart_type') in ('h_bar','v_bar') and sum(n['type']=='series' for n in record['nodes'])>1
    for node in record['nodes']:
        reason=None
        if node['type']=='axis_label': reason='Axis-description geometry not independently verified; displaced examples observed'
        elif node['type']=='tick_label' and not isinstance(node.get('text'),str): reason='Untranscribed tick geometry unverified; displaced examples observed'
        elif node['type']=='bar' and multi_bar: reason='Multi-model bar source geometry may describe stacked unions or displaced marks; physical-instance mapping unverified'
        if reason: exclusions.append({'id':node['id'],'reason':reason})
        else: kept.append(node)
    return {**record,'nodes':kept},exclusions


def chartqa_partial(annotation, image_size):
    width,height=image_size; nodes=[]; edges=[]; issues=[]
    kind=annotation.get('type')
    if kind not in ('h_bar','v_bar','line'): raise ValueError('Unsupported ChartQA chart type')
    def add(identity,typ,box=None,**attrs):
        bounds=None
        if isinstance(box,list):
            if len(box)==1: box=box[0]
            else:
                issues.append({'node':identity,'reason':'ambiguous multi-box field','source_bbox':box}); box=None
        if box is not None:
            try:
                x,y,w,h=[float(box[k]) for k in ('x','y','w','h')]
                if not all(math.isfinite(v) for v in (x,y,w,h)) or min(x,y,w,h)<0 or x+w>width or y+h>height: raise ValueError()
                if w==0 or h==0: issues.append({'node':identity,'reason':'zero-area box unavailable; no invented padding','source_bbox':box})
                else: bounds=[x,y,x+w,y+h]
            except (KeyError,TypeError,ValueError): issues.append({'node':identity,'reason':'invalid source box','source_bbox':box})
        nodes.append({'id':identity,'type':typ,'bbox':bounds,'source_bbox':box,**attrs})
    info=annotation.get('general_figure_info',{})
    add('plot_area','plot_area',info.get('figure_info',{}).get('bbox'))
    title=info.get('title',{})
    if title: add('title','title',title.get('bbox'),text=title.get('text'))
    for axis in ('x_axis','y_axis'):
        axis_info=info.get(axis,{})
        label=axis_info.get('label',{})
        if isinstance(label,dict) and label:
            boxes=label.get('bbox'); boxes=boxes if isinstance(boxes,list) else [boxes]
            for i,box in enumerate(boxes):
                add(f'{axis}:description:{i}','axis_label',box,text=label.get('text') if len(boxes)==1 else None,axis_key=axis)
            if len(boxes)>1: issues.append({'node':axis,'reason':'multi-box axis transcription not assigned to individual boxes'})
        major=axis_info.get('major_labels',{}); boxes=major.get('bboxes',[]); values=major.get('values',[])
        aligned=len(boxes)==len(values)
        if boxes and not aligned: issues.append({'node':axis,'reason':'missing or unaligned tick transcriptions; detection boxes retained'})
        for i,box in enumerate(boxes):
            add(f'{axis}:label:{i}','tick_label',box,text=str(values[i]) if aligned and values[i] is not None else None,axis_key=axis)
    for s,model in enumerate(annotation.get('models',[])):
        add(f'series:{s}','series',name=model.get('name'))
        boxes=model.get('bboxes',[]); xs=model.get('x',[]); ys=model.get('y',[])
        aligned=len(xs)==len(ys)==len(boxes)+(kind=='line')
        if not aligned: issues.append({'node':f'series:{s}','reason':'ambiguous model alignment; no inferred point/segment assignment'})
        for j,box in enumerate(boxes):
            attrs={'series':s,'source_ordinal':j}
            if kind=='line' and not aligned:
                add(f'mark:{s}:{j}','unsupported_model_box',box,**attrs); continue
            if aligned:
                if kind=='line': attrs['data_endpoints']={'x':xs[j:j+2],'y':ys[j:j+2]}
                else: attrs.update(category=xs[j],source_value=ys[j])
            add(f'mark:{s}:{j}','line_segment' if kind=='line' else 'bar',box,**attrs)
            edges.append({'source':f'mark:{s}:{j}','target':f'series:{s}','relation':'belongs_to','provenance':'source models grouping'})
    for i,item in enumerate(info.get('legend',{}).get('items',[])):
        label=item.get('label',{})
        add(f'legend:{i}','legend_item',label.get('bbox'),text=label.get('text'))
    return {'schema_version':'domain-real-partial-2','chart_type':kind,'nodes':nodes,'edges':edges,'issues':issues,
            'coordinate_mapping':'paired source xywh -> top-left pixel xyxy, identity scale',
            'missing':['exhaustive object coverage','unprovided transcriptions','axis/tick point boxes','line vertex boxes','QA evidence','calibration']}
