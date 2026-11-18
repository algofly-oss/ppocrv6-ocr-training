"""Optional official-weight selection-only precursor; no measured weights bundled."""
import hashlib
import json
from pathlib import Path
from .ppocr_models import sha

def receipt_digest(value):
    return hashlib.sha256(json.dumps({k:v for k,v in value.items() if k!='binding_sha256'},sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def membership(rows):
    """Ordered image bytes, decoded RGB pixels and literal labels; portable hashes."""
    from PIL import Image
    values=[]
    for row in rows:
        with Image.open(row['path']) as image:
            image=image.convert('RGB')
            rgb=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        values.append({'file_sha256':sha(Path(row['path'])), 'rgb_sha256':rgb, 'text':row['label']})
    return hashlib.sha256(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def validate_handoff(cfg, fit, dev):
    receipt=cfg.get('precursor_receipt')
    if not receipt:return
    value=json.loads(Path(receipt).read_text())
    if value.get('schema')!=1 or value.get('refit') is not False:
        raise ValueError('Selection-only precursor receipt required')
    if value.get('binding_sha256')!=receipt_digest(value):raise ValueError('Precursor receipt binding digest mismatch')
    if value['checkpoint_sha256']!=cfg['initial_checkpoint_sha256'] or sha(Path(cfg['initial_checkpoint']))!=value['checkpoint_sha256']:
        raise ValueError('Precursor initializer SHA mismatch')
    import torch
    checkpoint=torch.load(cfg['initial_checkpoint'],map_location='cpu',weights_only=True)
    policy=checkpoint.get('config',{})
    if checkpoint.get('full_fit') is not False or checkpoint.get('recover_bn') is not True or policy.get('recipe')!='selection_only_precursor_v1' or policy.get('refit') is not False or not 1<=checkpoint.get('epoch',0)<=30:
        raise ValueError('Checkpoint is not a selection-only precursor package')
    if policy!=value.get('training_config'):raise ValueError('Checkpoint precursor configuration differs from receipt')
    if value['fit_membership_sha256']!=membership(fit) or value['dev_membership_sha256']!=membership(dev):
        raise ValueError('Precursor FIT/DEV images/order/literal labels changed; declare a separate experiment instead')

def environment(package_versions):
    import torch, sys
    return {'python':sys.version,'packages':package_versions(),'CUDA_build':torch.version.cuda,
            'cudnn_version':torch.backends.cudnn.version(),'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,
            'TF32_cudnn':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}

def source_binding():
    result = {name:sha(Path(__file__).parent/name) for name in ['ppocr_precursor.py','ppocr_training.py','ppocr_models.py','ppocr_auxiliary.py','models.py','ppocr_data.py','metrics.py','ppocr_corrected_training.py','data.py','data_io.py']}
    vendor=Path(__file__).parent/'_vendor'/'parseq'
    result.update({str(path.relative_to(Path(__file__).parent)):sha(path) for path in sorted(vendor.rglob('*')) if path.is_file() and (path.suffix in ('.py','.yaml','.yml') or path.name=='source_pin.json')})
    return result

def train(config):
    from .ppocr_corrected_training import load_rows, package_versions
    from .ppocr_training import train as legacy_train
    cfg=dict(config)
    import sys
    vendor=(Path(__file__).parent/'_vendor'/'parseq').resolve()
    for name,module in tuple(sys.modules.items()):
        origin=getattr(module,'__file__',None)
        if (name=='strhub' or name.startswith('strhub.')) and origin and not Path(origin).resolve().is_relative_to(vendor):
            raise RuntimeError('Foreign strhub already imported; restart kernel/process with this public repository')
    if cfg.get('height')!=128 or cfg.get('width')!=320 or cfg.get('normalization','none')!='none':
        raise ValueError('Precursor requires128x320 and literal normalization:none')
    if cfg.get('device') not in ('cpu','cuda') or cfg.get('precision') not in ('fp32','bf16') or (cfg.get('device')=='cpu' and cfg.get('precision')!='fp32'):
        raise ValueError('Use CPU/fp32 or CUDA/fp32|bf16 explicitly')
    if Path(cfg['output_dir']).exists():raise ValueError('Fresh precursor output_dir required')
    fixed=dict(epochs=30,batch_size=16,learning_rate=1e-4,weight_decay=3e-5,refit=False,recover_bn=True,auxiliary=True,calibration_samples=128)
    for key,value in fixed.items():
        if key in cfg and cfg[key]!=value:raise ValueError(f'Precursor requires {key}={value}')
    if not cfg.get('train_manifest') or not cfg.get('validation_manifest'):
        raise ValueError('Selection-only precursor requires explicit FIT and DEV image_path/text manifests')
    cfg.update(recipe='selection_only_precursor_v1',refit=False,recover_bn=True,auxiliary=True,
               epochs=30,batch_size=16,learning_rate=1e-4,weight_decay=3e-5,
               calibration_samples=128,calibration_batch_size=cfg.get('calibration_batch_size',32))
    if cfg.get('calibration_batch_size',32)<1 or cfg.get('workers',0)<0:raise ValueError('Positive calibration batch and nonnegative workers required')
    cache=Path(cfg['pretrained_dir'])
    for name in ['model.onnx','trainable.onnx']:
        asset=cache/name
        if asset.exists() and cfg.get('expected_'+name.replace('.','_')+'_sha256')!=sha(asset):
            raise ValueError('Existing converted cache requires explicit expected_'+name.replace('.','_')+'_sha256; use a fresh conversion directory otherwise')
    from .ppocr_models import prepare_pretrained, FILES
    prepare_pretrained(cache)
    from .models import download_pretrained
    tiny=download_pretrained('parseq-tiny',cache/'parseq_tiny')
    assets={name:sha(cache/name) for name in list(FILES)+['model.onnx','trainable.onnx',str(tiny.relative_to(cache))]}
    sources=source_binding()
    fit,dev=load_rows(cfg,'train'),load_rows(cfg,'val')
    before=(membership(fit),membership(dev))
    checkpoint=legacy_train(cfg,fit,dev)
    if before!=(membership(fit),membership(dev)) or source_binding()!=sources or any(sha(cache/name)!=digest for name,digest in assets.items()):raise RuntimeError('Data/source/initialization assets changed during precursor')
    receipt={'schema':1,'refit':False,'checkpoint_sha256':sha(checkpoint),
             'fit_membership_sha256':before[0],'dev_membership_sha256':before[1],
             'environment':environment(package_versions),
             'training_config':{k:str(v) if isinstance(v,Path) else v for k,v in cfg.items()},
             'source_sha256':sources,'initialization_artifact_sha256':assets,
             'converted_cache_scope':'Existing caches are explicit user-pinned converted artifacts, not independently proven derived from official bytes; missing caches are freshly converted by the public path',
             'official_initialization':'Hash-verified official PP source revision e5a92bcbc5cc1b494628e458d267778f0704fd7c and public PARSeqTiny download; see source FILES/download_pretrained',
             'fit_n':len(fit),'dev_n':len(dev),'selection':'earliest literal exact; epochs1..30',
             'scope':'New recipe; historical initializer/accuracy not reproduced. DEV selection only, never gradient/refit.'}
    receipt['binding_sha256']=receipt_digest(receipt)
    path=Path(cfg['output_dir'])/'precursor_handoff.json'
    path.write_text(json.dumps(receipt,indent=2))
    return {'checkpoint':str(checkpoint),'checkpoint_sha256':receipt['checkpoint_sha256'],'precursor_receipt':str(path)}
