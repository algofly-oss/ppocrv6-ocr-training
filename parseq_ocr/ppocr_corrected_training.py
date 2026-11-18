"""Opt-in fixed-BN, micro1 corrected-price recipe. No measured weights bundled.

Initialization is REQUIRED and hash-bound: generic initialization is not silently
substituted for the historical proper-fitting BN+auxiliary warm start.
"""
from pathlib import Path
import contextlib,hashlib,json,os,random,sys,importlib.metadata
import numpy as np
import torch
from PIL import Image,ImageEnhance
from .ppocr_models import VOCAB,create_model,restore_bn,preprocess,decode,sha
from .ppocr_data import scan_labeled
from .metrics import EXACT_POLICY

RECIPE='corrected_fixed_bn_v1'
def tree(value):
 h=hashlib.sha256()
 def add(v):
  if torch.is_tensor(v):h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):h.update(repr(k).encode());add(v[k])
  elif isinstance(v,(list,tuple)):
   for x in v:add(x)
  else:h.update(repr(v).encode())
 add(value);return h.hexdigest()
def package_versions():
 result={}
 for name in ['torch','numpy','Pillow','opencv-python-headless','opencv-python','onnx2torch','onnx','paddle2onnx']:
  try:result[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:result[name]=None
 result['cv2_runtime']=__import__('cv2').__version__
 return result
def seed(value):
 torch.set_num_threads(2);random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False

def contiguous_gradient(g):return g.contiguous()
def _layout_hook(module,args,output):
 if torch.is_grad_enabled() and torch.is_tensor(output) and output.requires_grad:output.register_hook(contiguous_gradient)
 return None

def install_bn_layout(model):
 if not getattr(model,'_corrected_bn_layout',False):
  model._corrected_bn_handles=[b.register_forward_hook(_layout_hook)for b in model.modules()if isinstance(b,torch.nn.modules.batchnorm._BatchNorm)];model._corrected_bn_layout=True
 return len(model._corrected_bn_handles)
def train_mode(model):
 model.train()
 for b in model.modules():
  if isinstance(b,torch.nn.modules.batchnorm._BatchNorm):b.eval();assert b.weight.requires_grad and b.bias.requires_grad

def targets(texts,device):
 if any(not s or set(s)-set(VOCAB[1:])for s in texts):raise ValueError('Empty/unsupported literal label; printable ASCII plus space only; no normalization')
 if any(len(s)+sum(a==b for a,b in zip(s,s[1:]))>40 for s in texts):raise ValueError('CTC target including repeated-character blanks exceeds40frames')
 return torch.tensor([VOCAB.index(c)for s in texts for c in s],device=device),torch.tensor([len(s)for s in texts],device=device)
def ctc_sum(model,x,texts,global_images,amp=False):
 assert x.shape==(len(texts),3,128,320)and global_images>=len(texts)>0
 assert all(not b.training for b in model.modules()if isinstance(b,torch.nn.modules.batchnorm._BatchNorm))
 ids,lengths=targets(texts,x.device)
 with torch.autocast('cuda',dtype=torch.bfloat16)if amp else contextlib.nullcontext():z=model(x)
 assert z.shape==(len(texts),40,96)and torch.isfinite(z).all()
 logp=z.float().log_softmax(-1).transpose(0,1);steps=torch.full_like(lengths,40);loss=(torch.nn.functional.ctc_loss(logp,ids,steps,lengths,blank=0,reduction='none',zero_infinity=False)/lengths).sum()/global_images
 if not torch.isfinite(loss):raise RuntimeError('Nonfinite CTC loss')
 return loss,z

def pixels(row,augmentation_seed=None):
 with Image.open(row['path'])as src:image=src.convert('RGB')
 try:
  if augmentation_seed is not None:
   rng=random.Random(augmentation_seed)
   for cls in [ImageEnhance.Brightness,ImageEnhance.Contrast]:
    if rng.random()<.4:
     new=cls(image).enhance(rng.uniform(.8,1.2));image.close();image=new
   if rng.random()<.25:
    new=image.rotate(rng.uniform(-2,2),Image.Resampling.BILINEAR,fillcolor=(127,127,127));image.close();image=new
  return torch.from_numpy(preprocess(image,128))
 finally:image.close()
def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=3e-5)
def validate_config(cfg):
 if cfg.get('height')!=128 or cfg.get('width')!=320:raise ValueError('corrected_fixed_bn_v1 requires128x320,40frames,96classASCIIcodec')
 if cfg.get('initial_codec')!='ppocr_ascii96_v1' or VOCAB!=['']+list(__import__('string').ascii_letters+__import__('string').digits+__import__('string').punctuation+' '):raise ValueError('Required initial_codec:ppocr_ascii96_v1; blank0 + ordered95printableASCII')
 digest=cfg.get('initial_checkpoint_sha256')
 if not isinstance(digest,str)or len(digest)!=64 or set(digest)-set('0123456789abcdef'):raise ValueError('Required lowercase64hex initial_checkpoint_sha256')
 if not cfg.get('initial_checkpoint'):raise ValueError('Required explicit initial_checkpoint')

def load_rows(cfg,split):
 manifest=cfg.get('train_manifest'if split=='train'else'validation_manifest')
 if not manifest:return scan_labeled(Path(cfg['dataset_root'])/split)
 manifest=Path(manifest);rows=[]
 for line in manifest.read_text(encoding='utf-8').splitlines():
  value=json.loads(line)
  if not isinstance(value.get('text'),str)or not isinstance(value.get('image_path'),str):raise ValueError('JSONL needs literal text and image_path strings')
  p=Path(value['image_path']).expanduser();p=p if p.is_absolute()else manifest.parent/p
  rows.append({'image':str(p.resolve()),'path':str(p.resolve()),'label':value['text']})
 if not rows:raise ValueError('Empty image/text manifest')
 return rows

def require_initial(cfg):
 validate_config(cfg);p=Path(cfg['initial_checkpoint'])
 if not p.is_file()or sha(p)!=cfg['initial_checkpoint_sha256']:raise ValueError('Required initial checkpoint missing or SHA256 mismatch; no generic substitution')
 return p

def load_initial(cfg):
 p=require_initial(cfg)
 cp=torch.load(p,map_location='cpu',weights_only=True);state=cp.get('state_dict',cp)if isinstance(cp,dict)else None
 if not isinstance(state,dict)or not state or not all(torch.is_tensor(v)for v in state.values()):raise ValueError('Initialization must be a tensor state_dict or public PP checkpoint containing state_dict')
 model=create_model(cfg);restore_bn(model);model.load_state_dict(state,strict=True)
 if sum(p.numel()for p in model.parameters())!=15579938:raise ValueError('Expected15,579,938-parameter PP medium128x320 architecture')
 assert tree(model.state_dict())==tree(state);install_bn_layout(model);return model.eval()
def atomic_json(p,value):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def save_checkpoint(path,model,opt,cfg,metadata):
 # Keep inference payload compatible with existing native/export loaders.
 config={k:str(v)if isinstance(v,Path)else v for k,v in cfg.items()};payload={'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'config':config,'recover_bn':True,'selection_policy':EXACT_POLICY,'recipe':RECIPE,'metadata':metadata,'model_state_sha256':tree(model.state_dict()),'optimizer_sha256':tree(opt.state_dict()),'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':{'name':np.random.get_state()[0],'keys':np.random.get_state()[1].tolist(),'position':np.random.get_state()[2],'has_gauss':np.random.get_state()[3],'cached_gaussian':np.random.get_state()[4]},'cuda_rng':torch.cuda.get_rng_state()if torch.cuda.is_initialized()else None}
 tmp=Path(path).with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
def restore_checkpoint(path,model,opt):
 cp=torch.load(path,map_location='cpu',weights_only=True)
 if cp.get('recipe')!=RECIPE or cp.get('selection_policy')!=EXACT_POLICY:raise ValueError('Incompatible recipe/selection checkpoint')
 model.load_state_dict(cp['state_dict'],strict=True);opt.load_state_dict(cp['optimizer']);assert tree(model.state_dict())==cp['model_state_sha256']and tree(opt.state_dict())==cp['optimizer_sha256'];return cp
@torch.inference_mode()
def evaluate(model,rows,out):
 model.eval();predictions=[];out=Path(out);out.mkdir();p=out/'predictions.jsonl'
 with p.open('x')as f:
  for row in rows:
   x=pixels(row);z=model(x[None].to(next(model.parameters()).device)).float();assert z.shape==(1,40,96)and torch.isfinite(z).all();ids=z.argmax(-1)[0].cpu().tolist();raw=decode([ids])[0];predictions.append(raw);f.write(json.dumps({'image':row['image'],'prediction':raw,'CTC_frame_ids':ids},ensure_ascii=False)+'\n');f.flush()
  os.fsync(f.fileno())
 result={'literal_correct':sum(r['label']==p for r,p in zip(rows,predictions)),'n':len(rows),'raw_sha256':sha(p),'raw_durable_before_GT_join':True};atomic_json(out/'metrics.json',result);return result

def train(cfg,smoke_steps=None):
 if cfg.get('recipe')!=RECIPE:raise ValueError('Explicit corrected_fixed_bn_v1 recipe required')
 if cfg.get('normalization','none')!='none':raise ValueError('Literal labels require normalization:none')
 if cfg.get('epochs')!=20 or cfg.get('batch_size')!=16 or cfg.get('microbatch')!=1:raise ValueError('This named recipe fixes20epochs/effective16/micro1')
 if cfg.get('learning_rate',1e-4)!=1e-4 or cfg.get('weight_decay',3e-5)!=3e-5 or cfg.get('clip_norm',5)!=5 or cfg.get('schedule','constant')!='constant':raise ValueError('Recipe fixes AdamW lr1e-4/wd3e-5/clip5/constant schedule')
 if not cfg.get('recover_bn',True)or cfg.get('auxiliary',False):raise ValueError('Recipe reconstructs source BN and has no auxiliary reader')
 if cfg.get('precision')not in ('bf16','fp32')or cfg.get('device')not in ('cpu','cuda'):raise ValueError('Expected bf16 or fp32 and cpu or cuda')
 if cfg['device']=='cpu'and cfg['precision']!='fp32':raise ValueError('CPU uses fp32; measured recipe used CUDA bf16')
 if smoke_steps is not None and smoke_steps!=2:raise ValueError('Smoke is exactly2optimizer updates')
 require_initial(cfg);fit=load_rows(cfg,'train');dev=load_rows(cfg,'val')
 from .ppocr_precursor import validate_handoff
 validate_handoff(cfg,fit,dev)
 from .data import audit_splits
 if audit_splits(fit,dev)['pixel_duplicates_across_splits']:raise ValueError('FIT/DEV RGB overlap')
 for row in fit+dev:targets([row['label']],'cpu')
 if len(fit)<32 and smoke_steps:raise ValueError('Two effective16 smoke updates need at least32FIT images')
 smoke_receipt=Path(cfg['output_dir'])/'smoke/completion.json'
 if smoke_steps is None and not smoke_receipt.is_file():raise ValueError('Run own two-update --smoke-steps2 before full training')
 out=Path(cfg['output_dir'])/('smoke'if smoke_steps else'full_fit');out.mkdir(parents=True,exist_ok=False);seed(cfg.get('seed',20261114));m=load_initial(cfg).to(cfg['device']);opt=optimizer(m);initial=tree(m.state_dict());buffers=tree(dict(m.named_buffers()));steps=exposures=0;history=[];best=-1;selected=0
 binding={'precursor_receipt_sha256':sha(cfg['precursor_receipt']) if cfg.get('precursor_receipt') else None,'initial_checkpoint_sha256':sha(cfg['initial_checkpoint']),'initial_state_sha256':initial,'FIT':[{'file_sha256':sha(r['path']),'text_sha256':hashlib.sha256(r['label'].encode()).hexdigest()}for r in fit],'DEV':[{'file_sha256':sha(r['path']),'text_sha256':hashlib.sha256(r['label'].encode()).hexdigest()}for r in dev],'source_sha256':sha(__file__),'dependency_source_sha256':{name:sha(Path(__file__).with_name(name))for name in ['ppocr_models.py','ppocr_data.py','metrics.py','data.py','data_io.py','ppocr_precursor.py']},'launch':{'argv':sys.argv,'cwd':str(Path.cwd()),'python':sys.version,'packages':package_versions(),'CUDA_build':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,'TF32_cudnn':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark},'vocabulary':VOCAB,'recipe':RECIPE,'selection':'epochs1..20earliestliteral;epoch0diagnostic only; no refit orresume','config':{k:str(v)if isinstance(v,Path)else v for k,v in cfg.items()}};contract=tree({k:binding[k]for k in ['precursor_receipt_sha256','initial_checkpoint_sha256','initial_state_sha256','FIT','DEV','source_sha256','dependency_source_sha256','vocabulary','recipe','selection','config']});atomic_json(out/'binding.json',binding)
 if smoke_steps is None:
  smoke=json.loads(smoke_receipt.read_text())
  if not(smoke.get('pass')and smoke.get('steps')==2 and smoke.get('exposures')==32 and smoke.get('replay_contract_sha256')==contract):raise ValueError('Own smoke source/config/input/initialcheckpoint contract changed; choose a fresh output directory')
 history.append({'epoch':0,**evaluate(m,dev,out/'epoch0'),'selection_eligible':False})
 with (out/'inputs.jsonl').open('x')as stream:
  for epoch in range(1,21):
   order=list(range(len(fit)));random.Random(cfg.get('seed',20261114)+epoch).shuffle(order);losses=[]
   for start in range(0,len(order),16):
    indices=order[start:start+16];n=len(indices);train_mode(m);opt.zero_grad(set_to_none=True);total=0
    for position,index in enumerate(indices,start):
     row=fit[index];microseed=cfg.get('seed',20261114)+epoch*10000+position;x=pixels(row,microseed);torch.manual_seed(microseed)
     if cfg['device']=='cuda':torch.cuda.manual_seed_all(microseed)
     loss,_=ctc_sum(m,x[None].to(cfg['device']),[row['label']],n,amp=cfg['device']=='cuda'and cfg['precision']=='bf16');loss.backward();total+=float(loss.detach());stream.write(json.dumps({'epoch':epoch,'step':steps+1,'source_index':index,'position':position,'microseed':microseed,'pixel_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),'text_sha256':hashlib.sha256(row['label'].encode()).hexdigest()})+'\n');exposures+=1
    norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True));assert norm>0;opt.step();steps+=1;assert tree(dict(m.named_buffers()))==buffers;losses.append(total)
    if smoke_steps and steps==smoke_steps:break
   stream.flush();os.fsync(stream.fileno());score=evaluate(m,dev,out/f'epoch{epoch}');history.append({'epoch':epoch,'CTC_mean':sum(losses)/len(losses),**score,'selection_eligible':True});metadata={'epoch':epoch,'steps':steps,'exposures':exposures,'initial_state_sha256':initial,'binding_sha256':sha(out/'binding.json'),'BN_buffers_sha256':buffers};save_checkpoint(out/'last.pt',m,opt,cfg,metadata)
   if score['literal_correct']>best:best=score['literal_correct'];selected=epoch;save_checkpoint(out/'model.pt',m,opt,cfg,metadata)
   atomic_json(out/'history.json',history)
   if smoke_steps:break
 # Actual fresh models/optimizers and full validation replay for selected+terminal.
 del m,opt
 for role,expected in [('model',next(r for r in history if r['epoch']==selected)),('last',history[-1])]:
  m=load_initial(cfg).to(cfg['device']);opt=optimizer(m);restore_checkpoint(out/(role+'.pt'),m,opt);assert tree(dict(m.named_buffers()))==buffers;fresh=evaluate(m,dev,out/('fresh_'+role));assert fresh=={k:expected[k]for k in fresh};del m,opt
 if any(sha(Path(__file__).with_name(name))!=digest for name,digest in binding['dependency_source_sha256'].items()):raise RuntimeError('Training dependency source changed during run')
 if cfg.get('precursor_receipt') and sha(cfg['precursor_receipt'])!=binding['precursor_receipt_sha256']:raise RuntimeError('Precursor handoff changed during corrected training')
 if sha(__file__)!=binding['source_sha256'] or sha(cfg['initial_checkpoint'])!=binding['initial_checkpoint_sha256']:raise RuntimeError('Training source/initialcheckpoint changed during run')
 for name,h in binding['dependency_source_sha256'].items():
  if sha(Path(__file__).with_name(name))!=h:raise RuntimeError('Dependency source changed during run')
 for group,rows in [('FIT',fit),('DEV',dev)]:
  for row,expected in zip(rows,binding[group]):
   if sha(row['path'])!=expected['file_sha256']or hashlib.sha256(row['label'].encode()).hexdigest()!=expected['text_sha256']:raise RuntimeError('Image/text bindings changed')
 result={'pass':True,'recipe':RECIPE,'checkpoint':str(out/'model.pt'),'checkpoint_sha256':sha(out/'model.pt'),'selected_epoch':selected,'selected_literal_correct':best,'validation_n':len(dev),'steps':steps,'exposures':exposures,'replay_contract_sha256':contract,'BN_buffers_unchanged':True,'fresh_selected_terminal_replay':True,'scope':'User-supplied data/initialcheckpoint; recipe operations reproduced, not measured104weights/counts. CPUfp32smoke is not CUDAbf16training proof.'};atomic_json(out/'completion.json',result);return result
