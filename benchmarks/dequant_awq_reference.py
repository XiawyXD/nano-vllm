"""Independent, unfused AWQ -> FP16 diagnostic checkpoint; never a speed backend.

Decode the published AutoAWQ GEMM storage directly. No nano-vLLM unpacker,
fusion loader, Marlin repacker or execution kernel is imported here.
"""
import argparse, hashlib, json, pathlib, shutil, time
import torch
from safetensors.torch import load_file, save_file

def unpack(storage):
    # Physical low-to-high nibbles are channels 0,2,4,6,1,3,5,7.
    unsigned = storage.to(torch.int64) & 0xffffffff
    out = torch.empty((storage.shape[0], storage.shape[1]*8), dtype=torch.int16)
    for physical in range(8):
        logical = (physical % 4)*2 + physical//4
        out[:,logical::8] = ((unsigned >> (physical*4)) & 15).to(torch.int16)
    return out

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    a=p.parse_args(); torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False)
    config=json.loads((a.source/'config.json').read_text())
    q=config.pop('quantization_config')
    assert q['bits']==4 and q['group_size']==128 and q['zero_point'] and q['version'].lower()=='gemm'
    data={}; inputs={}; report=[]
    for f in sorted(a.source.glob('*.safetensors')):
        inputs[f.name]=sha(f)
        part=load_file(str(f)); assert not data.keys() & part.keys(); data.update(part)
    for key in list(data):
        if not key.endswith('.qweight'): continue
        prefix=key[:-8]
        weights=unpack(data.pop(key)).float()
        zeros=unpack(data.pop(prefix+'.qzeros')).float()
        scales=data.pop(prefix+'.scales').float()
        assert zeros.shape==scales.shape and weights.shape[0]==128*scales.shape[0]
        dense=((weights.reshape(scales.shape[0],128,-1)-zeros[:,None,:])*scales[:,None,:]).reshape(weights.shape).half().T.contiguous()
        assert torch.isfinite(dense).all()
        data[prefix+'.weight']=dense
        report.append(dict(name=prefix,shape=list(dense.shape),scale_min=scales.min().item(),scale_max=scales.max().item()))
    assert len(report)==28*7
    # Original non-quantized tensors retained bit-for-bit in the file; ordinary
    # FP16 model loading performs exactly the same activation-dtype conversion.
    output=a.output/'model.safetensors'
    save_file(data,str(output),metadata={'format':'pt'})
    config['torch_dtype']='float16'
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    for f in a.source.iterdir():
        if f.is_file() and f.name not in ('config.json','download_manifest.json','parent_download_manifest.json','README.md') and f.suffix!='.safetensors':
            shutil.copy2(f,a.output/f.name)
    manifest=dict(role='DIAGNOSTIC_DENSE_REFERENCE_NOT_QUANTIZED_EXECUTION',created=time.time(),source=str(a.source),source_weights_sha256=inputs,output_sha256=sha(output),projections=report,script_sha256=sha(pathlib.Path(__file__)))
    (a.output/'diagnostic_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(output=str(a.output),projections=len(report),sha256=manifest['output_sha256'])))

if __name__=='__main__': main()
