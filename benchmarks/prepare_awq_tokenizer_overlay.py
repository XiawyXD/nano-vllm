"""显式、可审计的 tokenizer 配置派生；绝不修改原始模型或放宽映射检查。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from transformers import AutoTokenizer


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    original=AutoTokenizer.from_pretrained(args.source,local_files_only=True)
    target=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    if original.get_vocab()!=target.get_vocab(): raise ValueError('token-to-ID mapping differs')
    for name in ('bos_token_id','eos_token_id','pad_token_id'):
        if getattr(original,name)!=getattr(target,name): raise ValueError(f'{name} differs')
    hashes={}
    for name in ('tokenizer.json','vocab.json','merges.txt'):
        left,right=sha(args.source/name),sha(args.tokenizer/name)
        if left!=right: raise ValueError(f'{name} contents differ; overlay is not safe')
        hashes[name]=left
    parent=json.loads((args.source/'download_manifest.json').read_text())
    if parent['status']!='VERIFIED_OFFICIAL_REVISION' and 'VERIFIED' not in parent['status']:
        raise ValueError('parent checkpoint lacks verified download provenance')
    args.output.mkdir(parents=True,exist_ok=False)
    for file in sorted(args.source.iterdir()):
        if not file.is_file(): continue
        if file.name=='tokenizer_config.json':
            shutil.copy2(args.tokenizer/file.name,args.output/file.name)
        elif file.name=='download_manifest.json':
            shutil.copy2(file,args.output/'parent_download_manifest.json')
        elif file.suffix=='.safetensors':
            os.link(file,args.output/file.name)  # byte-identical data; no rewritten weights.
        else:
            shutil.copy2(file,args.output/file.name)
    derived=AutoTokenizer.from_pretrained(args.output,local_files_only=True)
    from nanovllm.engine.proposer import validate_tokenizers
    validate_tokenizers(target,derived)
    report=dict(status='VERIFIED_DERIVED_TOKENIZER_OVERLAY',publisher_class='community_with_explicit_local_tokenizer_overlay',
        parent=str(args.source.resolve()),parent_revision=parent.get('revision'),
        parent_manifest_sha256=sha(args.source/'download_manifest.json'),
        official_tokenizer_path=str(args.tokenizer.resolve()),tokenizer_revision='UNKNOWN_LOCAL_CONTENT_ADDRESSED',
        unchanged_tokenizer_files=hashes,source_special_ids=original.all_special_ids,
        resulting_special_ids=derived.all_special_ids,
        changed_runtime_file='tokenizer_config.json',
        original_config_sha256=sha(args.source/'tokenizer_config.json'),
        new_config_sha256=sha(args.output/'tokenizer_config.json'),
        reason='Transformers5 interprets old extra_special_tokens={} differently; use verified identical tokenizer data plus explicit base classification/template.',
        weights_changed=False,license=parent.get('license','See unchanged upstream model card; community card declares GPL-3.0'),
        warning='Not an official Qwen quantized checkpoint; publisher calibration overlap remains unknown.',
        files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(args.output.iterdir()) if f.is_file()})
    (args.output/'download_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(status=report['status'],output=str(args.output),special_ids=derived.all_special_ids)))


if __name__=='__main__': main()
