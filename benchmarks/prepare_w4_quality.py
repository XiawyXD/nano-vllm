"""Freeze disjoint WikiText train/test slices and W4 quality gates before evaluation.

The official dataset viewer exports JSON rows, so no datasets/pyarrow dependency
is needed. Raw response hashes, row IDs and token IDs identify the exact snapshot.
This small English held-out test is not a general chat/reasoning quality claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import requests
from transformers import AutoTokenizer
from benchmarks.cache_draft_experiment import digest, write_new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='/home/hrxu/huggingface/Qwen3-0.6B')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rows-dir',type=Path,help='Previously downloaded official viewer JSON; keeps raw hashes')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    sources, splits={},{}
    for name,split,count in (('calibration','train',16),('validation','test',32)):
        selected=[]
        for offset in range(0,1000,100):
            url='https://datasets-server.huggingface.co/rows'
            params=dict(dataset='Salesforce/wikitext',config='wikitext-2-raw-v1',split=split,
                        offset=offset,length=100)
            filename=f'{split}_{offset}.json'
            if args.rows_dir is not None:
                raw=(args.rows_dir/filename).read_bytes()
                source_url=requests.Request('GET',url,params=params).prepare().url
            else:
                response=requests.get(url,params=params,timeout=(15,45))
                response.raise_for_status()
                raw=response.content
                source_url=response.url
            (args.output/filename).write_bytes(raw)
            sources[filename]=dict(url=source_url,sha256=hashlib.sha256(raw).hexdigest())
            for row in json.loads(raw)['rows']:
                text=row['row']['text']
                if text.strip().startswith('='):
                    continue
                ids=tokenizer.encode(text,add_special_tokens=False)
                if len(ids)>=129:
                    selected.append(dict(row=row['row_idx'],tokens=ids[:129],text=text,
                                         input_tokens=128,label_tokens=128))
                if len(selected)==count:
                    break
            if len(selected)==count:
                break
        if len(selected)!=count:
            raise ValueError(f'not enough eligible rows for {name}')
        splits[name]=selected
    fingerprints={name:{digest(s['tokens']) for s in rows} for name,rows in splits.items()}
    if fingerprints['calibration'] & fingerprints['validation']:
        raise ValueError('calibration/validation token overlap')
    manifest=dict(schema='w4-quality-v1',created_at=time.time(),
        source='https://huggingface.co/datasets/Salesforce/wikitext',
        license='CC-BY-SA-3.0 / GFDL; retain source attribution when redistributing raw rows',
        source_identity='Official viewer JSON is content-addressed here; no unsupported claim of a pinned dataset git revision',
        data_sources=sources,tokenizer_path=args.model,workloads=splits,split_overlap=0,
        thresholds=dict(max_validation_mean_nll_increase_nats=.10,require_finite=True,
            acceptance_rule='All finite AND pooled validation quantized NLL minus paired unquantized same-activation-dtype NLL <= 0.10; token equality is not a quality gate'),
        metrics=['mean_teacher_forced_nll','perplexity','top1_next_token_accuracy','finite_count'],
        precision_note='If kernel requires FP16, measure BF16->FP16 separately, then quantized->FP16; do not attribute the dtype change to quantization alone.',
        calibration_policy='Calibration split only for bounded backend/group-size selection; validation untouched until selection frozen. Publisher quantization training/calibration provenance must be separately disclosed.',
        limitations=['English WikiText subset: 2048 calibration and 4096 validation label tokens',
                     'Not comprehensive Chinese/code/chat/reasoning quality certification',
                     'Quality pass does not imply a speedup, nor preservation of BF16 target distribution'])
    manifest['sha256']=digest(manifest)
    write_new(args.output/'manifest.json',manifest)
    print(json.dumps(dict(status='FROZEN_BEFORE_QUANT_EVALUATION',sha256=manifest['sha256'],
        calibration_tokens=2048,validation_tokens=4096,threshold=manifest['thresholds']),indent=2))


if __name__=='__main__':
    main()
