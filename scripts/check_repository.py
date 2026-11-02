"""Standard-library repository checks; never import Torch or initialize CUDA."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED_DIRS={'.git','.venv','.venv-flashinfer','venv','env','__pycache__','.pytest_cache',
 'node_modules','build','dist'}
ROOT_ONLY_DIRS={'artifacts','cuda-learning','models','checkpoints','huggingface','results','runs','output'}
MODEL_SUFFIXES={'.safetensors','.pt','.pth','.ckpt','.gguf','.onnx','.bin','.so','.pyd','.dll','.whl'}
SECRET_PATTERNS={
 'private-key':re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
 'github-token':re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
 'github-fine-grained-token':re.compile(rb'\bgithub_pat_[A-Za-z0-9_]{50,}\b'),
 'huggingface-token':re.compile(rb'\bhf_[A-Za-z0-9]{25,}\b'),
 'aws-access-key':re.compile(rb'\bAKIA[0-9A-Z]{16}\b'),
 'api-secret-key':re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b'),
}
LINK=re.compile(r'!?\[[^\]\n]*\]\((<[^>\n]+>|[^\s)]+)(?:\s+["\'][^\n]*?["\'])?\)')

def git(args, *, data=None):
 return subprocess.run(['git',*args],cwd=ROOT,input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

def git_root_matches():
 r=git(['rev-parse','--show-toplevel'])
 return r.returncode==0 and Path(r.stdout.decode().strip()).resolve()==ROOT.resolve()

def files():
 if git_root_matches():
  r=git(['ls-files','--cached','--others','--exclude-standard','-z'])
  if r.returncode:raise RuntimeError('git file inventory failed')
  return sorted({ROOT/p.decode() for p in r.stdout.split(b'\0') if p and (ROOT/p.decode()).is_file()})
 return sorted(p for p in ROOT.rglob('*') if p.is_file()
     and p.relative_to(ROOT).parts[0] not in ROOT_ONLY_DIRS
     and not any(part in EXCLUDED_DIRS or part.endswith('.egg-info') for part in p.relative_to(ROOT).parts))

def secret_issues(raw, name):
 rows=[]
 for kind,pattern in SECRET_PATTERNS.items():
  for match in pattern.finditer(raw):
   rows.append(dict(kind=kind,path=name,line=raw[:match.start()].count(b'\n')+1))
 return rows

def check(history=False):
 issues=[];items=files();total=0;largest=[]
 for p in items:
  name=p.relative_to(ROOT).as_posix();size=p.stat().st_size;total+=size;largest.append((size,name))
  if p.is_symlink():issues.append(dict(kind='symlink-in-public-tree',path=name));continue
  if size>=50*2**20:issues.append(dict(kind='large-file',path=name,bytes=size));continue
  if p.suffix.lower() in MODEL_SUFFIXES:issues.append(dict(kind='model-or-binary',path=name))
  if name.startswith('artifacts/') or name.startswith('cuda-learning/') or p.name.startswith('.env') and p.name!='.env.example':
   issues.append(dict(kind='local-only-file',path=name))
  raw=p.read_bytes()
  if b'\x00' not in raw:issues.extend(secret_issues(raw,name))
  if p.suffix.lower()=='.md':
   content=raw.decode('utf-8-sig')
   for match in LINK.finditer(content):
    target=match.group(1).strip('<>');parsed=urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path:continue
    if target.startswith('/home/') or re.match(r'^[A-Za-z]:[/\\]',target):
     issues.append(dict(kind='machine-local-markdown-link',path=name,line=content[:match.start()].count('\n')+1))
     continue
    path=(ROOT/parsed.path.lstrip('/')) if parsed.path.startswith('/') else p.parent/unquote(parsed.path)
    if not path.exists():issues.append(dict(kind='missing-relative-link',path=name,target=target))
 history_report=dict(requested=history,checked=False)
 if history:
  if not git_root_matches():raise RuntimeError('--history requires this directory to be a Git root')
  rev=git(['rev-list','--objects','HEAD'])
  if rev.returncode:raise RuntimeError('could not enumerate committed history')
  rows=git(['cat-file','--batch-check=%(objecttype) %(objectname) %(objectsize) %(rest)'],data=rev.stdout)
  if rows.returncode:raise RuntimeError('could not inspect committed objects')
  blobs=[]
  for line in rows.stdout.decode().splitlines():
   parts=line.split(' ',3)
   if parts[0]!='blob':continue
   oid,size,name=parts[1],int(parts[2]),parts[3] if len(parts)>3 else ''
   blobs.append((oid,size,name))
   if size>=50*2**20 or name.startswith('artifacts/') or Path(name).suffix.lower() in MODEL_SUFFIXES:
    issues.append(dict(kind='history-payload',path=name,bytes=size))
  # Batch-read textual history without exposing matching credential bytes.
  request=''.join(oid+'\n' for oid,size,name in blobs if size<5*2**20).encode()
  content=git(['cat-file','--batch'],data=request)
  if content.returncode:raise RuntimeError('could not read history blobs')
  data=content.stdout;offset=0;names={oid:name for oid,size,name in blobs}
  while offset<len(data):
   end=data.index(b'\n',offset);header=data[offset:end].split();size=int(header[2]);oid=header[0].decode()
   raw=data[end+1:end+1+size];offset=end+1+size+1
   if b'\x00' not in raw:issues.extend(secret_issues(raw,'history:'+names[oid]))
  history_report=dict(requested=True,checked=True,blobs=len(blobs),max_blob_bytes=max((s for _,s,_ in blobs),default=0))
 return dict(status='PASS' if not issues else 'FAIL',files=len(items),bytes=total,
             largest=[dict(path=n,bytes=s) for s,n in sorted(largest,reverse=True)[:5]],
             history=history_report,issues=issues,
             scope='Source-size/credential-pattern/relative-link checks; not a GPU or performance test')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--history',action='store_true',help='also inspect objects reachable from HEAD')
 a=p.parse_args();report=check(a.history);print(json.dumps(report,indent=2,ensure_ascii=False))
 return 0 if report['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
