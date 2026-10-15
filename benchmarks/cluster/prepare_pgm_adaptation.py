"""Sample 9k official train and 1k official validation questions, independently of test."""
import argparse, hashlib, json, pathlib, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyarrow.parquet as pq
from vlm_bench.pgm_mirror import REVISION, REPOSITORY, decode_panels
from vlm_bench.data import render_pgm, PGM_PROMPT, sha256_file
from vlm_bench.run import atomic_json

p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args()
root=pathlib.Path(a.output); root.mkdir(parents=True,exist_ok=True)
spec=json.loads(pathlib.Path(__file__).with_name('pgm-adaptation-shards.json').read_text())
def download(x):
    path=root/'parquet'/pathlib.Path(x['path']).name; path.parent.mkdir(exist_ok=True)
    if not path.exists():
        temp=path.with_suffix('.partial')
        urllib.request.urlretrieve(f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{x["path"]}',temp)
        if sha256_file(temp)!=x['sha256']: raise ValueError('Downloaded shard checksum mismatch')
        temp.replace(path)
    if sha256_file(path)!=x['sha256']: raise ValueError('Existing shard changed')
    return path
with ThreadPoolExecutor(max_workers=2) as pool: paths=list(pool.map(download,spec))
manifests={}
for split,count in [('train',9000),('validation',1000)]:
    shards=[f for f in paths if f.name.startswith(split+'-')]
    refs=[]
    for path in shards:
        ids=pq.read_table(path,columns=['id']).column('id').to_pylist()
        for index,identifier in enumerate(ids):
            key=hashlib.sha256(f'20261007\0{split}\0{identifier}'.encode()).digest()
            refs.append((key,path,index,identifier))
    chosen=sorted(refs,key=lambda x:x[0])[:count]
    selected={(path,index):identifier for _,path,index,identifier in chosen}; records={}
    for path in shards:
        offset=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16,use_threads=False):
            for raw in batch.to_pylist():
                ref=(path,offset); offset+=1
                if ref not in selected: continue
                original=selected[ref]; assert raw['id']==original
                image=render_pgm(decode_panels(raw)); identifier=f'pgm/neutral/{split}/{original}'
                relative=f'images/{split}/{hashlib.sha256(identifier.encode()).hexdigest()[:24]}.png'
                target=root/relative; target.parent.mkdir(parents=True,exist_ok=True); image.save(target)
                records[identifier]={'id':identifier,'dataset':'pgm','task':'matrix-completion','regime':'neutral',
                  'split':split,'prompt':PGM_PROMPT,'choices':list('ABCDEFGH'),'answer':chr(65+int(raw['target'])),
                  'images':[relative],'metadata':{'contact_sheet_sha256':sha256_file(target),
                  'mirror_repository':REPOSITORY,'mirror_revision':REVISION,'source_shard':path.name,
                  'source_sha256':next(x['sha256'] for x in spec if pathlib.Path(x['path']).name==path.name),
                  'source_row_index':offset-1,'source_original_id':original}}
    assert len(records)==count
    dest=root/(split+'.jsonl')
    dest.write_text(''.join(json.dumps(records[f'pgm/neutral/{split}/{i}'],sort_keys=True)+'\n' for _,_,_,i in chosen))
    manifests[dest.name]=sha256_file(dest)
atomic_json(root/'READY.json',{'revision':REVISION,'manifests':manifests,'train_count':9000,'validation_count':1000,
 'selection':'smallest SHA256(seed, official split, original ID), across all shards; test never read', 'sources':spec})
