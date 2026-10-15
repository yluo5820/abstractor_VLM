from pathlib import Path
import json,re,collections
base=Path(__file__).parent;dest=base/'deliverables';dest.mkdir(exist_ok=True)
# Preserve catalogue provenance while repairing a small number of obvious title mismatches.
p=base/'corpus';rs=json.loads((p/'manifest.json').read_text());screen={}
for f in p.glob('screened_*.json'):
 d=json.loads(f.read_text());d=d.get('records',[]) if isinstance(d,dict) else d
 for r in d:screen[r.get('record_id',r.get('id'))]=r
fixes={324:('SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features','vlm_core'),368:('Qwen2.5 Technical Report','exclude_nonvisual'),608:('Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone','vlm_core'),950:('OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents','vlm_core'),1054:('BEATs: Audio Pre-Training with Acoustic Tokenizers','exclude_nonvisual'),1055:('Robust Speech Recognition via Large-Scale Weak Supervision','exclude_nonvisual'),1059:('Scaling Instruction-Finetuned Language Models','adjacent')}
for r in rs:
 i=r['record_id'];r['screening']=screen[i];r['review_depth']='title_and_catalogue_category'
 if i in fixes:
  title,disp=fixes[i];r['original_catalogue_title']=r['title'];r['title']=title;r['screening']['disposition']=disp;r['metadata_correction']='Canonical title verified on primary arXiv abstract page on 2026-09-14.'
  r['review_depth']='title_primary_metadata_verified'
 if i==1122:r['title']='A Survey of Token Compression for Efficient Multimodal Large Language Models'
counts=dict(collections.Counter(r['screening']['disposition'] for r in rs))
assert len(rs)==1220 and len(set(r['arxiv_id'] for r in rs))==1220
(dest/'literature_ledger.json').write_text(json.dumps({'cutoff':'2026-09-14','counts':counts,'screening_method':'Title and supplied catalogue category; not full-paper review. Canonical metadata unverified except explicitly marked.','records':rs},ensure_ascii=False,indent=2))
idx=['# VLM literature screening index','',f"1,220 distinct arXiv-linked catalogue records. Dispositions: {counts}. This is title/category screening, not 1,220 full-paper reviews. Canonical metadata remains unverified except explicitly marked. The report uses a separate focused primary-source audit.",'','## Catalogue sources','']
for f in ['harvest_sources.json','supplement_sources.json']:
 for x in json.loads((p/f).read_text()):
  if x.get('bytes'):idx.append(f"- [{x['repo']}](https://github.com/{x['repo']})")
idx+=['','## Screened records','','| ID | Paper / catalogue title | Disposition | Further relevance |','|---|---|---|---|']
for r in rs:
 s=r['screening'];note=s.get('shortlist_reason',s.get('screening_note',s.get('note','')));note=note if isinstance(note,str) else str(note)
 if r.get('metadata_correction'):note='Primary title corrected. '+note
 clean=lambda t:str(t).replace('|',' / ').replace('\n',' ')
 idx.append(f"| {r['record_id']} | [{clean(r['title'])}]({r['url']}) | {s['disposition']} | {clean(note)} |")
(dest/'literature_index.md').write_text('\n'.join(idx)+'\n')
# Normalize focused sources with their original inspection scopes preserved.
allrows=[];aliases={}
for fn in ['sources_reliability.json','sources_efficiency.json','sources_agents_reasoning.json']:
 d=json.loads((base/fn).read_text());rows=d.get('sources',d.get('papers',[]))
 for r in rows:
  x=dict(r);x['url']=r.get('url',r.get('primary_url'));x['date']=r.get('published',r.get('first_public_date',r.get('publication_or_submission_date','n.d.')));x['ledger_origin']=fn
  allrows.append(x)
  key=r.get('id',r.get('key',r.get('arxiv_id')))
  if key:aliases[key]=x
rename={'spatialbinding':'spatial_binding','breakeven':'break_even','etprune':'et_prune','qwen25':'qwen25_3b','qwen3':'qwen3_4b','toctou':'2604.18860','pixels':'2607.04334','crpo':'2605.21988','verigui':'2604.05477','viscritic':'2606.24525','ira':'2607.25904','worldmem':'2605.29341','memlens':'2605.14906','virtuebench':'2603.07071'}
for a,b in rename.items():aliases[a]=aliases[b]
extras=[
 {'id':'course','title':'CS 8803-VLM: Vision-Language Models & Multimodal Agents, Fall 2026','authors':['Georgia Institute of Technology, Zsolt Kira'],'date':'Fall 2026; accessed 2026-09-14','url':'https://faculty.cc.gatech.edu/~zk15/teaching/AY2026_cs8803vlm_fall/index.html','review_depth':'official_course_page','source_type':'official_page'},
 {'id':'ice','title':'ICE Cluster','authors':['Georgia Tech PACE'],'date':'accessed 2026-09-14','url':'https://pace.gatech.edu/ice-cluster/','review_depth':'official_cluster_page','source_type':'official_page'},
 {'id':'internvl','title':'InternVL3.5-4B-MPO model card and family specifications','authors':['OpenGVLab'],'date':'accessed 2026-09-14','url':'https://huggingface.co/OpenGVLab/InternVL3_5-4B-MPO','review_depth':'official_model_card','source_type':'model_card'},
 {'id':'whatsup','arxiv_id':'2310.19785','title':'What’s “up” with vision-language models? Investigating their struggle with spatial reasoning','authors':['Amita Kamath','Jack Hessel','Kai-Wei Chang'],'date':'2023-10-30','url':'https://arxiv.org/abs/2310.19785','review_depth':'abstract_and_dataset_description','source_type':'primary_paper'},
 {'id':'gqa','arxiv_id':'1902.09506','title':'GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering','authors':['Drew A. Hudson','Christopher D. Manning'],'date':'2019','url':'https://arxiv.org/abs/1902.09506','review_depth':'primary_dataset_description_excerpt','source_type':'primary_paper'},
 {'id':'videoagent','arxiv_id':'2403.10517','title':'VideoAgent: Long-form Video Understanding with Large Language Model as Agent','authors':['Xiaohan Wang','Yuhui Zhang','Orr Zohar','Serena Yeung-Levy'],'date':'2024-03-15','url':'https://arxiv.org/abs/2403.10517','review_depth':'abstract','source_type':'primary_paper'}]
for x in extras:aliases[x['id']]=x;allrows.append(x)
for key in ['qwen25','qwen3']:
 aliases[key]['authors']=['Qwen team']
 aliases[key]['date']='official model card; accessed 2026-09-14'
# Stable primary paper URLs for citation; original read URLs remain in ledgers.
for x in allrows:
 if x.get('arxiv_id'):x['citation_url']='https://arxiv.org/abs/'+x['arxiv_id']
 else:x['citation_url']=x['url']
unique={}
fulllevels={'selected_full_text_sections','full_text_sections','targeted_full_text','targeted_full_text_author_uploaded'}
for x in allrows:
 k=x.get('arxiv_id') or x['url'].rstrip('/')
 if k not in unique:unique[k]=x.copy();unique[k]['inspection_records']=[]
 unique[k]['inspection_records'].append({'depth':x.get('review_depth'),'origin':x.get('ledger_origin','root'),'read_url':x.get('read_url',x.get('full_text_url',x['url']))})
# Official model/data pages excluded from paper counts.
def ispaper(x):return bool(x.get('arxiv_id')) or x.get('source_type')=='primary_paper' or x.get('ledger_origin')=='sources_efficiency.json' or x.get('id')=='countingcopies'
papers=[x for x in unique.values() if ispaper(x)]
full=[x for x in papers if any(r['depth'] in fulllevels for r in x['inspection_records'])]
summary={'candidate_records':1220,'dispositions':counts,'focused_unique_papers':len(papers),'papers_with_selected_full_text_sections':len(full),'full_papers_read_cover_to_cover':0,'caveat':'All counts refer to explicitly recorded depth. A section read is not an all-page review.'}
(dest/'focused_sources.json').write_text(json.dumps({'summary':summary,'sources':list(unique.values())},ensure_ascii=False,indent=2))
# Final report text, calibrated to the bounded course scope.
s=(base/'report_draft.md').read_text()
s=s.replace('Estimated full study','Estimated course study').replace('| 2 | Do repeated images distort confidence or evidence attribution? | 20–60 GPU-hours |','| 2 | Do repeated images distort confidence or evidence attribution? | 25–80 GPU-hours |').replace('| 3 | When should images interact, and when should their evidence stay separate? | 25–70 GPU-hours |','| 3 | When should images interact, and when should their evidence stay separate? | 30–90 GPU-hours |')
s=s.replace('198 adjacent, 14 nonvisual, and 6 unusable or suspicious','199 adjacent, 17 nonvisual, and 2 unusable or suspicious')
s=s.replace('The detailed primary-source ledgers specify the depth paper by paper.',f"The focused audit covers {len(papers)} distinct research/position papers beyond the broad title screen; {len(full)} include selected full-text section reads. Other inspections use abstracts, introductions, method excerpts, or author project descriptions. The detailed primary-source ledgers specify the depth paper by paper.")
s=s.replace('Use approximately 400–600 base layouts.','Use 200 base scene pairs for the bounded course study; expand to 400–600 pairs only after the pilot.')
s=s.replace('Use 25%, 50%, and 75% retention as exploratory settings, with the final operating point chosen on development data.','Choose two retention budgets on development data, initially exploring 25%, 50%, and 75% only on the pilot subset.')
s=s.replace('Preserve the original spatial position IDs.','Preserve the original multimodal position IDs, including text positions, and verify that 100% retention reproduces full inference. Match answer spaces and add nonspatial binding controls of similar baseline difficulty.')
s=s.replace('Attention maps or linear probes alone do not establish the causal explanation.','Attention maps or linear probes alone do not establish the causal explanation. Include content replacement at fixed coordinates, with matched layer, token count, and donor magnitude; evaluate published pruning schedules separately from controlled same-layer interventions. A claim about background pixels would require a separate pixel intervention followed by rerunning the encoder.')
s=s.replace('Budget about 8,000–15,000 main short calls plus a limited activation-intervention subset: 25–70 GPU-hours as an initial planning envelope.','A concrete course cap is 200 scene pairs × 2 images × 2 question types, plus 400 natural-image queries: 1,200 queries total. Full inference plus three compressed policies at two budgets gives 8,400 main calls, followed by a bounded 200-case intervention study. Budget 25–70 GPU-hours for that scope; a larger two-architecture paper package may require 80–150 hours or more.')
s=s.replace('About 8,000–16,000 bounded calls yields a planning budget of 20–60 GPU-hours;','A minimal two-model study can target 8–20 GPU-hours; a larger study with ablations should reserve 25–80 GPU-hours;')
s=s.replace('Plan 25–70 GPU-hours for 10,000–20,000 short evaluations,','Plan 30–90 GPU-hours for a broader study of approximately 10,000–20,000 short evaluations,')
used=[]
def cite(m):
 key=m.group(1);assert key in aliases,key
 if key not in used:used.append(key)
 n=used.index(key)+1;x=aliases[key];url=x.get('citation_url',x['url'])
 return f'[{n}]({url})'
s=re.sub(r'\[@([a-z0-9_]+)\]',cite,s)
s+='\n\n## Sources\n\nNumbered references correspond to citations in the report. Full inspection depths and additional audited sources are in the accompanying primary-source ledger. Dates are first-public dates or the recorded publication date; updated versions and unverified venue claims are distinguished in the ledger.\n\n'
for n,key in enumerate(used,1):
 x=aliases[key];url=x.get('citation_url',x['url']);auth=x.get('authors') or ['AAAI' if 'aaai.org' in url else 'ACL Anthology' if 'aclanthology.org' in x.get('url','') else 'arXiv']
 if isinstance(auth,list):auth=', '.join(auth[:3])+(' et al.' if len(auth)>3 else '')
 date=str(x.get('date','n.d.')).replace('/','-');url=x.get('citation_url',x['url'])
 s+=f"{n}. {auth}. [{x['title']}]({url}). {date}.\n\n"
(dest/'vlm_project_report.md').write_text(s)
(base/'source_aliases.json').write_text(json.dumps({k:aliases[k] for k in used},ensure_ascii=False,indent=2))
(dest/'review_scope.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2));print('Report words',len(s.split()),'references',len(used))
