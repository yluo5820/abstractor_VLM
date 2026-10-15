import re,json,pathlib,html
base=pathlib.Path(__file__).parent
sources=json.loads((base/'harvest_sources.json').read_text())
rows={}
def clean(s):
 s=re.sub(r'!\[[^\]]*\]\([^)]*\)','',s)
 s=re.sub(r'\[([^\[\]]*)\]\([^)]*\)',r'\1',s)
 s=re.sub(r'<[^>]*>','',s)
 s=re.sub(r'[*_`#]','',s)
 return html.unescape(' '.join(s.split())).strip(' |;')
for src in sources:
 lines=pathlib.Path(src['file']).read_text().splitlines();section=''
 for i,l in enumerate(lines):
  if re.match(r'\s*#{2,}',l): section=clean(l)
  matches=list(re.finditer(r'https?://(?:www\.)?arxiv.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})',l))
  if not matches: continue
  for m in matches:
   aid=m.group(1)
   if aid[:4]>'2609': continue
   title=''
   links=re.findall(r'\[([^\[\]]+)\]\((https?://[^)]+)\)',l)
   titles=[t for t,u in links if aid in u and len(clean(t))>15 and clean(t).lower() not in ['paper link']]
   if titles: title=clean(titles[0])
   elif l.lstrip().startswith('|'): title=clean(l.split('|')[1])
   elif src['repo'].startswith('DishengLL'):
    title=clean(l[:m.start()]);title=re.sub(r'^-?\s*\[[^]]+\]\s*','',title);title=re.sub(r'\s*\([^)]*\)\s*\[\[?paper.*','',title)
   else:
    for prev in reversed(lines[max(0,i-5):i]):
     if ('**' in prev or '<b>' in prev) and len(clean(prev))>15:
      title=clean(prev);break
   title=title.strip('[] ')
   if not title or len(title)<10 or title.lower() in ['paper','paper link']: continue
   if 'shields.io' in title or 'Project Page' in title: continue
   row=rows.setdefault(aid,{'arxiv_id':aid,'title':title,'url':'https://arxiv.org/abs/'+aid,'source_catalogues':[],'catalogue_sections':[],'catalogue_rows':[],'review_depth':'bibliographic title screen pending'})
   if src['repo'] not in row['source_catalogues']: row['source_catalogues'].append(src['repo'])
   if section not in row['catalogue_sections']:row['catalogue_sections'].append(section)
   row['catalogue_rows'].append({'file':src['file'],'line':i+1})
result=sorted(rows.values(),key=lambda x:x['arxiv_id'],reverse=True)
for n,r in enumerate(result,1):r['record_id']=n
(base/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('Distinct arXiv IDs with parsed titles:',len(result))
for j in range(0,len(result),200):
 (base/f'screen_{j//200+1}.txt').write_text('\n'.join(f"{r['record_id']} | {r['arxiv_id']} | {r['title']} | {' / '.join(r['catalogue_sections'])}" for r in result[j:j+200]))
print('\n'.join(f"{r['record_id']} | {r['arxiv_id']} | {r['title']}" for r in result[:20]))
