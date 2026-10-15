from pathlib import Path
import re,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,KeepTogether,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
base=Path(__file__).parent;out=base.parent/'output/pdf';out.mkdir(parents=True,exist_ok=True)
font='/System/Library/Fonts/Supplemental/Arial.ttf';bold='/System/Library/Fonts/Supplemental/Arial Bold.ttf';italic='/System/Library/Fonts/Supplemental/Arial Italic.ttf'
for name,path in [('Report',font),('Report-Bold',bold),('Report-Italic',italic)]:pdfmetrics.registerFont(TTFont(name,path))
pdfmetrics.registerFontFamily('Report',normal='Report',bold='Report-Bold',italic='Report-Italic',boldItalic='Report-Bold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyReport',fontName='Report',fontSize=10.2,leading=14.5,spaceAfter=7,textColor=colors.HexColor('#222222'),allowWidows=0,allowOrphans=0))
styles.add(ParagraphStyle(name='TitleReport',fontName='Report-Bold',fontSize=23,leading=28,spaceAfter=17,keepWithNext=True))
styles.add(ParagraphStyle(name='SectionReport',fontName='Report-Bold',fontSize=14,leading=18,spaceBefore=15,spaceAfter=8,keepWithNext=True))
styles.add(ParagraphStyle(name='CellReport',fontName='Report',fontSize=8.5,leading=11.5,spaceAfter=0))
styles.add(ParagraphStyle(name='RefReport',fontName='Report',fontSize=8.5,leading=11.5,spaceAfter=6))
def fmt(t):
 t=t.replace('–','-').replace('—','-').replace('\u2011','-')
 t=html.escape(t)
 t=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',lambda m:f'<link href="{m.group(2)}" color="#333333">'+(f'<super>{m.group(1)}</super>' if m.group(1).isdigit() else '<u>'+m.group(1)+'</u>')+'</link>',t)
 t=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',t)
 t=re.sub(r'`([^`]+)`',r'<font face="Report">\1</font>',t)
 return t
s=(base/'deliverables/vlm_project_report.md').read_text();lines=s.splitlines();story=[];i=0;refs=False
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line.startswith('# '):story.append(Paragraph(fmt(line[2:]),styles['TitleReport']));i+=1;continue
 if line.startswith('## '):
  title=line[3:]
  if title=='Sources':story.append(PageBreak());refs=True
  story.append(Paragraph(fmt(title),styles['SectionReport']));i+=1;continue
 if line.startswith('|'):
  rows=[]
  while i<len(lines) and lines[i].strip().startswith('|'):
   vals=[x.strip() for x in lines[i].strip().strip('|').split('|')]
   if not all(re.fullmatch(r'[-: ]+',x or '-') for x in vals):rows.append(vals)
   i+=1
  n=len(rows[0]);width=504
  widths=([35,194,95,180] if n==4 else [170,334] if n==2 else [width/n]*n)
  rendered=[]
  for rn,row in enumerate(rows):rendered.append([Paragraph(('<b>'+fmt(v)+'</b>') if rn==0 else fmt(v),styles['CellReport']) for v in row])
  table=Table(rendered,colWidths=widths,repeatRows=1,hAlign='LEFT')
  table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#777777')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dddddd')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
  story.extend([table,Spacer(1,9)]);continue
 para=[line];i+=1
 while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|')):
  para.append(lines[i].strip());i+=1
 story.append(Paragraph(fmt(' '.join(para)),styles['RefReport' if refs else 'BodyReport']))
def footer(c,d):
 c.setFont('Report',8);c.setFillColor(colors.HexColor('#777777'));c.drawRightString(558,27,str(d.page))
path=out/'vlm_project_ideas.pdf'
doc=SimpleDocTemplate(str(path),pagesize=(612,792),rightMargin=54,leftMargin=54,topMargin=43,bottomMargin=44,title='Low-Compute VLM Research Project Ideas',author='Research planning report',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader
reader=PdfReader(path)
print('Created',str(path.resolve()),'pages',len(reader.pages),'bytes',path.stat().st_size)
qa=[]
for n,page in enumerate(reader.pages,1):qa.append({'page':n,'text_chars':len(page.extract_text() or '')})
(base/'deliverables/pdf_qa.json').write_text(__import__('json').dumps(qa,indent=2))
