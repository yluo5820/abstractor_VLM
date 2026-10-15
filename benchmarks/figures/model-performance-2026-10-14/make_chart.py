"""Reproduce the three-task figure from archived evaluation receipts."""
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

OUT = Path(__file__).resolve().parent
BENCH = OUT.parents[1]
CAMPAIGNS = BENCH / '.local/campaigns'
REMOTE = BENCH / '.local/remote-runs'
PAPER = 'https://arxiv.org/html/2403.03458v2'
MODELS = [('lfm25', 'LFM2.5'), ('smolvlm2', 'SmolVLM2'), ('spatialevo', 'SpatialEvo')]
TASKS = [('rmts', 'CLEVR-ART · RMTS', 50), ('idrules', 'CLEVR-ART · Identity', 25), ('pgm', 'PGM · Neutral', 12.5)]
INDEX = CAMPAIGNS / 'lora-share-2026-10-14/public/INDEX.json'
index = json.loads(INDEX.read_text())
current = {(e['model'], e['task']): e for e in index['adapters'] if e['current_default']}
rows = []
sources = {}


def load_receipt(path):
    sources[str(path.relative_to(BENCH))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


load_receipt(INDEX)
for task, _, _ in TASKS:
    for model, label in MODELS:
        if task != 'pgm':
            path = REMOTE / f'clevr-controls-2026-09-26/{model}/{task}/full/summary.json'
        elif model == 'lfm25':
            path = REMOTE / 'lfm25-pgm-neutral-h200-full/summary.json'
        else:
            path = REMOTE / f'pgm-controls-2026-09-24-final-segments/{model}/full/summary.json'
        summary = load_receipt(path)
        group = summary['groups'][0]
        assert abs(group['correct'] / summary['evaluated'] - group['accuracy']) < 1e-12
        rows.append(dict(model=model, label=label, task=task, setup='zero_shot',
                         accuracy_percent=100*group['accuracy'], correct=group['correct'],
                         evaluated=summary['evaluated'], planned=summary['planned'], complete=summary['complete'],
                         source=str(path.relative_to(BENCH))))
        entry = current[model, task]
        t = entry['test']
        rows.append(dict(model=model, label=label, task=task, setup='lora',
                         accuracy_percent=100*t['accuracy'], evaluated=t['evaluated'], planned=t['planned'], complete=t['complete'],
                         checkpoint_key=entry['key'], checkpoint=entry['checkpoint'],
                         unique_training_questions=entry['selected_checkpoint_unique_train_questions'],
                         training_example_exposures=entry['selected_checkpoint_seen_exposures'],
                         training_pool_questions=entry['training_pool_questions'],
                         rank=entry['lora_rank'], alpha=entry['lora_alpha'], dropout=entry['lora_dropout'],
                         source=str(INDEX.relative_to(BENCH))))

paper_rows = [
    dict(task='rmts', accuracy_percent=96.34, sem_percentage_points=0.50, trained_seeds=5, table=3, train_questions=10000, epochs=50),
    dict(task='idrules', accuracy_percent=91.61, sem_percentage_points=0.20, trained_seeds=5, table=3, train_questions=10000, epochs=200),
    dict(task='pgm', accuracy_percent=91.50, trained_seeds=1, table=4, train_questions=1200000, training_fraction=1.0, epochs=73),
    dict(task='pgm', accuracy_percent=71.25, trained_seeds=1, table=5, train_questions=240000, training_fraction=0.2,
         epochs=None, epoch_note='The reduced-data epoch count is not separately specified in the cited paper table.'),
]
for r in paper_rows:
    r.update(model='Slot Abstractor', setup='published_supervised', source=PAPER)
data = dict(as_of='2026-10-14', selection='Latest evaluated LoRA-only checkpoint per model/task, not chosen by test accuracy.',
            ours=rows, slot_abstractor=paper_rows, archive_sha256=sources,
            notes=[
                'These are test accuracies, not validation accuracies.',
                'CLEVR-ART comprises two OOD tasks; PGM uses the neutral regime, not an OOD regime.',
                'All available saved test predictions are represented. PGM denominators differ between some rows; partial runs are marked.',
                'Slot Abstractor scores are paper-reported, not reproduced here. CLEVR numbers are means and SEM over five independently trained networks.',
                'Our VLM results use one trained checkpoint per model/task. No multi-seed uncertainty estimates are available.',
                'Training coverage, number of passes, pretraining, and input presentation differ. This is not a compute-matched architecture ablation.',
                'PGM LoRA selected checkpoint unique coverage: LFM 240000/1200000, SmolVLM2 82880/1200000, SpatialEvo 98672/1200000.',
                'CLEVR LoRA training pool is 9000 questions per task; selected SmolVLM2 identity checkpoint had seen only 6080 unique questions.',
                'Our inputs are composite images plus prompts; Slot Abstractor uses native separate panels and candidates.',
            ])
(OUT/'performance-data.json').write_text(json.dumps(data, indent=2)+'\n')


def pct(value):
    return str(Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)) + '%'


plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'svg.fonttype':'none',
                     'axes.titleweight':'bold', 'savefig.facecolor':'white'})
fig = plt.figure(figsize=(17, 8.8), facecolor='white')
axs = fig.subplots(1,3, sharey=True)
fig.subplots_adjust(left=.125, right=.98, top=.755, bottom=.285, wspace=.10)
INK='#172B3A'; MUTED='#526574'; GRID='#E7EDF2'
ZERO='#789FBF'; LORA='#DB7833'; SLOT='#65528A'; REDUCED='#A393BC'; CHANCE='#7D8992'
fig.text(.035,.95,'Visual reasoning: zero-shot vs. LoRA fine-tuning',fontsize=23,weight='bold',color=INK)
fig.text(.035,.914,'Latest evaluated LoRA-only checkpoints · Slot Abstractor scores reported in the paper · Test accuracy (%)',fontsize=12,color=MUTED)
legend=[Patch(facecolor=ZERO,label='Our VLMs · zero-shot'),Patch(facecolor=LORA,label='Our VLMs · LoRA'),
        Patch(facecolor=SLOT,label='Slot Abstractor · supervised (paper)'),Line2D([0],[0],color=CHANCE,linestyle=(0,(4,4)),label='Random-choice chance')]
fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.032,.883),ncol=4,frameon=False,fontsize=11,handlelength=1.5,columnspacing=2.4)
centers=[.45,1.65,2.85,4.05]
for ax,(task,title,chance) in zip(axs,TASKS):
    ax.set_xlim(0,108);ax.set_ylim(4.65,-.10)
    ax.set_xticks([0,25,50,75,100]);ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    ax.tick_params(axis='x',colors=MUTED,labelsize=10,length=0,pad=9)
    ax.tick_params(axis='y',length=0,pad=13,labelsize=12,colors=INK)
    ax.set_axisbelow(True);ax.grid(axis='x',color=GRID,lw=.8)
    for s in ax.spines.values():s.set_visible(False)
    ax.axvline(chance,color=CHANCE,lw=1.05,linestyle=(0,(4,4)),zorder=0)
    ax.axhline(1.02,color=GRID,lw=1)
    ax.set_title(title,loc='left',fontsize=14,color=INK,pad=35)
    sub='10,000 OOD test questions · chance '+str(chance)+'%' if task!='pgm' else '200,000-question test split · chance 12.5%'
    ax.text(0,1.045,sub,transform=ax.transAxes,fontsize=9.7,color=MUTED)
    references=[r for r in paper_rows if r['task']==task]
    for i,r in enumerate(references):
        y=centers[0] if task!='pgm' else centers[0]+(-.21 if i==0 else .21)
        c=SLOT if i==0 else REDUCED;h=.43 if task!='pgm' else .33
        ax.barh(y,r['accuracy_percent'],height=h,color=c,zorder=3)
        if task!='pgm':
            label=f"{r['accuracy_percent']:.2f} ± {r['sem_percentage_points']:.2f}%"
            ax.errorbar(r['accuracy_percent'],y,xerr=r['sem_percentage_points'],color=INK,capsize=3,lw=1,zorder=4)
        else:
            label=pct(r['accuracy_percent'])
            ax.text(2,y,'100% train' if i==0 else '20% train',va='center',ha='left',color='white',fontsize=9.2,zorder=5)
        ax.text(r['accuracy_percent']-2,y,label,ha='right',va='center',fontsize=11.5,weight='bold',color='white',zorder=5)
    for i,(model,_) in enumerate(MODELS):
        center=centers[i+1]
        for setup,dy,color in [('zero_shot',-.195,ZERO),('lora',.195,LORA)]:
            r=next(x for x in rows if (x['task'],x['model'],x['setup'])==(task,model,setup))
            value=r['accuracy_percent'];y=center+dy
            ax.barh(y,value,height=.32,color=color,zorder=3)
            label=pct(value)+(' *' if not r['complete'] else '')
            inside=value>83
            ax.text(value+(-1.7 if inside else 1.6),y,label,ha='right' if inside else 'left',va='center',
                    fontsize=11.5,weight='bold' if setup=='lora' else 'normal',color='white' if inside else INK,zorder=5)
    ax.set_yticks(centers,labels=['Slot Abstractor\n(published)','LFM2.5','SmolVLM2','SpatialEvo'])
    if task!='rmts':ax.tick_params(axis='y',labelleft=False)

fig.text(.125,.230,'TEST COVERAGE',fontsize=10,weight='bold',color=INK)
fig.text(.125,.203,'CLEVR: all our bars use the full 10,000-question test per task.  * PGM bars marked with an asterisk are partial.',fontsize=10,color=MUTED)
fig.text(.125,.178,'PGM evaluated questions (zero-shot → LoRA): LFM 200,000 → 200,000; Smol 116,527 → 178,411; SpatialEvo 173,554 → 200,000.',fontsize=10,color=MUTED)
fig.text(.125,.139,'TRAINING & COMPARABILITY',fontsize=10,weight='bold',color=INK)
fig.text(.125,.112,'PGM LoRA unique train questions: LFM 240,000 (20%); Smol 82,880 (6.91%); SpatialEvo 98,672 (8.22%) of the 1.2M training split.',fontsize=10,color=MUTED)
fig.text(.125,.087,'CLEVR LoRA pool: 9,000 questions/task (Smol identity checkpoint saw 6,080). Paper CLEVR bars: mean ± SEM over 5 trained networks.',fontsize=10,color=MUTED)
fig.text(.125,.062,'Training schedules, pretraining, and input formats differ; these scores do not isolate architecture. Our results are single-run checkpoints.',fontsize=10,color=MUTED)
fig.text(.125,.027,'Sources: verified local evaluation archives; Slot Abstractors, arXiv:2403.03458v2, Tables 3–5.  •  Results available 14 Oct 2026.',fontsize=9,color=MUTED)
fig.savefig(OUT/'model-performance.png',dpi=200)
fig.savefig(OUT/'model-performance.svg')
plt.close(fig)
print(json.dumps({'png':str(OUT/'model-performance.png'),'svg':str(OUT/'model-performance.svg'),'data':str(OUT/'performance-data.json'),'ours_rows':len(rows),'paper_rows':len(paper_rows)},indent=2))
