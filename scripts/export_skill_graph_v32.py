"""Export the live registry without requiring a Graphviz installation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
from matplotlib import font_manager
from simbench.assembly.graph import catalog_graph

p=Path('docs/skill-graph');p.mkdir(parents=True,exist_ok=True)
g=catalog_graph()
(p/'skill_graph.json').write_text(json.dumps(g,ensure_ascii=False,indent=2),encoding='utf-8')
fonts={f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.sans-serif']=[f for f in ['Microsoft YaHei','SimHei','DejaVu Sans'] if f in fonts]
plt.rcParams['svg.fonttype']='none'
rows=[['detect','estimate_pose','estimate_grasp','estimate_push_pose','plan_path'],['gripper','move','grasp','place'],['insert','push','press','wipe']]
pos={name:(1.5+2.5*i,7.5-2.6*j) for j,row in enumerate(rows) for i,name in enumerate(row)}
fig,ax=plt.subplots(figsize=(15,10));fig.patch.set_facecolor('#f5f7fa');ax.set_facecolor('#f5f7fa')
boxes={}
for n in g['nodes']:
    x,y=pos[n['name']]
    boxes[n['name']]=FancyBboxPatch((x-1.02,y-.43),2.04,.86,boxstyle='round,pad=.08',facecolor='#dceafb' if n['kind']=='information' else '#e1f0e4',edgecolor='#71849b',zorder=3)
    ax.add_patch(boxes[n['name']])
    ax.text(x,y,n['label']+'\n'+n['name'],ha='center',va='center',fontsize=10,zorder=4)
for i,e in enumerate(g['edges']):
    a,b=pos[e['source']],pos[e['target']]
    ax.add_patch(FancyArrowPatch(a,b,patchA=boxes[e['source']],patchB=boxes[e['target']],connectionstyle='arc3,rad='+str(.13 if i%2 else -.13),arrowstyle='-|>',mutation_scale=12,color='#71849b',alpha=.65,linewidth=1,shrinkA=2,shrinkB=3,zorder=2))
ax.set_title(f'TwinGraph · {len(g["nodes"])} 个通用原子技能\n条件依赖图；边表示部分输入/状态，不保证执行成功',fontsize=17,pad=24)
ax.text(.5,.2,'保持夹持送入：抓取 → 移动 → 接触规划 → 插入 → 压靠 → 放置\n释放后推入：放置 → 退出 → 检测 → 推动位置估计 → 规划 → 移动 → 空夹爪闭合 → 推动\n两种方案由规划器选择；本次任务从滑块开始，到把手装好并释放后结束。',fontsize=11,va='center',color='#24384d')
ax.set_xlim(0,13);ax.set_ylim(-.5,8.7);ax.axis('off');fig.tight_layout()
fig.savefig(p/'skill_graph.svg',facecolor=fig.get_facecolor());fig.savefig(p/'skill_graph.png',dpi=150,facecolor=fig.get_facecolor())
print(json.dumps(dict(nodes=len(g['nodes']),edges=len(g['edges']))))
