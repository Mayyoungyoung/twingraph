"""Wait for the fixed dataset, audit, train, evaluate and write the final report."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from scripts.train_value_v32 import read,write,sha


def status(out,stage,**fields):
    write(out/'status.json',dict(stage=stage,updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**fields))


def render_report(out):
    audit=read(out/'dataset_audit.json'); training=read(out/'training/training_report.json')
    offline=read(out/'training/offline_test.json'); workflow=read(out/'workflow/comparison.json')
    if not workflow['complete']: raise ValueError('fresh comparison is not complete')
    lines=['# V32 价值训练与三方案对比结果','',
        f"有效样本 {audit['samples']}，正样本 {audit['positives']}，负样本 {audit['negatives']}，布局 {audit['layouts']}。全量审计通过。",'',
        '| 集合 | 布局 | 样本 | 正例 | 负例 |','|---|---:|---:|---:|---:|']
    for split in ('train','validation','test'):
        s=audit['by_split'][split];lines.append(f"| {split} | {s['layouts']} | {s['samples']} | {s['positives']} | {s['negatives']} |")
    lines+=['',f"模型：303维原子节点输入、64维隐藏层、两层关系消息传递、三路图池化，{training['model_parameters']}个参数。从头训练；只用验证集选择初始化{training['selected_initialization']}、epoch {training['selected_epoch']}和温度{training['temperature']:.4f}。",'',
        '## 冻结测试池的离线候选筛选','',
        '| K | 价值命中率 | 随机命中率精确期望 | 全池可解率 | 价值调用数 | 随机调用数期望 |',
        '|---:|---:|---:|---:|---:|---:|']
    for k,r in offline['by_k'].items():
        lines.append(f"| {k} | {r['value_success']['mean']:.1%} | {r['random_expected_success']['mean']:.1%} | {r['full_success']['mean']:.1%} | {r['value_calls']:.3f} | {r['random_expected_calls']:.3f} |")
    c=offline['classification']
    lines+=['',f"测试候选准确率 {c['accuracy']:.3f}，ROC-AUC {c['roc_auc']:.3f}，AP {c['average_precision']:.3f}，Brier {c['brier']:.3f}，同布局成对准确率 {c['pair_accuracy']:.3f}。",'',
        '## 新鲜完整流程重跑','',
        '每个方案均使用相同的13个测试布局和冻结候选，验证trial=0；选择通过者后从新鲜初态执行trial=1。整体成功率以最终独立执行验收为准。', '',
        '| 方案 | 最终成功/13 | 成功率 | 平均孪生调用 | 平均验证秒 | 平均最终执行秒 | 平均流程秒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in workflow['metrics'].items():
        t=r['seconds'];lines.append(f"| {name} | {r['successes']}/13 | {r['workflow_success']['mean']:.1%} | {r['mean_twin_calls']:.3f} | {t['validation_seconds']['mean']:.2f} | {t['deployment_seconds']['mean']:.2f} | {t['workflow_wall_seconds']['mean']:.2f} |")
    lines+=['','流程时间从加载冻结候选开始，包含价值编码/推理（如适用）、串行孪生验证及最终执行。未计新一次大模型计划生成；共同历史规划时间和模型冷加载另存于逐布局记录。最多4个布局并行，13个测试布局的区间较宽，不能当作真实机器人或新任务泛化结论。', '',
        '价值与各基线的成功率及耗时差异、以布局为单位的配对bootstrap区间见 workflow/comparison.json。采集耗时求和仅作为离线成本估计。若价值模型没有稳定优于随机，应如实报告，不用测试结果继续调参。', '',
        '原184条与修复后816条候选生成版本不同，物理控制器与验收一致；规划模型配置也有来源差异。全部记录和全负布局均保留。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(source,out):
    out.mkdir(parents=True,exist_ok=True)
    archive=out/'source';archive.mkdir(exist_ok=True)
    for file in ('run_value_v32_pipeline.py','train_value_v32.py','evaluate_workflow_v32.py','export_dataset_v32.py','audit_dataset_v32.py'):
        original=Path(__file__).with_name(file);target=archive/file
        if target.exists() and sha(target)!=sha(original): raise ValueError(f'pipeline source changed: {file}')
        if not target.exists(): shutil.copy2(original,target)
    while True:
        summary=read(source/'summary.json')
        if summary.get('halt_reason'): raise ValueError(f"collection halted: {summary['halt_reason']}")
        if summary.get('valid')==1000 and summary.get('completed_layouts')==125: break
        status(out,'waiting_for_1000',complete_layout_samples=summary.get('valid'));time.sleep(30)
    if not (out/'dataset_audit.json').exists():
        status(out,'exporting_and_auditing')
        from scripts.export_dataset_v32 import export
        from scripts.audit_dataset_v32 import audit
        export(source,out/'data');checked=audit(source,out/'data');write(out/'dataset_audit.json',checked)
    checked=read(out/'dataset_audit.json')
    if not checked.get('dataset_complete') or not checked.get('passed'): raise ValueError('full dataset audit failed')
    status(out,'training')
    with (out/'training.log').open('a') as log:
        subprocess.run([sys.executable,'-u','-m','scripts.train_value_v32','--source',str(source),'--exported',str(out/'data'),
            '--audit',str(out/'dataset_audit.json'),'--out',str(out/'training')],stdout=log,stderr=subprocess.STDOUT,check=True)
    status(out,'fresh_workflow_evaluation')
    with (out/'workflow.log').open('a') as log:
        subprocess.run([sys.executable,'-u','-m','scripts.evaluate_workflow_v32','--source',str(source),
            '--checkpoint',str(out/'training/value_v32.pt'),'--out',str(out/'workflow'),'--workers','4'],stdout=log,stderr=subprocess.STDOUT,check=True)
    render_report(out);status(out,'complete',report=str(out/'REPORT_ZH.md'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (a.out/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: run(a.source,a.out)
        except Exception as exc:
            status(a.out,'failed',error=f'{type(exc).__name__}: {exc}');raise
