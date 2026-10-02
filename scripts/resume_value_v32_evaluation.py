"""Resume the frozen V32 experiment after the result serialization repair."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
from scripts.train_value_v32 import read, write, sha
from scripts.run_value_v32_pipeline import render_report, status


def resume(source,out):
    training=out/'training'
    frozen=read(training/'selection_frozen.json')
    if sha(training/'value_v32.pt')!=frozen['checkpoint_sha256']:
        raise ValueError('checkpoint changed after selection freeze')
    audit=read(out/'dataset_audit.json')
    if not audit.get('passed') or not audit.get('dataset_complete'):
        raise ValueError('complete dataset audit required')
    for name in ('train_value_v32.py','export_dataset_v32.py','audit_dataset_v32.py'):
        if sha(out/'source'/name)!=sha(Path(__file__).with_name(name)):
            raise ValueError(f'frozen training/data code changed: {name}')
    protected={str(p.relative_to(out)):sha(p) for p in training.rglob('*') if p.is_file()}
    repair=out/'evaluation_repairs'/'numpy_serialization_20260927'
    if not repair.exists():
        errors=list((out/'workflow').glob('seed_*/error.json'))
        if len(errors)!=13 or any(read(p)['error']!='TypeError: Object of type ndarray is not JSON serializable' for p in errors):
            raise ValueError('unexpected failure requires separate diagnosis')
        if list((out/'workflow').glob('seed_*/*/summary.json')):
            raise ValueError('completed policies found; do not repeat them')
        repair.mkdir(parents=True)
        shutil.move(str(out/'workflow'),str(repair/'interrupted_workflow'))
        for name in ('status.json','workflow.log'):
            if (out/name).exists(): shutil.copy2(out/name,repair/name)
        for name in ('evaluate_workflow_v32.py','resume_value_v32_evaluation.py'):
            shutil.copy2(Path(__file__).with_name(name),repair/name)
        write(repair/'repair.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
            cause='Nested NumPy arrays in physical traces were not JSON serializable.',
            change='Convert result through existing plan.plain before JSON writing; no physical runtime or model changes.',
            interrupted_attempts_excluded_from_fresh_timing=True,protected_training_sha256=protected,
            original_evaluator_sha256=sha(out/'source/evaluate_workflow_v32.py'),
            repaired_evaluator_sha256=sha(Path(__file__).with_name('evaluate_workflow_v32.py'))))
    else:
        evidence=read(repair/'repair.json')
        if evidence['protected_training_sha256']!=protected:
            raise ValueError('training artifacts changed since evaluation repair')
        if evidence['repaired_evaluator_sha256']!=sha(Path(__file__).with_name('evaluate_workflow_v32.py')):
            raise ValueError('evaluator changed since recorded repair')
    status(out,'fresh_workflow_evaluation',repair=str(repair))
    with (out/'workflow_repaired.log').open('a') as log:
        subprocess.run([sys.executable,'-u','-m','scripts.evaluate_workflow_v32','--source',str(source),
            '--checkpoint',str(training/'value_v32.pt'),'--out',str(out/'workflow'),'--workers','4'],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    if any(sha(out/name)!=expected for name,expected in protected.items()):
        raise ValueError('training artifacts changed during evaluation')
    render_report(out)
    with (out/'REPORT_ZH.md').open('a',encoding='utf-8') as report:
        report.write('\n结果保存曾因 NumPy 数组序列化中断；中断尝试保留于 evaluation_repairs/，不计入新鲜流程计时。修复仅转换保存格式，冻结模型、候选与物理运行时未改。\n')
    status(out,'complete',report=str(out/'REPORT_ZH.md'),repair=str(repair))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    import fcntl
    with (args.out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: resume(args.source,args.out)
        except Exception as exc:
            status(args.out,'failed',error=f'{type(exc).__name__}: {exc}')
            raise
