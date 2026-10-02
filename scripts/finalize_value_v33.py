"""Verify all completed V33 raw workflows and produce the final Chinese report."""
from pathlib import Path
import numpy as np
from scripts.train_value_v32 import read,write,sha,binary_metrics
from scripts.evaluate_workflow_v33 import POLICIES,SCENARIOS,summarize
from scripts.collect_robust_value_v33 import RUNTIME
from scripts.collect_sliding_assembly_v23 import REQUIRED
from simbench.value.plan import digest
from simbench.value.physical import perturbation


def finalize(source,out):
    comparison=read(out/'workflow/comparison.json');frozen=read(out/'training/selection_frozen.json')
    if not comparison['complete'] or len(comparison['rows'])!=50:raise ValueError('450 fresh policy workflows required')
    if any(sha(out/'training'/name)!=h for name,h in frozen['checkpoint_sha256'].items()):raise ValueError('frozen model changed')
    train=read(out/'training/training_report.json')
    if set(range(4000,4050))&(set(train['train_layouts'])|set(train['validation_layouts'])):raise ValueError('test layout leakage')
    hashes={};counts=dict(validation=0,deployment=0);labels={};failures=[];candidate_labels=[];candidate_prob=[];pairs_correct=pairs_total=0.
    for row in comparison['rows']:
        seed=row['seed'];request=read(source/f'seed_{seed}'/'request.json')
        directory=out/'workflow'/f'seed_{seed}';manifest=read(directory/'manifest.json')
        assert digest(request)==manifest['request_sha256'] and len(row['scenarios'])==3
        assert manifest['model_freeze_sha256']==sha(out/'training/selection_frozen.json')
        for scenario in row['scenarios']:
            index=scenario['scenario'];vr,dr=SCENARIOS[index]
            for policy in POLICIES:
                folder=directory/f'scenario_{index}'/policy
                report=read(folder/'summary.json');selection=read(folder/'selection.json')
                assert report==scenario['policies'][policy] and report['valid']
                assert report['verify_repeat']==vr and report['deploy_repeat']==dr
                assert not selection['outcome_labels_read'] and selection['budget']==(8 if policy=='full8' else 3)
                if policy=='random_top3':assert selection['order']==np.random.default_rng(330927+seed*7+index).permutation(8).tolist()
                elif policy=='full8':assert selection['order']==list(range(8))
                else:
                    mean=np.asarray(selection['predictions']);unc=np.asarray(selection['uncertainty']);cost=mean[:,2]/max(float(np.median(mean[:,2])),1.)
                    scores=np.log(np.maximum(mean[:,1],1e-6))+.25*np.log(np.maximum(mean[:,0],1e-6))-frozen['alpha']*np.log(np.maximum(cost,.01))-frozen['beta']*unc
                    np.testing.assert_allclose(scores,selection['scores'],rtol=1e-6,atol=1e-7)
                    assert selection['order']==np.argsort(-scores,kind='stable').tolist()
                validations=report['validations'];assert len(validations)==report['twin_calls']
                assert [r['name'] for r in validations]==[request['candidates'][i]['name'] for i in selection['order'][:len(validations)]]
                hits=[r for r in validations if r['success']]
                if policy=='full8':assert len(validations)==8
                elif hits:assert validations[-1]['success'] and len(hits)==1 and len(validations)<=3
                else:assert len(validations)==3
                assert report['selected']==(hits[0]['name'] if hits else None)
                assert bool(report['deployment'])==bool(hits)==report['verification_hit']
                trials=[('validation',r,vr) for r in validations]
                if report['deployment']:trials.append(('deployment',report['deployment'],dr))
                for kind,r,repeat in trials:
                    path=folder/kind/r['name']/'result.json';raw=read(path)
                    entry=next(e for e in request['candidates'] if e['name']==r['name'])
                    assert raw['valid'] and not raw.get('timeout') and not raw.get('software_exception')
                    assert raw['source_plan_sha256']==r['plan_sha256']==entry['plan_sha256']==digest(entry['complete_candidate_plan_ir'])
                    assert raw['runtime_sha256']==RUNTIME==raw['source_runtime_sha256']
                    assert raw['trial']==perturbation(seed,repeat,'online') and raw['repeat']==repeat
                    assert raw['success']==r['success']==all(raw['stage_passes'].get(k) is True for k in REQUIRED)
                    assert raw['fresh_total_wall_seconds']==r['seconds']
                    tag=(seed,r['name'],repeat)
                    if tag in labels:assert labels[tag]==raw['success'],'non-deterministic paired trial requires investigation'
                    labels[tag]=raw['success'];counts[kind]+=1;hashes[str(path.relative_to(out))]=sha(path)
                    if kind=='deployment' and not raw['success']:failures.append(dict(seed=seed,scenario=index,policy=policy,name=r['name'],error=raw.get('error')))
                assert report['workflow_success']==bool(report['deployment'] and report['deployment']['success'])
            prediction=np.asarray(read(directory/f'scenario_{index}'/'value_top3/selection.json')['predictions'])[:,1]
            y=np.asarray([r['success'] for r in scenario['policies']['full8']['validations']],bool)
            candidate_labels.extend(y.tolist());candidate_prob.extend(prediction.tolist())
            pos=prediction[y];neg=prediction[~y]
            pairs_correct+=float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).sum());pairs_total+=len(pos)*len(neg)
    recalculated=summarize(comparison['rows'])
    assert recalculated==comparison
    audit=dict(passed=True,layouts=50,scenarios_per_layout=3,policy_workflows=450,trials=counts,
        checkpoint_sha256=frozen['checkpoint_sha256'],result_sha256=hashes,deployment_failures=failures,
        repeated_paired_labels_consistent=True,test_used_for_training=False)
    classification=dict(**binary_metrics(candidate_labels,candidate_prob),pair_accuracy=pairs_correct/pairs_total if pairs_total else None,
        evaluated_candidate_trials=len(candidate_labels),scope='Robust probability versus all Full8 verification outcomes across 3 independent scenarios, distinct from selected-plan deployment success.')
    write(out/'candidate_test_metrics.json',classification)
    write(out/'workflow_audit.json',audit)
    metrics=comparison['metrics'];value=metrics['value_top3']
    lines=['# V33 鲁棒价值：Top3三方案完整流程对比','',
        '主任务使用与V32一致的物理控制器和验收。本报告如实保留所有成功与失败；目标是价值方案取得优势，实际是否达到目标由以下冻结新测试结果决定。','',
        '## 数据与模型','',
        '原1000条数据/125布局的划分保留。只对100个train与12个validation布局的全部候选各补充3次扰动，共2688次新执行。原13个test布局未参与新训练；最终使用4000—4049这50个新布局，每布局8条共同候选。', '',
        '输入为执行前原子技能图，加冻结候选几何、控制参数和条件相对位置。输出名义成功、扰动成功、验证步数成本；初始化、epoch、集成、温度及排序权重只由训练/验证数据选择。', '',
        f"最终模型：{', '.join(frozen['paths'])}；成本权重{frozen['alpha']}，不确定性权重{frozen['beta']}。", '',
        '### 验证集消融（不是最终测试结果）','',
        '| 方案 | 平均独立扰动成功率 | 平均验证次数 | 鲁棒Brier |','|---|---:|---:|---:|']
    for name,m in train['ablations'].items():lines.append(f"| {name} | {m['workflow_success']:.1%} | {m['calls']:.3f} | {m['robust_brier']:.4f} |")
    lines+=['','新测试候选指标基于Full8的1200次验证结果：'+', '.join(f'{k}={classification[k]:.4f}' for k in ('accuracy','balanced_accuracy','roc_auc','average_precision','brier','nll','pair_accuracy'))+'。这是候选级指标，与最终独立部署成功率分开报告。']
    lines+=['','## 新测试的实测全流程','',
        '每布局3组验证/独立部署扰动，三个方案在相同候选和相同扰动下配对。价值、随机最多验证3条，首个通过即停；Full8全部验证8条后选择原顺序首个通过者。无通过者计整体失败。', '',
        '| 方案 | 最终成功/150 | 成功率及布局bootstrap 95%区间 | 平均验证次数 | 平均流程秒 |','|---|---:|---|---:|---:|']
    for policy,m in metrics.items():
        ci=m['workflow_success']['layout_bootstrap_95']
        lines.append(f"| {policy} | {m['successes']}/150 | {m['workflow_success']['mean']:.1%} [{ci[0]:.1%}, {ci[1]:.1%}] | {m['mean_twin_calls']:.3f} | {m['seconds']['workflow_wall_seconds']['mean']:.2f} |")
    lines+=['','### 时间拆解（每场景均值，秒）','','| 方案 | 加载计划 | 编码 | 推理 | 冷加载单列 | 孪生验证 | 独立部署 | 完整流程 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for policy,m in metrics.items():lines.append('| '+policy+' | '+' | '.join(f"{m['seconds'][k]['mean']:.3f}" for k in ('plan_load_seconds','encoding_seconds','inference_seconds','model_load_seconds','validation_seconds','deployment_seconds','workflow_wall_seconds'))+' |')
    lines+=['','上述为新鲜实测墙钟，包含失败，不是并行采集耗时求和。最多4布局并行，各流程串行。新鲜流程重放已冻结候选；计划生成历史开销和模型冷加载单列，不虚构本轮LLM时间。全部为仿真，未执行真实机器人。', '',
        '### 配对差异：价值减基线','',
        '| 基线 | 成功率差及95%区间 | 流程秒差及95%区间 |','|---|---|---|']
    for baseline,r in comparison['paired_value_minus_baseline'].items():
        s=r['workflow_success'];t=r['workflow_wall_seconds'];sc=s['layout_bootstrap_95'];tc=t['layout_bootstrap_95']
        lines.append(f"| {baseline} | {s['mean']:.1%} [{sc[0]:.1%}, {sc[1]:.1%}] | {t['mean']:.2f} [{tc[0]:.2f}, {tc[1]:.2f}] |")
    dominates=all(value['workflow_success']['mean']>=metrics[b]['workflow_success']['mean'] and value['seconds']['workflow_wall_seconds']['mean']<metrics[b]['seconds']['workflow_wall_seconds']['mean'] for b in ('random_top3','full8'))
    lines+=['','## 结论','',
        ('本轮价值方案在观测到的平均成功率不低于两基线的同时，平均流程时间更短；统计不确定性仍以以上区间为准。' if dominates else '本轮尚未实现价值方案在成功率和耗时上同时领先两个基线；保留真实结果，不根据测试结果更换候选、模型、阈值或随机种子。'),'',
        f"原始结果审计通过：{counts['validation']}次验证、{counts['deployment']}次独立部署。完整记录见workflow/，标签/哈希审计见workflow_audit.json，模型选择见training/selection_frozen.json。"]
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {k:v for k,v in audit.items() if k not in ('result_sha256','deployment_failures')}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(finalize(a.source,a.out))
