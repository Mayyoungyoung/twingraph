"""Audit saved V32 workflow evidence and extend its report; never execute/train."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def finalize(source,out):
    dataset=read(out/'dataset_audit.json'); manifest=read(source/'manifest.json')
    comparison=read(out/'workflow/comparison.json'); training=read(out/'training/training_report.json')
    frozen=read(out/'training/selection_frozen.json'); offline=read(out/'training/offline_test.json')
    predictions=read(out/'training/test_predictions.json')
    assert dataset['passed'] and dataset['dataset_complete'] and dataset['samples']==1000
    assert comparison['complete'] and comparison['completed_layouts']==13
    assert sha(out/'training/value_v32.pt')==frozen['checkpoint_sha256']==training['checkpoint_sha256']
    protected=read(out/'evaluation_repairs/numpy_serialization_20260927/repair.json')['protected_training_sha256']
    assert all(sha(out/name)==expected for name,expected in protected.items())
    test_seeds={j['seed'] for j in manifest['jobs'] if j['split']=='test'}
    assert len(test_seeds)==13 and {r['seed'] for r in comparison['rows']}==test_seeds
    assert not test_seeds.intersection(training['train_seeds'])
    assert not test_seeds.intersection(training['validation_seeds'])
    required=('assembly_pass','base_hole_engagement_pass','fixture_capture_pass',
              'final_seat_pass','final_release_and_retraction_pass')
    policies=('value_top4','random_top4','full8')
    counts=dict(validation=0,deployment=0); mismatches=[]; failures=[]; layout_rows=[]
    hashes={}; history=[]
    for row in sorted(comparison['rows'],key=lambda r:r['seed']):
        seed=row['seed']; root=out/'workflow'/f'seed_{seed}'
        request=read(source/f'seed_{seed}'/'request.json'); saved_manifest=read(root/'manifest.json')
        assert saved_manifest['request_sha256']==digest(request)
        assert saved_manifest['checkpoint_sha256']==frozen['checkpoint_sha256']
        assert saved_manifest['random_order']==np.random.default_rng(320927+seed).permutation(8).tolist()
        assert saved_manifest['policy_order']==np.random.default_rng(321027+seed).permutation(policies).tolist()
        history.append(dict(seed=seed,seconds=saved_manifest['original_plan_generation_seconds']))
        brief=dict(seed=seed)
        for policy in policies:
            folder=root/policy; report=read(folder/'summary.json'); selection=read(folder/'selection.json')
            assert report==row['policies'][policy] and report['valid']
            order=selection['order']; expected_budget=8 if policy=='full8' else 4
            assert selection['budget']==expected_budget and not selection['outcome_labels_read_for_ranking']
            if policy=='full8': assert order==list(range(8))
            elif policy=='random_top4': assert order==saved_manifest['random_order']
            else:
                assert order==np.argsort(-np.asarray(selection['probabilities']),kind='stable').tolist()
                expected={r['id'].split('_',2)[2]:r['probability'] for r in predictions if r['seed']==seed}
                assert all(abs(selection['probabilities'][i]-expected[e['name']])<1e-5 for i,e in enumerate(request['candidates']))
            validations=report['validations']; assert len(validations)==report['twin_calls']
            assert [v['name'] for v in validations]==[request['candidates'][i]['name'] for i in order[:len(validations)]]
            hits=[v for v in validations if v['success']]
            assert report['selected']==(hits[0]['name'] if hits else None)
            assert report['verification_hit']==bool(hits)
            if policy=='full8': assert len(validations)==8
            elif hits: assert validations[-1]['success'] and len(hits)==1
            else: assert len(validations)==4
            trials=[('validation',v,0) for v in validations]
            if report['deployment']: trials.append(('deployment',report['deployment'],1))
            assert bool(report['deployment'])==bool(hits)
            for kind,v,repeat in trials:
                p=folder/kind/v['name']/'result.json'; result=read(p)
                entry=next(e for e in request['candidates'] if e['name']==v['name'])
                assert digest(entry['complete_candidate_plan_ir'])==entry['plan_sha256']==result['source_plan_sha256']==v['plan_sha256']
                assert result['source_runtime_sha256']==request['runtime_sha256']
                assert result['runtime_sha256']==saved_manifest['execution_runtime_sha256']
                assert result['valid'] and not result.get('timeout') and not result.get('software_exception')
                assert result['repeat']==result['trial']['repeat']==repeat
                rng=np.random.default_rng(np.random.SeedSequence([seed,repeat,1753]))
                assert request['domain']=='online'
                assert result['trial']==dict(domain='online',repeat=repeat,friction_scale=float(rng.uniform(.85,1.15)),actuator_gain_scale=float(rng.uniform(.985,1.015)))
                assert result['success']==v['success']==all(result['stage_passes'].get(k) is True for k in required)
                assert result['fresh_total_wall_seconds']==v['seconds'] and v['seconds']>0
                counts[kind]+=1; hashes[str(p.relative_to(out))]=sha(p)
                if kind=='validation':
                    original=read(source/f'seed_{seed}'/'candidates'/v['name']/'result.json')
                    if original['success']!=result['success']: mismatches.append(dict(seed=seed,policy=policy,name=v['name']))
                elif not result['success']:
                    failures.append(dict(seed=seed,policy=policy,name=v['name'],error=result.get('error'),stage_passes=result['stage_passes']))
            assert report['workflow_success']==bool(report['deployment'] and report['deployment']['success'])
            assert report['workflow_wall_seconds']>=sum(report[k] for k in ('plan_load_seconds','encoding_seconds','inference_seconds','validation_seconds','deployment_seconds'))
            brief[policy]=dict(selected=report['selected'],success=report['workflow_success'],calls=report['twin_calls'],seconds=report['workflow_wall_seconds'])
        layout_rows.append(brief)
    for policy in policies:
        rows=[r['policies'][policy] for r in comparison['rows']]; metric=comparison['metrics'][policy]
        assert sum(r['workflow_success'] for r in rows)==metric['successes']
        assert np.isclose(np.mean([r['twin_calls'] for r in rows]),metric['mean_twin_calls'])
        for key,value in metric['seconds'].items(): assert np.isclose(np.mean([r[key] for r in rows]),value['mean'])
    audit=dict(passed=True,layouts=13,policy_runs=39,trials=counts,checkpoint_sha256=frozen['checkpoint_sha256'],
        protected_training_files=len(protected),fresh_validation_vs_collection_mismatches=mismatches,
        deployment_failures=failures,planning_history=history,result_sha256=hashes,
        scope='Saved evidence consistency: fixed test split, checkpoint freeze, selections, plans, perturbations, valid labels, success/cost aggregates; not proof of hardware generalization.')
    (out/'workflow_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    report=out/'REPORT_ZH.md'; text=report.read_text(encoding='utf-8')
    marker='\n## 完整证据核验与结果解读\n'
    text=text.split(marker)[0]
    lines=[marker,'',
        f"保存证据复核通过：13布局、39方案流程、{counts['validation']}次孪生验证、{counts['deployment']}次独立扰动执行，无无效样本。模型及训练产物哈希保持冻结；与原采集同候选、同repeat=0的标签不一致记录有{len(mismatches)}条。详见 workflow_audit.json。",'',
        '### 输入、输出与训练选择','',
        '输入为执行前初始观测与完整冻结原子计划，构成107或116节点、每节点303维、7类关系的有向图；不含执行结果、耗时、布局编号或候选编号。输出单个完整装配成功概率。303→64嵌入、两层关系消息传递、注意力/均值/最大值池化、192→64→1输出，165570参数。', '',
        '训练只用800条train；同布局BCE加0.5倍正负成对排序损失，保留全负布局；AdamW学习率0.001、weight decay 0.03。初始化7/17/29仅由96条validation选择，温度也只用validation拟合。104条test未参与选择。当前模型预测完整装配验收通过，不包含初始擦拭和最终滑动测试。', '',
        '### 候选分类与排序','',
        '| 指标 | 测试值 |','|---|---:|']
    for key,value in offline['classification'].items(): lines.append(f'| {key} | {value:.6f} |')
    lines+=['','### 实测时间拆解（每布局均值，秒）','','| 方案 | 计划加载 | 图编码 | 推理 | 冷加载（单列） | 孪生验证 | 独立执行 | 流程墙钟 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for policy,metric in comparison['metrics'].items():
        seconds=metric['seconds']
        lines.append('| '+policy+' | '+' | '.join(f"{seconds[k]['mean']:.4f}" for k in ('plan_load_seconds','encoding_seconds','inference_seconds','model_load_seconds','validation_seconds','deployment_seconds','workflow_wall_seconds'))+' |')
    lines+=['','冷加载不在流程墙钟内；本次三个方案均重用冻结计划，没有新LLM调用。全部验证与独立执行都是数字孪生仿真，独立执行使用repeat=1的新摩擦/增益扰动，不是实体机器人实验。最多4布局共享主机并行，方案内部串行；Full8按协议全部运行8条后再选首个通过者，未测并行Full8或提前停止的Full8。', '',
        '### 离线耗时估计（每布局秒，仅孪生验证）','','这些数值来自并行采集时各候选墙钟的加权或求和，不能作为上表的新鲜流程耗时。','','| K | 价值 | 随机精确期望 | Full8 |','|---:|---:|---:|---:|']
    for k,r in offline['by_k'].items(): lines.append(f"| {k} | {r['value_twin_seconds_estimate']:.2f} | {r['random_twin_seconds_estimate']:.2f} | {r['full_twin_seconds_estimate']:.2f} |")
    lines+=['','### 逐布局整体结果','','| seed | 价值 成功/调用/秒 | 随机 成功/调用/秒 | Full8 成功/调用/秒 |','|---:|---|---|---|']
    for row in layout_rows: lines.append('| '+str(row['seed'])+' | '+' | '.join(f"{int(row[p]['success'])} / {row[p]['calls']} / {row[p]['seconds']:.1f}" for p in policies)+' |')
    recorded=[h['seconds'] for h in history if h['seconds'] is not None]
    lines+=['',f"原候选生成阶段历史墙钟在{len(recorded)}/13布局有记录，均值{np.mean(recorded):.2f}秒；包含原流程等待/调用开销，不是本次重新测量的LLM推理时长，也未加入本次流程墙钟。",'',
        '独立扰动执行失败：价值方案3302/3390为推物后接触进度不足、3317为双侧抓取接触缺失；随机与Full8均在3317抓取接触缺失、3372放置支撑接触缺失。3316原候选池全负，三方案均没有可执行通过者，保留在总体分母。详见 workflow_audit.json 的逐次失败证据。']
    lines+=['','### 解释与不确定性','',
        '离线K=4候选命中率：价值92.3%，随机无放回精确期望78.9%。实测固定随机排列恰好也在12/13布局找到通过者，三方案验证命中均为92.3%；最终独立扰动执行后价值9/13、随机10/13、Full8 10/13。因此本轮证明了筛选效率改善，但没有证明整体成功率提高。离线随机期望与一次随机实测的统计对象不同。', '',
        f"价值流程均时较随机降低{1-comparison['metrics']['value_top4']['seconds']['workflow_wall_seconds']['mean']/comparison['metrics']['random_top4']['seconds']['workflow_wall_seconds']['mean']:.1%}，较Full8降低{1-comparison['metrics']['value_top4']['seconds']['workflow_wall_seconds']['mean']/comparison['metrics']['full8']['seconds']['workflow_wall_seconds']['mean']:.1%}。以下为布局配对bootstrap 95%区间，不是对未来任务的保证。",'',
        '| 对比：价值减基线 | 成功率差（百分点）及95%区间 | 流程秒差及95%区间 |','|---|---|---|']
    for baseline,r in comparison['paired_value_minus_baseline'].items():
        s=r['success_difference'];t=r['seconds_difference'];sc=s['layout_bootstrap_95'];tc=t['layout_bootstrap_95']
        lines.append(f"| {baseline} | {s['mean']*100:.2f} [{sc[0]*100:.2f}, {sc[1]*100:.2f}] | {t['mean']:.2f} [{tc[0]:.2f}, {tc[1]:.2f}] |")
    lines+=['','测试只有13布局；与随机的成功率差和耗时差区间均跨0，不据此声称稳定优于随机。预测训练标签为repeat=0的完整成功，独立扰动鲁棒性未被直接监督；这是下一轮值得检验的方向，而非本轮已经证明的失败原因。不根据当前test调整模型，也不重分数据。', '',
        '来源：原184条与修复后816条的候选几何版本已登记，物理控制器、验收与编码器不变；规划来源为gpt-6-astra/high与gpt-6-sol/low，保留原始请求和日志。这些来源差异及仿真状态位姿、无附加感知噪声限制了泛化解释。', '',
        '完整协议：TRAINING_PROTOCOL_20260927.md；数据审计：dataset_audit.json；模型：training/value_v32.pt；选择冻结：training/selection_frozen.json；离线测试：training/offline_test.json；新鲜实测：workflow/comparison.json；结果复核：workflow_audit.json。']
    report.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
    return {k:v for k,v in audit.items() if k not in ('result_sha256','deployment_failures','planning_history')}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(finalize(args.source,args.out)))
