import json

from scripts.analyze_v13_mechanism_matrix import analyze
from scripts.run_v13_mechanism_matrix import SCHEMA,file_sha256


def _case(root,seed,labels):
    directory=root/f"seed_{seed}"/"carriage";directory.mkdir(parents=True)
    pool=[dict(name=name) for name in labels]
    request=dict(schema=SCHEMA,seed=seed,n=len(pool),stop_after="carriage",pool=pool,
                 runtime_sha256="a"*64)
    trials=[dict(name=name,success=value) for name,value in labels.items()]
    summary=dict(schema=SCHEMA,seed=seed,stop_after="carriage",complete=True,
                 runtime_sha256="a"*64,trials=trials)
    (directory/"request.json").write_text(json.dumps(request))
    (directory/"summary.json").write_text(json.dumps(summary))


def test_crossover_gate_requires_coverage_mixed_pools_and_reversal(tmp_path):
    _case(tmp_path,1,{"grounded_000":True,"grounded_001":False,"grounded_002":True})
    _case(tmp_path,2,{"grounded_000":False,"grounded_001":True,"grounded_002":True})
    report=analyze([tmp_path],tmp_path/"analysis")
    assert report["crossover_gate_pass"] is True
    assert report["pairwise_crossover_count"]==1


def test_no_good_plan_fails_gate(tmp_path):
    _case(tmp_path,1,{"grounded_000":False,"grounded_001":False})
    _case(tmp_path,2,{"grounded_000":True,"grounded_001":False})
    report=analyze([tmp_path],tmp_path/"analysis")
    assert report["crossover_gate_pass"] is False


def test_result_fingerprint_hashes_exact_nonstandard_json_bytes(tmp_path):
    path=tmp_path/"result.json"
    path.write_text('{"diagnostic": Infinity}\n',encoding="utf-8")
    assert file_sha256(path)==__import__("hashlib").sha256(path.read_bytes()).hexdigest()
