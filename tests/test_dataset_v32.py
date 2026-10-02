"""Readiness must use complete physical outcomes, including censored failures."""
from concurrent.futures import Future
import json
import sys
import pytest
from scripts import collect_dataset_v32 as collection


class ImmediatePool:
    def __init__(self, **kwargs):pass
    def __enter__(self):return self
    def __exit__(self, *args):pass
    def submit(self, fn, arg):
        f=Future();f.set_result(fn(arg));return f


@pytest.mark.parametrize('counts,invalid,ready,halt',[
    ([1,1,1],False,True,None),
    ([3,0,0],False,False,None),
    ([0,0,0],False,False,'all_negative_batch'),
    ([1,1,1],True,False,'invalid_or_censored_sample'),
])
def test_gate_requires_mixed_layouts_and_uncensored_trials(tmp_path,monkeypatch,counts,invalid,ready,halt):
    def worker(job):
        seed,out=job;valid=7 if invalid and seed==100 else 8
        return dict(seed=seed,out=out,attempted=8,valid=valid,
                    positives=counts[seed-100],negatives=valid-counts[seed-100])
    monkeypatch.setattr(collection,'worker',worker)
    monkeypatch.setattr(collection,'ProcessPoolExecutor',ImmediatePool)
    monkeypatch.setattr(collection,'fingerprint',lambda:dict(sha256='test-frozen-runtime'))
    monkeypatch.setattr(sys,'argv',['collect','--out',str(tmp_path/'data'),'--seeds','100','101','102'])
    result=collection.main()
    report=json.loads((tmp_path/'data/summary.json').read_text())
    assert report['ready_for_collection'] is ready
    assert report['halt_reason']==halt
    assert result==(2 if halt else 0)
    manifest=json.loads((tmp_path/'data/manifest.json').read_text())
    assert manifest['planned_samples']==24
    assert manifest['outcome_based_resampling'] is False
    assert len({job['seed'] for job in manifest['jobs']})==3
