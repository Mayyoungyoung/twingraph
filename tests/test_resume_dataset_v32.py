import json
from pathlib import Path
import pytest
from scripts.resume_dataset_v32 import verified_existing, summarize, run


def make_layout(tmp_path, count=8, invalid=False):
    entries=[dict(name=f'llm_{i:03d}',plan_sha256=f'plan{i}') for i in range(8)]
    (tmp_path/'request.json').write_text(json.dumps(dict(runtime_sha256='old',candidates=entries)))
    for i,entry in enumerate(entries[:count]):
        folder=tmp_path/'candidates'/entry['name']; folder.mkdir(parents=True)
        (folder/'result.json').write_text(json.dumps(dict(valid=not(invalid and i==0),success=i%2==0,
            runtime_sha256='old',plan_sha256=entry['plan_sha256'],timeout=False)))


def test_complete_layout_preserves_original_labels_and_runtime(tmp_path):
    make_layout(tmp_path)
    row=verified_existing(tmp_path,3300,{'old','new'})
    assert row['valid']==8 and row['positives']==4 and row['runtime_sha256']=='old'


def test_partial_layout_is_not_counted_as_complete(tmp_path):
    make_layout(tmp_path,count=7)
    assert verified_existing(tmp_path,3300,{'old'}) is None


def test_invalid_existing_sample_cannot_be_silently_retried(tmp_path):
    make_layout(tmp_path,invalid=True)
    with pytest.raises(ValueError,match='invalid'): verified_existing(tmp_path,3300,{'old'})


def test_unregistered_runtime_cannot_enter_resume(tmp_path):
    make_layout(tmp_path)
    with pytest.raises(ValueError,match='unregistered'): verified_existing(tmp_path,3300,{'new'})


def test_summary_does_not_count_pre_execution_errors_as_negatives():
    report=summarize([dict(seed=1,attempted=8,valid=8,positives=2,negatives=6),
        dict(seed=2,error='geometry failed')],'fixed',1000,'invalid_or_censored_sample')
    assert report['valid']==8 and report['negatives']==6 and report['completed_layouts']==1
    assert report['ready_for_collection'] is False
