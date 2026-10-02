from itertools import permutations
import numpy as np
import pytest
from scripts.train_value_v32 import binary_metrics,random_expected,ordered,rank_metrics


@pytest.mark.parametrize('labels',[[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,1]])
@pytest.mark.parametrize('k',[1,2,4])
def test_random_expectations_match_exhaustive_permutations(labels,k):
    rows=[dict(id=str(i),label=y,seconds=float((i+1)**2)) for i,y in enumerate(labels)]
    actual=random_expected(rows,k)
    trials=[ordered(rows,p,k) for p in permutations(range(4))]
    assert actual['success_rate']==pytest.approx(np.mean([r['success'] for r in trials]))
    assert actual['calls']==pytest.approx(np.mean([r['calls'] for r in trials]))
    assert actual['seconds']==pytest.approx(np.mean([r['seconds'] for r in trials]))


def test_probability_ties_are_not_credited_as_perfect_ranking():
    report=binary_metrics([0,1,0,1],[.5,.5,.5,.5])
    assert report['roc_auc']==.5 and report['average_precision']==.5


def test_layout_ranking_keeps_unsolvable_layout_in_denominator():
    rows=[dict(id=f'{s}_{i}',seed=s,label=int(s==1 and i==0)) for s in (1,2) for i in range(4)]
    report=rank_metrics(rows,[.9,.3,.2,.1,.9,.3,.2,.1],1)
    assert report['hit']==.5 and len(report['layouts'])==2
    assert report['calls']==1.
