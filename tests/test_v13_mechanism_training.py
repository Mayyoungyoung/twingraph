import json

import pytest

pytest.importorskip("torch")

from scripts.train_value_v13_mechanism import fit


def test_training_refuses_layout_leakage(tmp_path):
    with pytest.raises(ValueError,match="layout leakage"):
        fit([],[],"a"*64,{"train":[1],"validation":[1],"k":2},tmp_path,epochs=1)


def test_split_is_layout_disjoint_and_development_only():
    split=json.loads(open("docs/evidence/value_v13_mechanism/splits_development.json").read())
    assert set(split["train"]).isdisjoint(split["validation"])
    assert split["k"]==2
