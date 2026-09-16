"""Real-file hash verification for portable frozen-model system dispatches."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import export_value_v5_predictions as exporter
from scripts import run_value_v5_system_batch as batch
from simbench.value import system_v5


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    names = {"mlp_17": ("mlp", 17), "mlp_29": ("mlp", 29),
             "mlp_43": ("mlp", 43), "linear_17": ("linear", 17)}
    root = tmp_path / "relocated"
    models = {}
    for name, (kind, seed) in names.items():
        path = root / name / "best.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes((name + ":frozen-weights").encode())
        models[name] = dict(path=str(tmp_path / "absent-original-host" / name / "best.pt"),
            sha256=exporter.sha(path), kind=kind, seed=seed, source_sha256="source",
            encoding_sha256="encoding", interface_sha256="interface",
            validation_nominal_brier=.1 if name == "mlp_29" else .2)
    spec = dict(source_sha256="source", encoding_sha256="encoding", interface_sha256="interface",
        models=dict(kinds=[dict(name=n, kind=k, seed=s) for n, (k, s) in names.items()]),
        collection=dict(test_seeds=[31], validation_seeds=[21], nominal_repeats=1),
        candidates=dict(n=12, primary_k=4), geometry_sources={},
        system=dict(n=12, k=4, seeds=[41, 42, 43, 44], workers=4,
                    independent_namespaces=dict(twin=5107, target=7901)))
    selection = dict(schema=exporter.SELECTION_SCHEMA, models=models, selected="mlp_29",
        source_sha256="source", test=dict(seeds=[31], n=12, repeats=1),
        validation_seeds=[21], primary_k=4)
    planner = dict(interface_sha256="interface", task_text="frozen task")
    put(tmp_path / "experiments/value_v5/planner_record.json", planner)
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "validate_protocol", lambda value: dict(protocol_sha256=batch.digest(value)))
    monkeypatch.setattr(system_v5, "validate_planner_record", lambda value: copy.deepcopy(value))
    return SimpleNamespace(root=root, spec=spec, selection=selection, planner=planner, tmp=tmp_path)


def test_dispatch_checks_all_four_relocated_files_without_mutating_freeze(frozen, monkeypatch):
    before = copy.deepcopy(frozen.selection)
    accessed = []
    original_sha = exporter.sha
    def tracked(path):
        accessed.append(Path(path))
        return original_sha(path)
    monkeypatch.setattr(exporter, "sha", tracked)
    binding = batch.validate_dispatch(frozen.spec, frozen.selection, frozen.planner,
                                      checkpoint_root=frozen.root)
    assert set(accessed) == {frozen.root / name / "best.pt" for name in before["models"]}
    assert binding["checkpoint"] == str((frozen.root / "mlp_29/best.pt").resolve())
    assert binding["model_selection_sha256"] == batch.digest(before)
    assert frozen.selection == before


@pytest.mark.parametrize("name", ["mlp_17", "mlp_29", "mlp_43", "linear_17"])
def test_tampering_any_relocated_model_is_rejected(frozen, name):
    (frozen.root / name / "best.pt").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match=name + ": checkpoint hash differs"):
        batch.validate_dispatch(frozen.spec, frozen.selection, frozen.planner,
                                checkpoint_root=frozen.root)


def test_selected_only_override_stays_compatible(frozen):
    for name, row in frozen.selection["models"].items():
        if name != frozen.selection["selected"]:
            row["path"] = str(frozen.root / name / "best.pt")
    before = copy.deepcopy(frozen.selection)
    binding = batch.validate_dispatch(frozen.spec, frozen.selection, frozen.planner,
        selected_checkpoint=frozen.root / "mlp_29/best.pt")
    assert binding["selected"] == "mlp_29"
    assert frozen.selection == before


def test_relocation_options_cannot_be_combined(frozen):
    with pytest.raises(ValueError, match="mutually exclusive"):
        batch.validate_dispatch(frozen.spec, frozen.selection, frozen.planner,
            selected_checkpoint=frozen.root / "mlp_29/best.pt", checkpoint_root=frozen.root)


def test_check_only_relocates_without_creating_results(frozen):
    protocol = frozen.tmp / "protocol.json"
    selection = frozen.tmp / "selection.json"
    planner = frozen.tmp / "planner.json"
    for p, value in [(protocol, frozen.spec), (selection, frozen.selection), (planner, frozen.planner)]:
        put(p, value)
    original_bytes = selection.read_bytes()
    output = frozen.tmp / "must-not-exist"
    result = batch.run_batch(protocol, selection, planner, output,
        check_only=True, checkpoint_root=frozen.root)
    assert result["status"] == "validated_only"
    assert result["checkpoint_sha256"] == frozen.selection["models"]["mlp_29"]["sha256"]
    assert not output.exists()
    assert selection.read_bytes() == original_bytes


def test_collection_test_relocation_rejects_unselected_tampering_before_dispatch(frozen, monkeypatch):
    from scripts import run_value_v5_collection as collection
    spec = copy.deepcopy(frozen.spec)
    spec["source_sha256"] = collection.digest({})
    selection = copy.deepcopy(frozen.selection)
    selection["source_sha256"] = spec["source_sha256"]
    thresholds = dict(source_sha256=spec["source_sha256"], selected=selection["selected"],
        selection_metadata=dict(model_selection_sha256=collection.digest(selection)), primary_k=4,
        methods={name: dict(model_sha256=row["sha256"]) for name, row in selection["models"].items()})
    paths = [frozen.tmp / name for name in ["protocol.json", "selection.json", "thresholds.json"]]
    for path, value in zip(paths, [spec, selection, thresholds]):
        put(path, value)
    original = paths[1].read_bytes()
    (frozen.root / "linear_17/best.pt").write_bytes(b"tampered unselected model")
    monkeypatch.setattr(collection, "source_manifest", lambda: {})
    out = frozen.tmp / "no-collection"
    monkeypatch.setattr(collection.sys, "argv", ["collect", "--protocol", str(paths[0]),
        "--out", str(out), "--splits", "test", "--selection", str(paths[1]),
        "--thresholds", str(paths[2]), "--checkpoint-root", str(frozen.root)])
    with pytest.raises(ValueError, match="linear_17: checkpoint hash differs"):
        collection.main()
    assert not out.exists()
    assert paths[1].read_bytes() == original
