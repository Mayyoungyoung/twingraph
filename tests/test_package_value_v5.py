"""Packaging checks use tiny synthetic files; no model loading or simulation."""
import io
import tarfile

import pytest

from scripts import package_value_v5 as packaging


def fixture_run(tmp_path):
    repo, run = tmp_path / "repo", tmp_path / "run"
    code = repo / "simbench/value/example.py"
    code.parent.mkdir(parents=True)
    code.write_text("print('frozen source')\n", encoding="utf-8")
    source = {"simbench/value/example.py": packaging.sha(code)}
    group = run / "data/group_1"
    request = dict(seed=1, split="train", n=1, repeats=1)
    inp = dict(schema="twingraph.group.v5", declared_split="train", candidates=[dict(id="a")],
               source_sha256=packaging.digest(source))
    trial = dict(candidate_id="a", trial={"domain": "nominal", "repeat": 0},
                 success=False, timeout=False, valid=True)
    for name, value in {
        "request.json": request, "inputs.json": inp, "source.json": source,
        "outcomes.json": dict(input_sha256=packaging.digest(inp), trials=[trial]),
        "complete.json": dict(input_sha256=packaging.digest(inp), request_sha256=packaging.digest(request),
                              candidates=1, trials=1, all_successes=0),
    }.items():
        packaging.write(group/name, value)
    dev = run / "development_failed/group_7"
    packaging.write(dev/"failure.json", dict(request=dict(seed=7, split="development", n=12, repeats=1),
                                             message="physical proposal failure"))
    packaging.write(dev/"source.json", {"simbench/value/historical.py": "a"*64})
    models = run / "models/mlp_17"
    models.mkdir(parents=True)
    (models/"best.pt").write_bytes(b"synthetic checkpoint archive fixture; never torch-loaded")
    split = {"train": [1], "val": [2]}
    for name, value in {"source.json": source, "split.json": split, "history.json": [],
        "input_schema.json": {"fixture": True}, "summary.json": dict(kind="mlp",seed=17,
            checkpoint_sha256=packaging.sha(models/"best.pt"), source_sha256=packaging.digest(source),
            split_sha256=packaging.digest(split))}.items():
        packaging.write(models/name,value)
    system = run / "systems/case1/top_k"
    inputs = dict(schema="twingraph.system_inputs.v5", candidates=[])
    packaging.write(system/"inputs.json", inputs)
    packaging.write(system/"source.json", source)
    packaging.write(system/"result.json", dict(schema="twingraph.system_run.v5",method="top_k",
        config_id="case1",status="executed_independent_target", inputs_sha256=packaging.digest(inputs),
        implementation_sha256=packaging.digest(source), simulation_calls=dict(twin_validation=0,target_execution=1),
        validation=[], target_trials=[dict(status="executed",success=False,isolation=dict(
            distinct_session=True,distinct_context=True,distinct_model=True,distinct_data=True,snapshot_transfer=False))],
        target_successes=0,requested_target_trials=1,seconds=dict(total=1.)))
    (system/"target_state_trace_0.npz").write_bytes(b"synthetic state trace retained byte-for-byte")
    (system/"target_terminal_0.png").write_bytes(b"synthetic failed terminal image retained byte-for-byte")
    freeze = run/"frozen.json"
    packaging.write(freeze,dict(schema="test.freeze.v5",models=[dict(path="mlp_17/best.pt",sha256=packaging.sha(models/"best.pt"))]))
    return repo, run, freeze


def test_package_preserves_all_failed_dev_model_and_target_trace_files(tmp_path, monkeypatch):
    repo,run,freeze=fixture_run(tmp_path)
    monkeypatch.setattr(packaging,"environment",lambda: {"kind":"test_fixture"})
    result=packaging.package(run,tmp_path/"artifact","recorded-test-commit",freeze,repo=repo)
    manifest=packaging.read(tmp_path/"artifact/manifest.json")
    assert result["models"]==1 and result["systems"]==1 and result["groups"]==2
    assert len(manifest["failed_development"])==1
    assert manifest["failed_development"][0]["status"]=="failed"
    assert any(row["development"] and not row["exact_bytes_complete"] for row in manifest["source_manifests"])
    assert any(name.endswith("target_state_trace_0.npz") for name in manifest["retained_files"])
    assert any(name.endswith("target_terminal_0.png") for name in manifest["retained_files"])
    assert set(manifest["retained_files"])=={p.relative_to(run).as_posix() for p in packaging.files_under(run)}
    assert all(artifact["bytes"]<packaging.LIMIT for artifact in manifest["artifacts"])
    with pytest.raises(ValueError,match="nonempty"):
        packaging.package(run,tmp_path/"artifact","recorded-test-commit",freeze,repo=repo)


def test_tampered_raw_outcome_hash_blocks_package(tmp_path):
    repo,run,freeze=fixture_run(tmp_path)
    path=run/"data/group_1/outcomes.json"
    changed=packaging.read(path);changed["input_sha256"]="b"*64;packaging.write(path,changed)
    with pytest.raises(ValueError,match="outcomes/input hash"):
        packaging.package(run,tmp_path/"artifact","test",freeze,repo=repo)


def test_missing_exact_formal_source_blocks_package(tmp_path):
    repo,run,freeze=fixture_run(tmp_path)
    (repo/"simbench/value/example.py").write_text("print('changed')\n",encoding="utf-8")
    with pytest.raises(ValueError,match="exact formal source bytes unavailable"):
        packaging.package(run,tmp_path/"artifact","test",freeze,repo=repo)


def test_snapshot_recovers_exact_script_bytes_and_rejects_unsafe_names(tmp_path):
    repo,run,freeze=fixture_run(tmp_path)
    payload=b"print('historical script')\n"
    snapshot=tmp_path/"old_source_snapshot.tar.gz"
    with tarfile.open(snapshot,"w:gz") as tar:
        member=tarfile.TarInfo("saved_root/scripts/old.py");member.size=len(payload)
        tar.addfile(member,io.BytesIO(payload))
    resolver=packaging.ExactSources(repo,run,"test",[snapshot])
    recovered,origin=resolver.resolve("scripts/old.py",packaging.hashlib.sha256(payload).hexdigest())
    assert recovered==payload and origin=="retained_collection_source_snapshot"
    with pytest.raises(ValueError,match="unsafe archive path"):
        packaging.safe_archive_name("../escape.py")


def test_changed_frozen_checkpoint_blocks_package(tmp_path):
    repo,run,freeze=fixture_run(tmp_path)
    packaging.write(freeze,dict(checkpoint_sha256="f"*64))
    with pytest.raises(ValueError,match="missing or changed checkpoint"):
        packaging.package(run,tmp_path/"artifact","test",freeze,repo=repo)
