"""Write the prospective v5 data/model/system specification before collection."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simbench.value.plan import digest
from simbench.value.program_input_v5 import source_manifest, encoding_hash
from simbench.value.skill_graph import interface_hash
from simbench.assembly.scene import SCENE


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv)>1 else "experiments/value_v5/protocol.json")
    if out.exists():
        raise FileExistsError("prospective protocol is immutable; choose a new named protocol for amendments")
    sources = source_manifest()
    root=Path(__file__).resolve().parents[1]
    geometry_sources={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [SCENE,*sorted((root/"simbench/assets").rglob("*"))]
                      if p.is_file()}
    spec = dict(schema="twingraph.prospective_protocol.v5", status="FROZEN_BEFORE_FORMAL_COLLECTION",
        created_unix=time.time(), code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        sources=sources, source_sha256=digest(sources), encoding_sha256=encoding_hash(), interface_sha256=interface_hash(),
        geometry_sources=geometry_sources, geometry_sha256=digest(geometry_sources),
        task=dict(family="sliding_stage_full_v5", initial="five parts unassembled in supply positions",
            parts=["carriage","end_stop","pin_left","pin_right","handle"],
            scope="five-part pose/release assembly; no sliding stroke qualification",
            terminal=dict(position_tolerance_m=.0015,tilt_tolerance_deg=3.,minimum_eef_clearance_m=.02,
                          released=True,no_finger_contact=True)),
        candidates=dict(n=12,primary_k=4,secondary_k=[1,2],
            supply_xy_half_width_m=.006,grasp_height_offsets_m=[-.001,0.,.001],
            grasp_yaw_proposals_rad=[0.,1.5707963267948966],grasp_force_N=[2.8,3.,3.2],
            guarded_speed_m_s=[.006,.007],initial_route_proposals=3,
            generation="uniform shuffled product of geometry-admissible grasp/control/order/route choices; execution-content deduplication; no outcome filtering"),
        collection=dict(train_seeds=list(range(61100,61148)),validation_seeds=list(range(61200,61212)),
            test_seeds=list(range(61300,61312)),nominal_repeats=1,workers=16,timeout_seconds=240.,
            retain_all_requests=True,no_missing_config_replacement=True),
        models=dict(kinds=[dict(name=f"mlp_{s}",kind="mlp",seed=s) for s in (17,29,43)]+[
            dict(name="linear_17",kind="linear",seed=17)],epochs=120,batch_size=48,learning_rate=.001,
            weight_decay=.01,mlp_hidden=[64,32],dropout=.1,objective="nominal physical-success binary cross entropy",
            configuration_weighting="equal",selection="minimum validation configuration-mean Brier; model-name tie break",
            classification_threshold="validation balanced accuracy, then recall, distance to0.5, lower threshold",
            no_test_retuning=True),
        system=dict(seeds=list(range(61400,61404)),n=12,k=4,validation_repeats=2,target_repeats=1,
            accept_rate=.5,friction_half_width=.03,actuator_gain_half_width=.005,
            independent_namespaces=dict(twin=5107,target=7901),workers=4,
            selection="observed twin success fraction descending, then original candidate index",
            comparison="actual TopK then actual all-N runs per case, no cached outcomes",
            target="independent model/data/unassembled scene; rebind selected initial trajectory only"),
        controls=dict(ranking="exact uniform random subset expectation and original input order",
            classification="constant label chosen from validation majority"),
        evaluation=dict(bootstrap_unit="whole physical configuration",bootstrap_samples=2000,
            include_failed_setup_denominators=True,nominal_classification_is_not_robustness_probability=True),
        provenance=dict(planner="experiments/value_v5/planner_record.json; current assistant as explicit LLM proxy",
            development="seeds61000 and61001, N12/R1, excluded from every formal split",
            source_snapshot_required=True,test_generation_after_model_and_threshold_freeze=True))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(out)
