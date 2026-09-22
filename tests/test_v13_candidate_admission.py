from copy import deepcopy

import pytest

import simbench.value.planner_v12 as planner


def _row(**updates):
    row=dict(part="carriage",yaw=0.,height=0.,placement_yaw=0.,status="rejected",
        min_clearance_m=-.004,required_clearance_m=.003,
        source_body_min_clearance_m=-.004,environment_min_clearance_m=.002,
        source_body_invalid_grasp_face_contact=False)
    row.update(updates);return row


def _run(monkeypatch,row):
    monkeypatch.setattr("simbench.assembly.placement_catalog_v13.placement_clearance_catalog",
                        lambda *a,**k:[deepcopy(row)])
    # Limit the contract test to the non-pin branch under test.
    monkeypatch.setattr(planner,"PARTS",("carriage",))
    observation=dict(fixtures={"guide_base":{"valid":True}},assembly_targets={"carriage":{}})
    cad=dict(parts={},collision_primitives={},pin_shaft_offsets_m=[])
    return planner._geometry_catalogs(observation,cad,set())


def test_only_source_target_shell_contact_is_deferred_to_twin(monkeypatch):
    usable,audit=_run(monkeypatch,_row())
    assert audit["carriage"][0]["status"]=="rejected"
    assert usable["carriage"][0]["status"]=="unknown"
    assert usable["carriage"][0]["deferred_source_target_contact"] is True


@pytest.mark.parametrize("change",[
    dict(environment_min_clearance_m=-.001),
    dict(source_body_invalid_grasp_face_contact=True),
])
def test_environment_collision_or_bad_pad_face_stays_rejected(monkeypatch,change):
    with pytest.raises(Exception,match="no executable necessary-geometry choices"):
        _run(monkeypatch,_row(**change))
