from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts.check_v12_replanning import validate_request,prefix_preserved,DisturbedReleaseMonitor


def source():
    return dict(valid=True,success=True,runtime_sha256="frozen",geometry_version="printed_functional_assembly_v12",
        stage_passes=dict(cleaning_pass=True,assembly_pass=True,functional_test_pass=True,
                          fixture_capture_pass=True,final_release_and_retraction_pass=True),
        proposal={"order":["carriage","end_stop","handle"],"choices":{"carriage":{"force":3},"end_stop":{"force":4},"handle":{"force":5}}},
        boundaries=[dict(stage="end_stop",held=None,completed=["cleaning","carriage","end_stop"])])


def test_preflight_rejects_failed_or_incompatible_sources_and_completed_body():
    def validate(s,part="handle",delta=(.02,0,0)):
        return validate_request(s,"end_stop",part,delta,runtime_sha="frozen",n=48,k=4,settle=.25)
    assert validate(source())["held"] is None
    for change in ({"success":False},{"valid":False},{"runtime_sha256":"other"}):
        s=source();s.update(change)
        with pytest.raises(ValueError): validate(s)
    with pytest.raises(ValueError,match="pending"): validate(source(),part="end_stop")
    with pytest.raises(ValueError,match="horizontal"): validate(source(),delta=(.02,0,.005))
    with pytest.raises(ValueError,match="30 mm"): validate(source(),delta=(.05,0,0))


def test_prefix_preservation_requires_same_done_choices_and_order():
    current=source()["proposal"]; replacement=deepcopy(current)
    done=["cleaning","carriage","end_stop"]
    replacement["choices"]["handle"]["force"]=6
    assert prefix_preserved(current,replacement,done)
    replacement["choices"]["carriage"]["force"]=6
    assert not prefix_preserved(current,replacement,done)


def test_monitor_passes_fresh_detector_output_never_fixture_truth(monkeypatch,tmp_path):
    import scripts.check_v12_replanning as script
    from simbench.value import system_v11
    captured=[]
    # The fixture operator knows an arbitrary truth coordinate. Its value
    # must not replace the independent detector's intentionally different pose.
    monkeypatch.setattr(script,"apply_fixture_shift",lambda *args:{"truth_position":[9,9,9]})
    fresh=dict(stage="end_stop",completed=["cleaning","carriage","end_stop"],held=None,
        observation={"backend":"rgbd_geometry","sha256":"fresh","objects":{"handle":{"position_m":[.1,.2,.8],"valid":True}}})
    monkeypatch.setattr(system_v11,"observe_boundary",lambda *args:deepcopy(fresh))
    loop=lambda s,actual,current:captured.append(deepcopy(actual))
    monitor=DisturbedReleaseMonitor(loop,tmp_path,"end_stop","handle",[.02,0,0],.25)
    session=SimpleNamespace(results=[],ctx=SimpleNamespace(data=SimpleNamespace(time=1.)),hold=lambda t:None)
    actual=deepcopy(fresh);actual["observation"]["sha256"]="stale"
    monitor(session,actual,source()["proposal"])
    assert captured[0]["observation"]==fresh["observation"]
    assert "truth_position" not in captured[0] and monitor.injected
