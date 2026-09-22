"""Deterministic-clock budget tests; fixtures are not physical trial evidence."""
from types import SimpleNamespace
import numpy as np
import pytest

from simbench.value import physical,system_v12 as system


class Clock:
    def __init__(self): self.now=0.
    def __call__(self): return self.now
    def advance(self,seconds): self.now+=seconds


def fixture(monkeypatch):
    clock=Clock()
    monkeypatch.setattr(physical.time,"perf_counter",clock)
    monkeypatch.setattr(physical,"fingerprint",lambda s:"fixture")
    session=SimpleNamespace(snapshot=lambda:{},restore=lambda state:None,results=[],parts=[],
        arm=SimpleNamespace(trace=[]),decision_observation={},stage_targets={},artifacts={},task_version="fixture")
    session.ctx=SimpleNamespace(model=SimpleNamespace(actuator_gainprm=np.ones((1,3)),
        actuator_biasprm=np.ones((1,3)),geom_friction=np.ones((1,3)),opt=SimpleNamespace(timestep=.01)),
        data=SimpleNamespace(time=0.),step=lambda:None)
    plan=SimpleNamespace(id="fixture",prefix=dict(start_state="fixture"))
    monkeypatch.setattr(physical,"execute_suffix",lambda s,p:None)
    return clock,session,plan


def test_optional_pause_preserves_full_wall_cost_and_legacy_deadline(monkeypatch):
    clock,session,plan=fixture(monkeypatch)
    paused=[0.]
    def execute(s,p):
        clock.advance(200.);s.ctx.step()
        clock.advance(1000.);paused[0]+=1000.  # synchronous suffix search
        clock.advance(200.);s.ctx.step()
    monkeypatch.setattr(physical,"execute_prefix",execute)
    trial=dict(friction_scale=1.,actuator_gain_scale=1.)
    result=physical.PhysicalRunner(session,timeout=600.,excluded_wall_seconds=lambda:paused[0]).run(plan,trial)
    assert result["success"] and not result["timeout"]
    assert result["wall_seconds"]==1400. and result["excluded_monitor_wall_seconds"]==1000.
    assert result["execution_budget_wall_seconds"]==400. and result["execution_timeout_seconds"]==600.
    # Default callers still count the entire wall interval against 600 s.
    clock.now=0.;paused[0]=0.
    legacy=physical.PhysicalRunner(session,timeout=600.).run(plan,trial)
    assert legacy["timeout"] and not legacy["success"] and legacy["wall_seconds"]==1400.


@pytest.mark.parametrize("active_after_monitor,expected_success",[(100.,True),(401.,False)])
def test_v12_excludes_monitor_only_and_censors_actual_execution_timeout(monkeypatch,tmp_path,active_after_monitor,expected_success):
    clock,session,plan=fixture(monkeypatch)
    monkeypatch.setattr(system,"require_frozen_source",lambda:dict(sha256="a"*64))
    monkeypatch.setattr(system,"source_fingerprint",lambda:dict(sha256="a"*64))
    monkeypatch.setattr(system,"make_scene",lambda *a,**k:(None,session,None,None))
    monkeypatch.setattr(system,"normalized_graph",lambda *a,**k:{})
    monkeypatch.setattr(system,"bind",lambda *a,**k:plan)
    def monitor(s,actual,current): clock.advance(1000.)
    def stages(s,*,monitor,**kwargs):
        clock.advance(200.);s.ctx.step()
        monitor(s,dict(stage="fixture"),{})
        clock.advance(active_after_monitor);s.ctx.step()
    monkeypatch.setattr(system,"run_staged",stages)
    monkeypatch.setattr(physical,"execute_prefix",lambda s,p:s.full_task_controller(s))
    result=system.rollout(1600,dict(name="fixture"),tmp_path,domain="deployment",monitor=monitor)
    assert result["success"] is expected_success
    assert result["valid"] is expected_success
    assert result["resource_censored"] is (not expected_success)
    assert result["timeout"] is (not expected_success)
    assert result["monitor_wall_seconds"]==result["excluded_monitor_wall_seconds"]==1000.
    assert result["execution_budget_wall_seconds"]==200.+active_after_monitor
    assert result["total_wall_seconds"]==result["wall_seconds"]==1200.+active_after_monitor
    assert result["execution_timeout_seconds"]==600.
    if not expected_success:
        assert "resource-censored" in result["invalid_reason"]
