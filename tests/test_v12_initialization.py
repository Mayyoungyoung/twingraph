"""Regression for physically impossible native box-box tangent contacts."""
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from simbench.assembly.control import HOME
from simbench.core.sim_context import MjContext
from simbench.value.stage_v7 import StageV7Spec
from simbench.value.stage_v12 import INITIAL_DROP_GAP_M, write_scene
from simbench.value.supply_layout_v13 import apply


@pytest.mark.parametrize("seed,installed", [(1801,False),(1620,True)])
def test_finite_gap_and_identical_cuboids_settle_without_impossible_contacts(tmp_path,seed,installed):
    path=write_scene(StageV7Spec.sample(seed,"L1"),tmp_path,
        preinstalled_end_stop=installed,scene_layout_hook=None if installed else apply)
    root=ET.parse(path).getroot();stop=root.find(".//body[@name='end_stop']")
    initial=np.fromstring(stop.get("pos"),sep=" ")
    expected=initial-[0.,0.,INITIAL_DROP_GAP_M]
    ctx=MjContext(path,control_freq=50);ctx.reset();ctx.data.qpos[ctx.arm_qadr]=HOME
    mujoco.mj_forward(ctx.model,ctx.data);ctx.hold_arm();ctx.set_finger_ctrl(.04)
    minimum_contact=0.;maximum_height=initial[2]
    for _ in range(800):
        minimum_contact=min(minimum_contact,min((float(c.dist) for c in ctx.data.contact),default=0.))
        maximum_height=max(maximum_height,float(ctx.obj_pose("end_stop")[0][2]))
        ctx.step(nsub=1)
    final,quat=ctx.obj_pose("end_stop")
    # Numerical-model regression, not an assembly task's pose-success gate.
    assert minimum_contact>-.003
    assert maximum_height<=initial[2]+.001
    np.testing.assert_allclose(final,expected,atol=.0001)
    assert abs(quat[1])+abs(quat[2])<.001
    manifest=json.loads((tmp_path/"geometry_manifest.json").read_text())
    assert manifest["initialization"]["gap_m"]==.001
    assert manifest["initialization"]["outcome_filtering"] is False


def test_preinstalled_fixture_uses_static_cad_mate_under_base_transform(tmp_path):
    yaw=.2
    def placement(root,spec):
        base=root.find(".//body[@name='guide_base']")
        base.set("pos",".095 .070 .812")
        base.set("quat",f"{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}")
    root=ET.parse(write_scene(StageV7Spec.sample(1620,"L1"),tmp_path,
        preinstalled_end_stop=True,scene_layout_hook=placement)).getroot()
    stop=root.find(".//body[@name='end_stop']")
    expected=np.array([.095,.070,.812])+[-.092*np.cos(yaw),-.092*np.sin(yaw),.024+.001]
    np.testing.assert_allclose(np.fromstring(stop.get("pos"),sep=" "),expected,atol=1e-7)
    np.testing.assert_allclose(np.fromstring(stop.get("quat"),sep=" "),[np.cos(yaw/2),0,0,np.sin(yaw/2)],atol=1e-7)
