from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from simbench.assembly import pushing
from simbench.assembly.graph import catalog_graph, validate_skeleton


def inputs():
    return (dict(xyz=[0.,0.,.82],quat=[1.,0.,0.,0.]),
            [dict(type='box',pos=[0.,0.,.025],size=[.022,.014,.015])],
            dict(body_z_interval_m=[.01,.04]))


def test_push_estimate_rigid_transform_and_no_object_name():
    pose,geometry,region=inputs()
    a=pushing.estimate(pose,geometry,region,[.1,0.,.82],[1.,0.,0.])
    rotation=Rotation.from_euler('z',.71)
    translation=np.array([.2,-.3,.04])
    q=rotation.as_quat()
    b=pushing.estimate(dict(xyz=rotation.apply(pose['xyz'])+translation,quat=q[[3,0,1,2]]),
        geometry,region,rotation.apply([.1,0.,.82])+translation,rotation.apply([1.,0.,0.]))
    for key in ('contact','hover','position','target'):
        np.testing.assert_allclose(b[key],rotation.apply(a[key])+translation,atol=1e-12)
    with pytest.raises(ValueError):
        pushing.estimate(pose,geometry,region,[.1,0.,.82],[0.,0.,0.])
    with pytest.raises(ValueError):
        pushing.estimate(pose,geometry,region,[0.,0.,.82],[1.,0.,0.])


def test_passive_motion_cannot_become_positive(monkeypatch):
    pose,geometry,region=inputs()
    plan=pushing.estimate(pose,geometry,region,[.03,0.,.82],[1.,0.,0.])
    plan.update(speed=.015,force_limit=10.,press_force=.05)
    state={'object':np.array(pose['xyz']),'eef':np.array(plan['contact'])}
    def servo(command):
        state['object']=state['object']+np.array([.0005,0.,0.])
        state['eef']=np.asarray(command).copy()
    ctx=SimpleNamespace(pad_span=lambda:.008,obj_pos=lambda part:state['object'].copy(),
        eef_pos=lambda:state['eef'].copy(),control_dt=.02)
    session=SimpleNamespace(ctx=ctx,held=None,arm=SimpleNamespace(servo=servo))
    monkeypatch.setattr(pushing,'finger_contact_force',lambda ctx,part:0.)
    result=pushing.execute(session,'unseen_box',plan)
    assert result.metrics['progress_m']>.01
    assert not result.ok
    assert result.metrics['contact_steps']==0


def test_push_contract_requires_pose_plan_and_empty_jaws():
    graph=catalog_graph()
    nodes={n['name']:n for n in graph['nodes']}
    assert {'estimate_push_pose','push','gripper'} <= nodes.keys()
    assert all(c['name']=='push_object' for c in nodes['push']['contracts'])
    bad=validate_skeleton([dict(skill='push',params=dict(part='carriage'))])
    assert not bad['valid']
    occupied=validate_skeleton([dict(skill='gripper',params=dict(mode='close_empty'))],initial_held='carriage')
    assert not occupied['valid']


def test_flat_mesh_support_is_stable(tmp_path):
    import mujoco
    import xml.etree.ElementTree as ET
    from simbench.assembly.printed_kit import _exact_cuboid_collisions
    root=ET.fromstring('''<mujoco><option timestep=".002"><flag multiccd="enable"/></option>
      <asset/><worldbody><body name="support"><geom name="surface" type="box" size=".1 .1 .01"/></body>
      <body name="object" pos="0 0 .0201"><freejoint/><geom type="box" size=".027 .023 .01" mass=".03"/></body>
      </worldbody></mujoco>''')
    _exact_cuboid_collisions(root.find('asset'),root.find(".//body[@name='support']"))
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    d=mujoco.MjData(m)
    for _ in range(2500):mujoco.mj_step(m,d)
    assert np.linalg.norm(d.qpos[:2])<.0001
    assert abs(d.qpos[2]-.02)<.0001


def test_kinematic_jacobian_matches_full_forward():
    import mujoco
    from simbench.assembly.control import make_context
    ctx=make_context()
    m=ctx.model
    a,b=mujoco.MjData(m),mujoco.MjData(m)
    rng=np.random.default_rng(32)
    for _ in range(5):
        a.qpos[:]=ctx.data.qpos
        a.qpos[ctx.arm_qadr]+=rng.uniform(-.1,.1,7)
        b.qpos[:]=a.qpos
        mujoco.mj_forward(m,a)
        mujoco.mj_kinematics(m,b);mujoco.mj_comPos(m,b)
        ja,ra,jb,rb=[np.zeros((3,m.nv)) for _ in range(4)]
        mujoco.mj_jacSite(m,a,ja,ra,ctx.eef_site_id)
        mujoco.mj_jacSite(m,b,jb,rb,ctx.eef_site_id)
        np.testing.assert_array_equal(a.site_xpos,b.site_xpos)
        np.testing.assert_array_equal(ja,jb)
        np.testing.assert_array_equal(ra,rb)
