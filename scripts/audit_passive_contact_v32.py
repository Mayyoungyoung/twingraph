"""Isolated resting-body numerical audit; never produces assembly labels."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

root=ET.parse('results/v32_push_diagnostic/stage_v12_scene.xml').getroot()
for node in list(root):
    if node.tag in ('include','actuator','sensor','contact','equality','tendon'):
        root.remove(node)
world=root.find('worldbody')
for body in list(world):
    if body.tag=='body' and body.get('name') not in ('guide_base','carriage'):
        world.remove(body)
carriage=world.find("body[@name='carriage']")
carriage.set('pos','-.030 .085 .8241'); carriage.set('quat','1 0 0 0')
result=[]
for dt,multiccd in ((.002,False),(.001,False),(.0005,False),(.002,True)):
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    model.opt.timestep=dt
    if multiccd:
        model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    data=mujoco.MjData(model)
    start=data.qpos[:3].copy()
    positions=[]
    for i in range(round(10/dt)):
        mujoco.mj_step(model,data)
        if i%round(1/dt)==0:positions.append(data.qpos[:3].tolist())
    result.append(dict(timestep=dt,multiccd=multiccd,drift=(data.qpos[:3]-start).tolist(),positions=positions))
print(json.dumps(result,indent=2))
Path('results/v32_passive_contact.json').write_text(json.dumps(result,indent=2))
