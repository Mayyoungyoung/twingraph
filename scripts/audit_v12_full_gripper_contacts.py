"""Check compiled masks and robot self-contact at declared HOME jaw gaps."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mujoco
from simbench.assembly.control import HOME
from simbench.assembly.gripper_clearance_v12 import PANDA,SHELLS,PADS
from simbench.value.stage_v7 import StageV7Spec
from simbench.value.stage_v12 import write_scene


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    scene=write_scene(StageV7Spec.sample(1600,"L1"),out/"scene")
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
    adr=lambda name:int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)])
    d.qpos[[adr(f"joint{i}") for i in range(1,8)]]=HOME
    rows=[]
    for gap in (0.,.0095,.02,.04):
        d.qpos[[adr("finger_joint1"),adr("finger_joint2")]]=[gap,-gap]
        mujoco.mj_forward(m,d)
        contacts=[]
        for c in d.contact:
            names=[m.geom(c.geom1).name,m.geom(c.geom2).name]
            if not any(name in SHELLS+PADS for name in names):continue
            contacts.append(dict(geoms=names,body_names=[m.body(int(m.geom_bodyid[g])).name for g in (c.geom1,c.geom2)],
                distance_m=float(c.dist)))
        rows.append(dict(finger_gap_m=gap,contacts=contacts))
    result=dict(panda_xml_sha256=hashlib.sha256(PANDA.read_bytes()).hexdigest(),
        masks={name:dict(contype=int(m.geom(name).contype),conaffinity=int(m.geom(name).conaffinity)) for name in SHELLS+PADS},
        records=rows,physical_steps=0,source="static robot FK/collision at HOME; no task-success claims")
    (out/"contacts.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);run(p.parse_args().out)
