"""Replay one supplied development result with read-only contact diagnostics.

Run inside an untouched copy of the result's runtime. No observer data is
returned to the control/perception stack; this script adds no recovery.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def run(source_result, output):
    import mujoco
    from simbench.assembly.library import Session
    from simbench.value.provenance_v12 import fingerprint
    from simbench.value.system_v12 import rollout
    source_result=Path(source_result);output=Path(output)
    original=json.loads(source_result.read_text());provenance=fingerprint()
    if provenance["sha256"]!=original["runtime_sha256"]:
        raise ValueError("audit runtime differs from the supplied result")
    output.mkdir(parents=True,exist_ok=False)
    events=(output/"observer.jsonl").open("w",encoding="utf8")
    state=dict(part=None,phase=None,samples=0);sessions=[]
    def snapshot(session,kind):
        if state["part"]!="end_stop":return
        ctx=session.ctx;m,d=ctx.model,ctx.data
        stop=ctx.body_id("end_stop");base=ctx.body_id("guide_base")
        R=d.xmat[stop].reshape(3,3);bR=d.xmat[base].reshape(3,3)
        reg=getattr(session,"held_visual_transforms_v12",{}).get("end_stop")
        sensor=(ctx.eef_pos()+ctx.eef_mat()@reg["local_position"]).tolist() if reg else None
        contacts=[]
        for i,c in enumerate(d.contact):
            bodies=list(map(int,m.geom_bodyid[[c.geom1,c.geom2]]))
            if stop not in bodies:continue
            f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
            contacts.append(dict(geoms=[m.geom(c.geom1).name,m.geom(c.geom2).name],
                bodies=[m.body(b).name for b in bodies],distance_m=float(c.dist),
                force_contact_frame=f.tolist(),normal_world=c.frame[:3].tolist(),
                position_world_m=c.pos.tolist()))
        row=dict(kind=kind,phase=state["phase"],time_s=float(d.time),held=session.held,
            true_position_m=d.xpos[stop].tolist(),true_quat_wxyz=d.xquat[stop].tolist(),
            true_center_base_m=(bR.T@(d.xpos[stop]-d.xpos[base])).tolist(),
            true_tilt_deg=float(np.degrees(np.arccos(np.clip(R[:,2]@bR[:,2],-1,1)))),
            registered_fk_position_m=sensor,eef_position_m=ctx.eef_pos().tolist(),
            bilateral_contact=ctx.grasp_contacts("end_stop"),contacts=contacts,
            source="independent simulator diagnostic observer; never read by controller")
        if kind!="control_step":
            row["arm_qpos"]=ctx.arm_qpos.tolist()
            row["finger_qpos"]=ctx.finger_qpos.tolist()
        events.write(json.dumps(row)+"\n");state["samples"]+=1
    old_init,old_call=Session.__init__,Session.call
    def initialize(self,*args,**kwargs):
        old_init(self,*args,**kwargs);sessions.append(self)
        self.ctx.on_control_step=lambda ctx:snapshot(self,"control_step")
    def call(self,name,**params):
        if params.get("part") is not None:state["part"]=params["part"]
        previous=state["phase"];state["phase"]=name
        snapshot(self,"before_skill")
        if state["part"]=="end_stop" and name in ("place","place_object"):
            np.savez_compressed(output/"before_place_diagnostic_state.npz",qpos=self.ctx.data.qpos,
                qvel=self.ctx.data.qvel,ctrl=self.ctx.data.ctrl,time=self.ctx.data.time)
        try:return old_call(self,name,**params)
        finally:
            snapshot(self,"after_skill");state["phase"]=previous
    Session.__init__,Session.call=initialize,call
    protocol=dict(source_result=str(source_result),source_result_sha256=hashlib.sha256(source_result.read_bytes()).hexdigest(),
        runtime=provenance,observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose="one unchanged development replay with contact diagnostics; no training/validation label",
        observational_only=True,no_video=True,excluded_from_timing_comparison=True)
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf8")
    try:
        result=rollout(original["seed"],original["proposal"],output/"rollout",domain=original["domain"],record=False)
    finally:
        events.close();Session.__init__,Session.call=old_init,old_call
    if sessions:
        ctx=sessions[-1].ctx
        np.savez_compressed(output/"final_diagnostic_state.npz",qpos=ctx.data.qpos,qvel=ctx.data.qvel,ctrl=ctx.data.ctrl,time=ctx.data.time)
    print(json.dumps(dict(success=result["success"],error=str(result.get("error","")).split("{")[0],
        samples=state["samples"],wall_seconds=result["wall_seconds"])))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source-result",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();run(a.source_result,a.out)
