"""Group-level splits and strict input/outcome provenance checks."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from .plan import PlanIR, digest
from .encode import encode_plan


@dataclass
class Group:
    id: str
    path: Path
    inputs: dict
    plans: list
    encoded: list
    trials: np.ndarray
    prefix: np.ndarray
    full: np.ndarray
    visual: np.ndarray | None

    @property
    def reference(self):
        return self.full/self.trials


def split_name(group_id, seed=17):
    value=int.from_bytes(hashlib.sha256(f"{seed}:{group_id}".encode()).digest()[:8],"big")%100
    return "train" if value<70 else "val" if value<85 else "test"


def load_groups(directory, vision=True):
    groups=[]
    seen=set()
    protocols=set()
    for complete in sorted(Path(directory).glob("group_*/complete.json")):
        p=complete.parent
        inputs=json.loads((p/"inputs.json").read_text())
        outcomes=json.loads((p/"outcomes.json").read_text())
        if digest(inputs)!=outcomes["input_sha256"]:
            raise ValueError(f"input/output checksum mismatch: {p}")
        gid=inputs["split_group"]
        if gid in seen:
            raise ValueError("duplicate configuration would bias data splits")
        seen.add(gid)
        protocols.add(inputs["protocol"])
        plans=[PlanIR.from_dict(x) for x in inputs["candidates"]]
        ids=[plan.id for plan in plans]
        if len(ids)!=len(set(ids)):
            raise ValueError("duplicate candidates")
        counts={cid:[0,0,0] for cid in ids}
        seen_trials=set()
        for trial in outcomes["trials"]:
            if not trial["valid"]:
                raise ValueError("invalid trials require collection repair")
            cid=trial["candidate_id"]
            pair=(cid,trial["trial"]["repeat"])
            if cid not in counts or pair in seen_trials:
                raise ValueError("unknown candidate or duplicate trial")
            seen_trials.add(pair)
            a,b=trial["prefix_success"],trial["suffix_success"]
            if not isinstance(a,bool) or (not a and b is not None) or (a and not isinstance(b,bool)):
                raise ValueError("conditional label semantics violated")
            if trial["full_success"]!=bool(a and b):
                raise ValueError("inconsistent full-success label")
            counts[cid][0]+=1
            counts[cid][1]+=int(a)
            counts[cid][2]+=int(bool(a and b))
        numbers=np.array([counts[c] for c in ids],np.float32)
        if np.any(numbers[:,0]==0) or len(set(numbers[:,0]))!=1:
            raise ValueError("incomplete/unpaired candidate trials")
        visual=None
        if vision:
            features=np.load(p/"visual.npz",allow_pickle=False)
            if str(features["input_sha256"])!=digest(inputs):
                raise ValueError("stale visual feature cache")
            visual=features["features"]
            if visual.ndim!=2 or visual.shape[1]!=512 or not np.isfinite(visual).all():
                raise ValueError("invalid visual features")
        groups.append(Group(gid,p,inputs,plans,[encode_plan(inputs["observation"],x) for x in plans],numbers[:,0],numbers[:,1],numbers[:,2],visual))
    if not groups or len(protocols)!=1:
        raise ValueError("empty dataset or mixed execution protocols")
    return groups

