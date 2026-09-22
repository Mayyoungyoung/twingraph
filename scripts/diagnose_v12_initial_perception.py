import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simbench.value.stage_v12 import make_scene
from simbench.assembly.skills_v12 import configure_v12_skills
from simbench.value.stage_v7 import refresh_visual_observation, capture_detector
import numpy as np
for level in ("L0", "L1"):
    out = Path("results/white_initial_diagnostic") / level
    _, session, _, _ = make_scene(1600, out, level=level)
    first = session.decision_observation
    configure_v12_skills(session)
    second = refresh_visual_observation(session)
    (out / "first.json").write_text(json.dumps(first, indent=2))
    (out / "second.json").write_text(json.dumps(second, indent=2))
    frames, cal = capture_detector(session)
    np.savez_compressed(out / "frames.npz", **{v+"_"+k:a for v,f in frames.items() for k,a in f.items()})
    (out / "calibrations.json").write_text(json.dumps({v:c.manifest() for v,c in cal.items()}))
    print(level, json.dumps({"first":{p:r["valid"] for p,r in first["objects"].items()}, "second":{p:r["valid"] for p,r in second["objects"].items()}}))
