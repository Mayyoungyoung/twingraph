"""Capture/replay real RGB-D; never assume a simulator camera extrinsic."""
import argparse
import json
from pathlib import Path
from simbench.value.realsense_v12 import capture,save_frame,load_frame


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--world-from-optical-json",type=Path)
    parser.add_argument("--serial")
    parser.add_argument("--bag",type=Path)
    parser.add_argument("--replay",type=Path)
    parser.add_argument("--infer",action="store_true")
    args=parser.parse_args()
    if args.replay:
        frame,calibration=load_frame(args.replay)
    else:
        if not args.world_from_optical_json: parser.error("capture requires measured --world-from-optical-json")
        payload=json.loads(args.world_from_optical_json.read_text(encoding="utf-8"))
        transform=payload.get("world_from_optical") if isinstance(payload,dict) else payload
        frame,calibration,metadata=capture(transform,serial=args.serial,bag=args.bag)
        save_frame(args.out,frame,calibration,metadata)
    if args.infer:
        from simbench.value.cad_rgbd_v12 import estimate_scene,templates
        from simbench.value.rgbd_perception import to_sensor_observation
        # Keep camera inference usable without importing the MuJoCo runtime.
        ALL_PARTS=("carriage","end_stop","pin_left","pin_right","handle","wipe_tool")
        result=estimate_scene({"realsense":frame},{"realsense":calibration},templates())
        observation=to_sensor_observation(result,ALL_PARTS)
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/"observation.json").write_text(json.dumps(observation,indent=2),encoding="utf-8")
        print(json.dumps({p:dict(valid=r["valid"],position_m=r["position_m"]) for p,r in observation["objects"].items()}))


if __name__=="__main__": main()
