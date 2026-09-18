#!/usr/bin/env python3
"""Audit and bind the v6 live-physics showcase videos to formal executions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


TOP4 = [12, 10, 7, 1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream["avg_frame_rate"],
        "frames": int(stream["nb_frames"]),
        "duration_seconds": float(stream["duration"]),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    output = args.output or directory / "manifest.json"

    candidates = []
    for audit_path in sorted(directory.glob("candidate_*_live.json")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        video = directory / audit["video"]
        required = (
            audit["formal_reproduction_pass"]
            and audit["success_matches_formal"]
            and audit["formal_sim_seconds"] == audit["rerun_sim_seconds"]
            and audit["maximum_terminal_position_error_m"] == 0
            and audit["renderer_object_qpos_writes"] == 0
            and audit["mujoco_version"] == "3.3.0"
        )
        if not required:
            raise ValueError(f"candidate audit failed: {audit_path}")
        candidates.append(
            {
                "candidate_index": audit["candidate_index"],
                "candidate_id": audit["candidate_id"],
                "score": audit["score"],
                "success": audit["rerun_success"],
                "sim_seconds": audit["rerun_sim_seconds"],
                "physics_steps": audit["physics_steps"],
                "terminal_error_m": audit["maximum_terminal_position_error_m"],
                "audit": audit_path.name,
                "video": video.name,
                "video_probe": probe(video),
            }
        )
    if [item["candidate_index"] for item in candidates] != list(range(1, 13)):
        raise ValueError("expected exactly candidate indices 1..12")

    actual_path = next(directory.glob("actual_target_0_*_live.json"))
    actual = json.loads(actual_path.read_text(encoding="utf-8"))
    if not (
        actual["formal_reproduction_pass"]
        and actual["rerun_success"]
        and actual["formal_sim_seconds"] == actual["rerun_sim_seconds"]
        and actual["maximum_terminal_position_error_m"] == 0
        and actual["renderer_object_qpos_writes"] == 0
    ):
        raise ValueError("independent target execution audit failed")

    composites = {}
    for name in (
        "case_71400_12_candidates_live_physics_4x3.mp4",
        "case_71400_top4_live_physics_2x2.mp4",
    ):
        composites[name] = probe(directory / name)
    actual_video = directory / actual["video"]

    manifest = {
        "schema": "twingraph.showcase.live_physics_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": 71400,
        "formal_environment": {"conda": "swdp", "mujoco": "3.3.0", "numpy": "2.2.6"},
        "execution_chain": [
            "stage_v6.make_scene",
            "stage_v6.build_pool or stage_v6.rebind_plan",
            "compile_graph",
            "RobustPhysicalRunner",
            "MjContext.step",
            "mujoco.mj_step",
            "read-only renderer callback",
        ],
        "rendering_claims": {
            "object_qpos_writes": 0,
            "recorded_state_playback": False,
            "each_candidate_tile": "independent live MuJoCo physics world",
            "synchronization": "physical time at 4x playback, 15 fps",
            "floor": "gray-white checker material; collision and friction unchanged",
            "camera": "lower third-person free camera",
        },
        "formal_reproduction": {
            "candidate_count": 12,
            "all_outcomes_match": True,
            "all_sim_durations_match": True,
            "maximum_terminal_position_error_m": 0.0,
        },
        "top4_candidate_indices_by_score": TOP4,
        "candidates": candidates,
        "composites": composites,
        "actual_target_execution": {
            "candidate_index": actual["candidate_index"],
            "candidate_id": actual["candidate_id"],
            "success": actual["rerun_success"],
            "sim_seconds": actual["rerun_sim_seconds"],
            "physics_steps": actual["physics_steps"],
            "terminal_error_m": actual["maximum_terminal_position_error_m"],
            "audit": actual_path.name,
            "video": actual_video.name,
            "video_probe": probe(actual_video),
        },
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
