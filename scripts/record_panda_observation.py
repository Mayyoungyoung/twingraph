"""Record an operator's actual checkpoint measurement; never operate a robot.

The input contains only objects/scalars/flags that the operator has just
measured. Required acceptance values are not filled from the task program.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def record(schedule, command_id, measured, *, timestamp=None):
    checkpoints = [c for c in schedule.get("commands", [])
                   if c.get("id") == command_id and c.get("kind") == "verify_observation"]
    if len(checkpoints) != 1:
        raise ValueError("command-id must identify one measured observation checkpoint")
    if not isinstance(measured, dict) or not any(measured.get(k) for k in ("objects", "scalars", "flags")):
        raise ValueError("input must contain actual objects, scalars or flags measurements")
    checkpoint = checkpoints[0]
    for expected, field in (("expected_objects", "objects"), ("expected_scalars", "scalars"),
                            ("expected_flags", "flags")):
        if not set(checkpoint.get(expected, {})).issubset(measured.get(field, {})):
            raise ValueError("input is missing checkpoint measurements: " + field)
    time_value = timestamp or measured.get("measured_at_utc")
    if not time_value:
        raise ValueError("provide measured_at_utc or explicitly --measured-now for a current reading")
    parsed = datetime.fromisoformat(time_value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("measurement timestamp must include timezone")
    return dict(task_id=schedule["task_id"], command_id=command_id,
                source="operator_measured_fixture", measured_at_utc=time_value,
                **{k: measured[k] for k in ("objects", "scalars", "flags") if k in measured})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--command-id", required=True)
    parser.add_argument("--input", type=Path, required=True, help="actual measured values, not expected values")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-now", action="store_true",
                        help="operator explicitly confirms input describes a measurement just taken now")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).isoformat() if args.measured_now else None
    observation = record(json.loads(args.schedule.read_text(encoding="utf-8")), args.command_id,
                         json.loads(args.input.read_text(encoding="utf-8")), timestamp=timestamp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(observation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(dict(recorded=str(args.output), command_id=args.command_id,
                          motion_started=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
