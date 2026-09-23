#!/usr/bin/env python3
"""Run one auditable Codex model call for a frozen atomic composition prompt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def digest(value):
    # Prompt and response are JSON values, so this matches PlanIR's digest.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--executable", default="codex")
    parser.add_argument("--model")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    request = json.loads(args.prompt.read_text(encoding="utf-8"))
    if request.get("schema") not in (
            "twingraph.llm_atomic_composition_request.v20.r1",
            "twingraph.llm_atomic_composition_request.v20.r2"):
        raise ValueError("unsupported frozen composition prompt")
    schema = args.prompt.with_suffix(".schema.json")
    if not schema.is_file():
        raise ValueError(f"missing strict response schema: {schema}")
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "model_output.json"
    if output.exists():
        raise ValueError(f"refusing to overwrite a model response: {output}")
    text = (
        "You plan a complete robotic task using the supplied generic atomic skill graph. "
        "The only positive label is success of ALL SIX final physical acceptance predicates "
        "after the entire plan; later actions may disturb earlier placements. "
        f"Return exactly {request['candidate_count']} distinct complete plans named "
        f"llm_000 through llm_{request['candidate_count']-1:03d} in order. "
        "For each plan choose one listed legal order donor, one cleaning donor, and one "
        "listed donor for every manipulated role. A role donor supplies its complete "
        "precomputed solver-grounded atomic ports and geometry conditions. Use the task, "
        "observation, skill graph and pre-execution conditions to choose diverse plausible "
        "solutions. Do not use or guess rollout labels and do not invent skill values. "
        "Copy observation_sha256 and base_pool_sha256 exactly. Return only JSON satisfying "
        "the output schema. "
        + ("Use baseline_donor_index for every donor slot in llm_000, then diversify "
           "the remaining plans. " if "baseline_donor_index" in request else "")
        + "\n\n"
        + json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    )
    command = [args.executable, "exec", "-s", "read-only", "--ephemeral",
               "--skip-git-repo-check", "--output-schema", str(schema),
               "-o", str(output)]
    if args.model:
        command += ["-m", args.model]
    command.append("-")
    started = time.perf_counter()
    result = subprocess.run(command, input=text.encode("utf-8"), capture_output=True,
        timeout=args.timeout_seconds, check=False)
    seconds = time.perf_counter() - started
    stderr = result.stderr.decode("utf-8", errors="replace")
    (args.out / "model_stderr.log").write_text(stderr, encoding="utf-8")
    version = subprocess.run([args.executable, "--version"], capture_output=True,
        text=True, check=False).stdout.strip()
    model = re.search(r"^model:\s*(.+)$", stderr, flags=re.MULTILINE)
    effort = re.search(r"^reasoning effort:\s*(.+)$", stderr, flags=re.MULTILINE)
    metadata = dict(schema="twingraph.llm_atomic_generation.v20.r1",
        returncode=result.returncode, seconds=seconds, cli_version=version,
        model=(model.group(1).strip() if model else args.model),
        reasoning_effort=(effort.group(1).strip() if effort else None),
        prompt_sha256=digest(request), output_exists=output.is_file())
    if output.is_file():
        answer = json.loads(output.read_text(encoding="utf-8"))
        metadata["response_sha256"] = digest(answer)
        metadata["candidate_count"] = len(answer.get("plans", []))
        if (answer.get("observation_sha256") != request["observation_sha256"]
                or answer.get("base_pool_sha256") != request["base_pool_sha256"]
                or metadata["candidate_count"] != request["candidate_count"]):
            metadata["response_contract_valid"] = False
        else:
            metadata["response_contract_valid"] = True
    (args.out / "model_run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))
    if result.returncode != 0 or not metadata.get("response_contract_valid"):
        raise RuntimeError("model generation failed; inspect model_run.json and model_stderr.log")


if __name__ == "__main__":
    main()
