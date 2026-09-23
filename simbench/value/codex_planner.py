"""Optional Codex CLI transport for the generic atomic planning contract."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from .atomic_flow import RESPONSE_SCHEMA, compile_candidates, compose_from_template
from .plan import digest


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "atomic_candidate_response.schema.json"
PATCH_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "atomic_port_compositions.schema.json"


def decode_model_response(raw, observation, *, expected_count):
    """Parse strict model output; reject invalid programs without repair."""
    if not isinstance(raw, dict) or set(raw) != {"observation_sha256", "plans"}:
        raise ValueError("model output has an unexpected shape")
    if raw["observation_sha256"] != digest(observation):
        raise ValueError("model output is bound to a different observation")
    plans = []
    for row in raw["plans"]:
        calls = []
        for item in row["calls"]:
            if not isinstance(item, dict) or set(item) != {"skill", "manipulated", "params_json"}:
                raise ValueError("model call has an unexpected shape")
            params = json.loads(item["params_json"])
            if not isinstance(params, dict):
                raise ValueError("params_json must encode a parameter mapping")
            roles = {} if item["manipulated"] is None else {"manipulated": item["manipulated"]}
            calls.append(dict(skill=item["skill"], params=params, roles=roles))
        plans.append(dict(name=row["name"], calls=calls))
    response = dict(schema=RESPONSE_SCHEMA,
                    observation_sha256=raw["observation_sha256"], plans=plans)
    candidates = compile_candidates(response, observation, expected_count=expected_count)
    return response, candidates


def generate_with_codex(request, *, out, executable="codex", timeout_seconds=600):
    """Invoke one read-only model turn and archive its unedited JSON output."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if not isinstance(request, dict) or request.get("schema") != "twingraph.atomic_candidate_request.v1":
        raise ValueError("unsupported planner request")
    observation = request["observation"]
    prompt = ("You are a task planner. Produce exactly "
              f"{request['candidate_count']} distinct complete executable atomic plans. "
              "Use only the registered skills and implementation parameter signatures in the JSON request. "
              "Each params_json must be a JSON object encoded as a string. "
              "Object references must appear in the observation. "
              "Use different legitimate grasp/pose/path solutions and skill sequences where supported. "
              "Do not infer rollout outcomes, fabricate success labels, or invent unregistered skills. "
              "Copy observation_sha256 exactly. Return only the JSON object required by the output schema.\n\n"
              + json.dumps(request, ensure_ascii=False, sort_keys=True, allow_nan=False))
    (out / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    command = [executable, "exec", "-s", "read-only", "--ephemeral",
               "--skip-git-repo-check", "--output-schema", str(SCHEMA_PATH),
               "-o", str(out / "model_output.json"), "-"]
    result = subprocess.run(command, input=prompt.encode("utf-8"), capture_output=True,
                            timeout=timeout_seconds, check=False)
    (out / "model_stderr.log").write_text(result.stderr.decode("utf-8", errors="replace"), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Codex planner exited {result.returncode}; inspect model_stderr.log")
    raw = json.loads((out / "model_output.json").read_text(encoding="utf-8"))
    response, candidates = decode_model_response(raw, observation,
        expected_count=request["candidate_count"])
    (out / "accepted_response.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    return response, candidates


def generate_port_compositions_with_codex(task_spec, observation, template, allowed_values,
                                          *, candidate_count, out, executable="codex",
                                          timeout_seconds=600):
    """Ask the model to compose solver-grounded ports on a complete atomic plan."""
    from .atomic_flow import skill_catalog
    template = template.to_dict() if hasattr(template, "to_dict") else template
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    request = dict(schema="twingraph.atomic_port_composition_request.v1",
        task_spec=task_spec, observation=observation,
        observation_sha256=digest(observation), template_plan=template,
        allowed_values=allowed_values, atomic_skills=skill_catalog(),
        candidate_count=candidate_count,
        rule="Choose only listed pre-execution port values. Return complete plan variants as edits to the template. Do not use outcome labels.")
    (out / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt = (f"Produce exactly {candidate_count} distinct atomic plan variants for the task. "
              "Each edit must reference a template call id and argument and choose a value exactly from allowed_values. "
              "Use value_json to encode that chosen JSON value as a string. "
              "The empty edits list is allowed for one baseline. Do not invent skills or values. "
              "Copy observation_sha256 exactly. Return only the JSON object required by the schema.\n\n"
              + json.dumps(request, ensure_ascii=False, sort_keys=True, allow_nan=False))
    command = [executable, "exec", "-s", "read-only", "--ephemeral",
               "--skip-git-repo-check", "--output-schema", str(PATCH_SCHEMA_PATH),
               "-o", str(out / "model_output.json"), "-"]
    result = subprocess.run(command, input=prompt.encode("utf-8"), capture_output=True,
                            timeout=timeout_seconds, check=False)
    (out / "model_stderr.log").write_text(result.stderr.decode("utf-8", errors="replace"), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Codex planner exited {result.returncode}; inspect model_stderr.log")
    raw = json.loads((out / "model_output.json").read_text(encoding="utf-8"))
    if set(raw) != {"observation_sha256", "plans"} or raw["observation_sha256"] != digest(observation):
        raise ValueError("port composition response is not bound to the observation")
    if len(raw["plans"]) != candidate_count:
        raise ValueError("port composition candidate count differs")
    proposals = []
    for row in raw["plans"]:
        proposals.append(dict(name=row["name"], edits=[
            dict(call_id=edit["call_id"], argument=edit["argument"],
                 value=json.loads(edit["value_json"])) for edit in row["edits"]]))
    candidates = compose_from_template(template, observation, proposals, allowed_values)
    (out / "accepted_compositions.json").write_text(
        json.dumps(proposals, ensure_ascii=False, indent=2), encoding="utf-8")
    for i, candidate in enumerate(candidates):
        (out / f"candidate_{i:02d}_plan.json").write_text(
            json.dumps(candidate["plan"].to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return proposals, candidates
