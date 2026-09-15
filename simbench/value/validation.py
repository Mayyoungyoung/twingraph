"""Explicit candidate/repetition budgets, independent of the ranking model.

Quality: empirical success fraction, then successful-rollout mean sim time,
then input ranking. Equal, complete repeat blocks only; unused remainder is
reported. A failed block is evidence about a candidate, never infeasibility.
"""
import math
from .plan import PlanIR


def select_and_validate(ranking, validate, budget, *, mode="first_verified",
                        repeats=1, accept_rate=1.0, allow_expand=False,
                        expansion_size=None):
    if mode not in {"first_verified", "best_within_budget"}:
        raise ValueError("unknown validation mode")
    if not isinstance(budget, int) or budget < 0 or repeats < 1 or not isinstance(repeats, int):
        raise ValueError("budget and repeats must be nonnegative/positive integers")
    if not 0 < accept_rate <= 1:
        raise ValueError("accept_rate must be in (0,1]")
    rows = ranking["ranked"]
    ids = [r["candidate_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate visit in ranking")
    initial = ranking["top_k"]
    initial_ids = [r["candidate_id"] for r in initial]
    if initial_ids != ids[:len(initial)]:
        raise ValueError("top_k must be a prefix of ranked")
    k = len(initial)
    expansion_size = k if expansion_size is None else expansion_size
    if expansion_size < 1 and rows:
        raise ValueError("positive expansion size required")
    limit = len(rows) if allow_expand else k
    checked, summaries, batches = [], [], []
    chosen = None
    for pos, row in enumerate(rows[:limit]):
        if budget - len(checked) < repeats:
            break
        batch = 0 if pos < k else 1 + (pos-k)//expansion_size
        if not batches or batches[-1]["batch"] != batch:
            batches.append(dict(batch=batch, candidate_ids=[]))
        plan = PlanIR.from_dict(row["plan"])
        if plan.id != row["candidate_id"]:
            raise ValueError("ranked identity differs from executable plan")
        batches[-1]["candidate_ids"].append(plan.id)
        trials = []
        for repeat in range(repeats):
            outcome = validate(plan)
            if isinstance(outcome, bool):
                outcome = dict(success=outcome, sim_seconds=0.0)
            if not isinstance(outcome, dict) or not isinstance(outcome.get("success"), bool):
                raise TypeError("validator must return bool or a success record")
            if outcome.get("valid", True) is not True:
                raise RuntimeError("invalid execution is not a physical failure label")
            cost = float(outcome.get("sim_seconds", 0))
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("invalid rollout cost")
            record = dict(outcome, candidate_id=plan.id, repeat=repeat,
                          rollout_index=len(checked), batch=batch, visit=pos)
            checked.append(record)
            trials.append(record)
        successes = [r for r in trials if r["success"]]
        rate = len(successes)/repeats
        successful_cost = sum(r.get("sim_seconds", 0) for r in successes)/max(1,len(successes))
        summary = dict(candidate_id=plan.id, repeats=repeats, successes=len(successes),
                       rate=rate, accepted=rate >= accept_rate,
                       successful_sim_seconds=successful_cost, rank=pos)
        summaries.append(summary)
        if summary["accepted"] and mode == "first_verified":
            chosen = plan.to_dict()
            break
    if mode == "best_within_budget":
        accepted = [s for s in summaries if s["accepted"]]
        if accepted:
            best = min(accepted, key=lambda s: (-s["rate"],s["successful_sim_seconds"],s["rank"]))
            chosen = rows[best["rank"]]["plan"]
    return dict(mode=mode, k=k, budget=budget, repeats=repeats,
                accept_rate=accept_rate, allow_expand=allow_expand,
                expansion_size=expansion_size, top_k=initial,
                validated=checked, candidate_results=summaries, batches=batches,
                chosen=chosen, validation_calls=len(checked),
                unique_candidates=len(summaries), unused_budget=budget-len(checked),
                budget_fully_spent=budget-len(checked)<repeats,
                budget_exhausted=chosen is None and budget-len(checked)<repeats,
                candidate_set_exhausted=len(summaries)==limit,
                status="validated" if chosen else "unresolved_no_accepted_candidate",
                quality_rule="success_rate_desc, successful_sim_seconds_asc, ranking_order")
