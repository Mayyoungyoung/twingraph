"""The curated pool changes executable call arguments, not only plan labels."""

from simbench.value import stage_v5
from simbench.value.plan import plain
from simbench.value.v9_candidates import proposals, SOURCE
from scripts.run_v9_target_screening import reference_proposal


def _command_signature(proposal):
    calls = []
    for index, part in enumerate(proposal["order"]):
        calls += stage_v5.stage_calls(part, [0., 0., .85],
                                      proposal["choices"][part], index,
                                      v7=True, functional_clearance=True)
    assembly = tuple((call.skill, tuple(sorted((name, repr(plain(argument.value)))
                                      for name, argument in call.arguments.items())))
                     for call in calls)
    cleaning = (proposal["wipe_variant"], proposal["wipe_force"], proposal["wipe_duration"])
    return assembly, cleaning


def test_twelve_curated_v9_candidates_have_distinct_executable_commands():
    pool = proposals()
    assert len(pool) == 12
    assert len({proposal["name"] for proposal in pool}) == 12
    assert all(proposal["source"] == SOURCE for proposal in pool)
    assert all(tuple(proposal["order"]) in stage_v5.legal_orders() for proposal in pool)
    assert len({_command_signature(proposal) for proposal in pool}) == 12
    reference = pool[0]
    assert reference == reference_proposal()
    assert reference["name"] == "reference"
    assert pool[1]["order"] != reference["order"]
    assert pool[2]["choices"]["pin_left"]["speed"] != reference["choices"]["pin_left"]["speed"]
    assert pool[3]["choices"]["pin_left"]["force"] != reference["choices"]["pin_left"]["force"]
    assert pool[5]["choices"]["pin_left"]["height"] != reference["choices"]["pin_left"]["height"]
    assert pool[7]["choices"]["pin_left"]["clearance"] != reference["choices"]["pin_left"]["clearance"]
    assert pool[10]["wipe_variant"] != reference["wipe_variant"]
    assert pool[11]["wipe_duration"] != reference["wipe_duration"]
