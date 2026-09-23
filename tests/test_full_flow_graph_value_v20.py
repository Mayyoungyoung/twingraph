"""Whole-flow value input binds controller ports without reading outcomes."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from simbench.assembly.skills_v12 import configure_v12_skills
from simbench.value.full_flow_graph_value_v20 import FEATURES, encode_graph
from simbench.value.generic_graph_value_v15 import encode_graph as encode_assembly
from simbench.value.planner_v12 import normalized_graph, propose
from simbench.value.stage_v12 import make_scene


@pytest.fixture(scope="module")
def graphs():
    # MuJoCo mesh paths are relative to the generated scene directory, which
    # must be on the repository drive on Windows.
    with TemporaryDirectory(prefix="full_flow_graph_v20_", dir=Path(__file__).resolve().parents[1]) as directory:
        _, session, _, _ = make_scene(1900, Path(directory))
        configure_v12_skills(session)
        pool, _ = propose(session.decision_observation, cad=session.planning_cad, n=2, seed=1900)
        yield [normalized_graph(session, candidate) for candidate in pool]


def test_full_flow_controllers_are_in_one_graph(graphs):
    first, second = [encode_graph(graph) for graph in graphs]
    assert first["x"].shape[0] == encode_assembly(graphs[0])["x"].shape[0] + 2
    assert first["relations"].shape[-1] == first["x"].shape[0]
    assert first["relations"].sum() > encode_assembly(graphs[0])["relations"].sum()
    assert not np.array_equal(first["x"][0], second["x"][0])
    assert first["x"][-1, FEATURES.index("port_count")] == 2 / 32


def test_controller_ports_are_bound_to_executable_plan(graphs):
    changed = deepcopy(graphs[0])
    changed["cleaning"]["wipe_force"] += .1
    with pytest.raises(ValueError, match="disagree"):
        encode_graph(changed)


def test_post_execution_metadata_is_not_a_value_feature(graphs):
    changed = deepcopy(graphs[0])
    changed["rollout_label"] = True
    baseline, updated = encode_graph(graphs[0]), encode_graph(changed)
    np.testing.assert_array_equal(baseline["x"], updated["x"])
    np.testing.assert_array_equal(baseline["relations"], updated["relations"])
