"""Ten task-facing atoms. Direction, carrying and solver choices are parameters.

Legacy names remain callable but are not additional skill-graph nodes.
"""

PUBLIC_SKILLS = {
    "detect": dict(
        label="检测",
        kind="information",
        inputs="观测范围",
        outputs="对象及观测",
        description="发现场景中的对象",
    ),
    "estimate_pose": dict(
        label="物体位姿估计",
        kind="information",
        inputs="对象观测",
        outputs="物体位姿及不确定性",
        description="估计物体在哪里、朝向如何",
    ),
    "estimate_grasp": dict(
        label="抓取位姿估计",
        kind="information",
        inputs="物体位姿、几何",
        outputs="多个抓取候选",
        description="给出夹爪应放在哪里、开多大",
    ),
    "plan_path": dict(
        label="路径规划",
        kind="information",
        inputs="起点、目标、约束",
        outputs="路径候选或接触运动参数",
        description="生成运动参数，不移动机器人",
    ),
    "move": dict(
        label="移动",
        kind="execution",
        inputs="目标/位移/路径、持物对象及约束",
        outputs="到达误差、抓持状态",
        description="移动末端或持物；所有方向和搬运共用一个节点",
    ),
    "grasp": dict(
        label="抓取",
        kind="execution",
        inputs="对象、夹持力",
        outputs="双侧稳定抓持",
        description="在已有抓取姿态处闭爪并建立抓持，不包含搬运",
    ),
    "place": dict(
        label="放置",
        kind="execution",
        inputs="持物对象、支撑目标及容差",
        outputs="释放且位置稳定",
        description="在已有支撑位置释放并检查落稳，不包含远距离移动",
    ),
    "insert": dict(
        label="插入",
        kind="execution",
        inputs="配合对象、插入参数、策略",
        outputs="插入深度/误差/接触状态",
        description="沿配合约束插入，可选反馈或学习策略",
    ),
    "press": dict(
        label="压靠",
        kind="execution",
        inputs="对象、压靠高度、力阈值",
        outputs="接触力与压靠误差",
        description="建立指定压靠接触；需要接触和高度同时满足",
    ),
    "wipe": dict(
        label="擦拭",
        kind="execution",
        inputs="擦拭工具、表面路径、接触力与策略",
        outputs="实际接触覆盖率、力与跟踪误差",
        description="沿学习的覆盖轨迹保持表面接触；当前模拟擦拭，不模拟磨料去除",
    ),
}

# Parameter alternatives of the same atom, not additional public nodes.
INTERFACES = {
    "detect": {"scene": "observe_parts"},
    "estimate_grasp": {"generate": "propose_grasps"},
    "plan_path": {
        "joint": "plan_transfer",
        "cartesian": "plan_linear",
        "contact": "plan_insertion",
        "recovery": "plan_recovery",
        "surface": "plan_wipe",
    },
    "move": {
        "joint_path": "execute_joint_path",
        "cartesian_path": "execute_cartesian_path",
        "lift": "lift",
        "lower": "lower",
        "home": "home",
        "retreat": "retreat",
        "approach": "approach",
        "orient": "orient_wrist",
        "align": "align_axis",
        "guarded": "guarded_descent",
        "retract": "retract_contact",
        "constrained": "move_constrained",
        "target": "move_to",
    },
    "grasp": {"close": "close_gripper"},
    "place": {"release": "place_object"},
    "insert": {
        "axis": "slide_insert",
        "spiral": "spiral_search",
        "learned": "learned_insert",
    },
    "press": {"seat": "press_seat"},
    "wipe": {"learned": "wipe_surface"},
}

# Feedback and acceptance utilities remain callable, but are not skill nodes.
AUXILIARY_INTERFACES = {
    "measure": {"value": "measure_value"},
    "inspect": {
        "pose": "inspect_seat",
        "pin": "inspect_pin_inserted",
        "pin_joint": "inspect_pin_joint",
        "receiver_relation": "inspect_receiver_relation",
        "grasp": "verify_grasp",
        "stroke": "verify_stroke",
        "clean": "verify_clean",
        "measurement": "inspect_measurement",
    },
}

# Compatibility only: these entries do not appear in the task-facing graph.
LEGACY_INTERFACES = {
    "observe": {"scene": "observe_parts"},
    "gripper": {"open": "open_gripper", "close": "close_gripper"},
    "guarded_move": {
        "descend": "guarded_descent",
        "seat": "press_seat",
        "retract": "retract_contact",
    },
    "actuate": {"slide": "move_constrained"},
}


def resolve(name, params):
    params = dict(params)
    if name == "move" and "mode" not in params:
        selectors = [
            key for key in ("delta", "target", "path", "grasp") if key in params
        ]
        if len(selectors) != 1:
            raise ValueError("move requires exactly one of delta, target, path, grasp")
        if "delta" in params:
            import math

            delta = params["delta"]
            if len(delta) != 3 or not all(math.isfinite(float(x)) for x in delta):
                raise ValueError("move delta must have three finite coordinates")
            if delta[0] == 0 and delta[1] == 0 and delta[2] != 0:
                if params.get("part") is not None or delta[2] > 0:
                    params.pop("delta")
                    params["height"] = abs(float(delta[2]))
                    return (
                        ("lift" if delta[2] > 0 else "lower")
                        if params.get("part") is not None
                        else "retreat"
                    ), params
            return "move_to", params
        if "path" in params:
            params["artifact"] = params.pop("path")
            space = params.pop("space", "joint")
            if space not in ("joint", "cartesian"):
                raise ValueError("move space must be joint or cartesian")
            return (
                "execute_joint_path" if space == "joint" else "execute_cartesian_path"
            ), params
        if "grasp" in params:
            params["artifact"] = params.pop("grasp")
            return "approach", params
        if isinstance(params["target"], str) and params["target"] == "home":
            params.pop("target")
            return "home", params
        reference = params.pop("reference", "eef")
        if reference not in ("eef", "object"):
            raise ValueError("move reference must be eef or object")
        if reference == "object":
            return "align_axis", params
        return "move_to", params
    if name == "plan_path":
        params["mode"] = params.pop("method", params.get("mode", "joint"))
    elif name == "inspect":
        params["mode"] = params.pop("what", params.get("mode", "pose"))
    elif name == "insert":
        params["mode"] = params.pop("strategy", params.get("mode", "axis"))
    modes = INTERFACES.get(
        name, AUXILIARY_INTERFACES.get(name, LEGACY_INTERFACES.get(name))
    )
    if modes is not None:
        mode = params.pop("mode", None)
        if mode is None and len(modes) == 1:
            mode = next(iter(modes))
        if mode not in modes:
            raise ValueError(
                f"{name}: invalid parameter choice; expected {tuple(modes)}"
            )
        name = modes[mode]
    return name, params


def family(component):
    if component in ("select_grasp", "propose_grasps"):
        return "estimate_grasp"
    if component == "measure_clearance":
        return "auxiliary"
    if any(component in modes.values() for modes in AUXILIARY_INTERFACES.values()):
        return "auxiliary"
    if component in PUBLIC_SKILLS:
        return component
    return next(
        (name for name, modes in INTERFACES.items() if component in modes.values()),
        "internal",
    )
