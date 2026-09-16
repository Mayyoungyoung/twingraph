"""Serialize the assistant-authored symbolic proposal used in the v5 experiment.

This is a reproducible record of the present assistant's response, not an API
client and not evidence of repeated or independent LLM planning trials.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simbench.assembly.interfaces import PUBLIC_SKILLS, INTERFACES, AUXILIARY_INTERFACES
from simbench.value.plan import digest
from simbench.value.skill_graph import interface_hash


def record():
    catalog = dict(public_skills=PUBLIC_SKILLS, parameter_dispatch=INTERFACES,
                   feedback_and_acceptance_utilities=AUXILIARY_INTERFACES)
    orders = [["carriage", "end_stop", "pin_left", "pin_right", "handle"],
              ["carriage", "end_stop", "pin_right", "pin_left", "handle"]]
    task = "从供料位置抓取并装配滑台的滑块、端挡、左右定位销和手柄。全部零件应装到目标位姿，释放后稳定，机器人退出。"
    constraints = ["滑块从导轨左端沿导轨插入后，才可安装封闭入口的端挡。",
                   "两枚定位销依赖端挡就位，两者安装顺序可以互换。",
                   "两枚定位销均就位后安装手柄；本次终验包含五部件位姿与释放，不包含滑动行程试验。",
                   "仅从已注册技能及反馈检查接口组成程序，具体抓取、关节轨迹和接触参数由任务编译器和现有求解器实例化。"]
    response = dict(proposed_orders=orders,
        symbolic_recipe={
            "common_pick": ["detect", "estimate_pose", "estimate_grasp", "plan_path(joint)",
                            "move(joint_path)", "move(approach)", "grasp", "inspect(grasp)", "move(lift)"],
            "carriage": ["plan_path(joint)", "move(joint_path)", "move(lower)", "move(align)",
                         "plan_path(contact)", "insert(axis)", "press", "place", "move(retreat)", "inspect(pose)"],
            "remaining_parts": ["plan_path(joint)", "move(joint_path)", "move(align)",
                                "move(guarded)", "press", "place", "move(retreat)", "inspect(pose)"],
            "terminal": ["inspect(pose) for each of the five parts", "move(home)"]},
        instantiation="在两个合法零件顺序上，由任务编译器和几何求解器生成共12个不同的可执行候选。先验碰撞与可达性检查不读取执行标签；保存每个候选的全部参数和已规划轨迹。")
    return dict(schema="twingraph.planner_record.v5", task_text=task,
        source="llm_proxy", provider="OpenAI", model="Codex assistant in this task",
        prompt=dict(task=task, constraints=constraints, skill_catalog=catalog,
                    requested_output="给出完整合法零件顺序和技能骨架，供编译器实例化12个候选，不生成虚构执行结果。"),
        proposed_orders=orders, proposed_skill_programs=response["symbolic_recipe"],
        raw_response=json.dumps(response, ensure_ascii=False, indent=2),
        skill_catalog=catalog, skill_catalog_sha256=digest(catalog),
        interface_sha256=interface_hash(),
        dispatch_scope="The system consumes proposed_orders. Symbolic recipes are recorded context; stage_v5 is the explicit task compiler expanding those orders into registered skill calls.",
        provenance_scope="Authored by the current assistant with user-authorized LLM proxy; no external API invocation or planner benchmark is claimed.")


if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/value_v5/planner_record.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record(), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(path)
