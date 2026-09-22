"""Plot an archived V12 interface without reading results or creating a scene.

The eight stage cards are the existing value encoder's summary, not a claim
that every task stage has an expanded atomic graph. All displayed relation
arrows are projected from archived edges, or explicitly labelled data flow.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
from pathlib import Path
import sys


LABELS = {"cleaning":"擦拭", "carriage":"滑块", "end_stop":"端挡",
          "pin_left":"左销", "pin_right":"右销", "handle":"手柄",
          "bidirectional_stroke":"双向推动", "pin_retention":"销保持验收"}
COLORS = dict(known="#cce8ef", deferred="#f7d9a8", mixed="#daccf0",
              no_numeric="#f6eded", absent="#edf0f3", ink="#172f44", muted="#586775")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_interface(path, runtime, detail_stage):
    sys.dont_write_bytecode = True
    runtime = Path(runtime).resolve()
    sys.path.insert(0, str(runtime))
    encoder = importlib.import_module("simbench.value.graph_value_v12")
    contracts = importlib.import_module("simbench.value.skill_graph")
    if Path(encoder.__file__).resolve() != runtime/"simbench/value/graph_value_v12.py":
        raise ValueError("encoder imported from a different runtime; start a fresh process")
    path = Path(path).resolve()
    before = sha(path)
    graph = json.loads(path.read_bytes())
    if graph.get("schema") != "twingraph.full_task_graph.v12":
        raise ValueError("requires an archived V12 full-task input_graph.json")
    assembly = graph["assembly"]
    if assembly["plan_sha256"] != digest(assembly["plan"]):
        raise ValueError("archived assembly PlanIR hash mismatch")
    contracts.validate_graph(assembly)
    encoded = encoder.encode_graph(graph)  # Also validates the full-task envelope.
    stages = list(encoder.STAGES)
    order = ["cleaning", *assembly["plan"]["prefix"]["order"], "bidirectional_stroke", "pin_retention"]
    if len(order) != len(stages) or set(order) != set(stages):
        raise ValueError("stage order does not cover the exact frozen encoder stages")
    by_stage = {stage:[] for stage in stages}
    owner = {}
    for node in assembly["nodes"]:
        role = node["roles"].get("manipulated")
        if role in by_stage:
            by_stage[role].append(node)
            owner[node["index"]] = role
    if detail_stage not in by_stage or not by_stage[detail_stage]:
        raise ValueError("detail stage has no archived atomic nodes; do not fabricate a subgraph")
    projected = defaultdict(list)
    for i, edge in enumerate(assembly["edges"]):
        if edge["relation"] != "execution":
            continue
        a, b = owner.get(edge["source"]), owner.get(edge["target"])
        if a is not None and b is not None and a != b:
            projected[(a,b)].append(i)
    phase_edges = graph.get("phase_edges", [])
    if any(a not in stages or b not in stages for a,b in phase_edges):
        raise ValueError("unsupported archived phase edge")
    masks, rows = {}, []
    for stage in order:
        nodes = by_stage[stage]
        ports = [p for n in nodes for p in n["ports"]]
        mask = []
        for name, _ in encoder.PORTS:
            ps = [p for p in ports if p["name"] == name]
            # Exactly the numeric port K/D/P predicates used by encode_graph.
            known = any(p["status"] == "known" and isinstance(p["value"], (int,float)) for p in ps)
            deferred = any(p["status"] == "deferred" for p in ps)
            mask.append([int(known), int(deferred), int(bool(ps))])
        # The same per-call path-yaw slots as path_yaw_features(), rather than
        # averaging source and destination orientations into one mask.
        yaw_features = encoder.path_yaw_features(nodes)
        mask += [[int(v) for v in yaw_features[i*5+2:i*5+5]] for i in range(encoder.MAX_PATH_YAWS)]
        masks[stage] = mask
        obj = assembly["observation"]["objects"].get("wipe_tool" if stage == "cleaning" else stage)
        rows.append(dict(stage=stage, atom_indices=[n["index"] for n in nodes], atomic_nodes=len(nodes),
            known_ports=sum(p["status"] == "known" for p in ports),
            deferred_ports=sum(p["status"] == "deferred" for p in ports),
            none_values=sum(p["value"] is None for p in ports),
            valid_observation=None if obj is None else bool(obj.get("valid", obj.get("position") is not None)),
            encoder_active=bool(encoded["active"][stages.index(stage)])))
    detail = by_stage[detail_stage]
    if len(detail) > 28:
        raise ValueError("detail stage exceeds the declared 28-node page; choose another stage or extend the plot explicitly")
    detail_ids = {n["index"] for n in detail}
    detail_edges = [e for e in assembly["edges"] if e["relation"] == "execution"
                    and e["source"] in detail_ids and e["target"] in detail_ids]
    if sha(path) != before:
        raise ValueError("input graph changed during read-only rendering")
    source_files = ("simbench/value/graph_value_v12.py", "simbench/value/skill_graph.py", "simbench/value/plan.py")
    report = dict(schema="twingraph.graph_interface_visualization.v12", input_graph=str(path),
        input_file_sha256=before, graph_content_sha256=digest(graph),
        candidate=graph["proposal"]["name"], geometry_version=graph["task_geometry_version"],
        assembly_plan_sha256=assembly["plan_sha256"], interface_sha256=assembly["interface_sha256"],
        observation_sha256=assembly["observation"].get("perception", {}).get("observation_sha256"),
        graph_contract_validated=True, full_task_envelope_matches_value_input=True,
        value_encoder_schema=encoder.SCHEMA, value_feature_shape=list(encoded["x"].shape),
        relation_tensor_shape=list(encoded["relations"].shape), encoder_stage_order=stages,
        candidate_execution_order=order, stages=rows, numeric_port_names=[p[0] for p in encoder.PORTS],
        mask_columns=[p[0] for p in encoder.PORTS]+[f"path_yaw_{i+1}" for i in range(encoder.MAX_PATH_YAWS)], masks=masks,
        actual_atomic_nodes=len(assembly["nodes"]), actual_atomic_edges=len(assembly["edges"]),
        actual_edge_type_counts=dict(Counter(e["relation"] for e in assembly["edges"])),
        unassigned_atomic_nodes=[n["index"] for n in assembly["nodes"] if n["index"] not in owner],
        projected_execution_edges=[dict(source=a, target=b, archived_edge_indices=indices) for (a,b),indices in projected.items()],
        archived_phase_edges=phase_edges, detail_stage=detail_stage, detail_nodes=detail, detail_execution_edges=detail_edges,
        limits=["stage cards are an encoder summary, not the full atomic graph",
                "stages without atom nodes use archived task metadata; no atomic expansion is invented",
                "execution rebinds deferred values from later observations and grasp registration",
                "no value prediction, physical execution, success label, or new graph edge is produced"],
        read_only_source_sha256={p:sha(runtime/p) for p in source_files}, tool_sha256=sha(__file__))
    return report


def render(report, out, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
    from matplotlib.path import Path as PlotPath
    import numpy as np
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((name for name in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "SimSun") if name in fonts), None)
    if cjk is None:
        raise RuntimeError("a Chinese font is required for readable labels; install/provide one before plotting")
    plt.rcParams.update({"font.family":cjk, "axes.unicode_minus":False, "pdf.fonttype":42,
                         "font.size":10, "text.color":COLORS["ink"]})
    fig = plt.figure(figsize=(18, 14), facecolor="white")
    fig.text(.045, .964, "V12 技能图谱接口 · 阶段摘要", size=23, weight="bold")
    fig.text(.045, .937, f"{label}  |  {report['candidate']}  |  实际原子图见 JSON；本图不是执行成功证据", size=12, color=COLORS["muted"])

    flow = fig.add_axes([.045, .812, .91, .101]); flow.set(xlim=(0,100), ylim=(0,10)); flow.axis("off")
    def box(ax, x,y,w,h,face="#f3f6f9",edge="#c8d2dc"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=.15",linewidth=1,edgecolor=edge,facecolor=face))
    box(flow, 0,.7,31,8.3)
    flow.text(1.4,7.2,"归档 input_graph.json",weight="bold",size=12)
    flow.text(1.4,5.1,f"assembly.plan：{report['actual_atomic_nodes']} 原子节点 / {report['actual_atomic_edges']} 条边",size=10)
    flow.text(1.4,3.2,f"PlanIR SHA：{report['assembly_plan_sha256'][:18]}…",size=9)
    flow.text(1.4,1.6,"图 / PlanIR / 接口哈希校验通过",size=10,color="#18705e")
    box(flow, 38,5.25,62,3.75,face="#ecf5f8")
    flow.text(39.4,7.4,f"价值输入：8 阶段 × {report['value_feature_shape'][1]} 维 + {report['relation_tensor_shape'][0]} 类关系矩阵",size=11,weight="bold")
    flow.text(39.4,5.9,"当前 linear 头仅用阶段特征，不使用关系矩阵；参数与 masks 一起输入",size=10)
    box(flow,38,.7,62,3.75,face="#f2f5ec")
    flow.text(39.4,2.9,"执行封装：同一 order / choices / 擦拭参数，一致性检查通过",size=11,weight="bold")
    flow.text(39.4,1.35,"deferred 值在执行时由新观测和抓取配准绑定；此处不运行执行器",size=10)
    for y in (7.1,2.5):
        flow.add_patch(FancyArrowPatch((31,4.9),(38,y),arrowstyle="-|>",mutation_scale=11,color="#657f93",linewidth=1.2))
    flow.text(34.5,.03,"数据路径示意（非技能图边）",ha="center",size=8,color=COLORS["muted"])

    fig.text(.045,.786,"01  八阶段与候选执行次序",size=14,weight="bold")
    stage_ax=fig.add_axes([.045,.619,.91,.157]); stage_ax.set(xlim=(-.1,8),ylim=(-.78,1.8));stage_ax.axis("off")
    order=report["candidate_execution_order"]
    places={s:i for i,s in enumerate(order)}
    for a,b in report["archived_phase_edges"]:
        x1,x2=places[a]+.45,places[b]+.45
        peak=.80+min(abs(x2-x1),3)*.21
        path=PlotPath([(x1,.66),(x1,peak),(x2,peak),(x2,.66)], [PlotPath.MOVETO,PlotPath.CURVE4,PlotPath.CURVE4,PlotPath.CURVE4])
        stage_ax.add_patch(FancyArrowPatch(path=path,arrowstyle="-|>",mutation_scale=9,color="#a19aad",linestyle=(0,(3,2)),linewidth=1))
    for row in report["stages"]:
        s=row["stage"]; i=places[s]
        box(stage_ax,i,-.15,.9,.84,face="#eff5f9" if row["atomic_nodes"] else "#f6f5f0")
        stage_ax.text(i+.45,.47,f"{i+1:02d}  {LABELS[s]}",ha="center",weight="bold",size=11)
        stage_ax.text(i+.45,.24,f"{row['atomic_nodes']} 个原子节点" if row["atomic_nodes"] else "阶段元数据",ha="center",size=10)
        seen="物体观测有效" if row["valid_observation"] else "物体观测未知"
        if row["valid_observation"] is None: seen="无独立物体观测项"
        stage_ax.text(i+.45,.025,seen,ha="center",size=8.5,color=COLORS["muted"])
    stage_ax.text(0,-.43,"虚线仅为归档 phase_edges；卡片编号是候选主体阶段顺序。原子 execution 关系与末尾验收回边见 JSON。",size=10,color=COLORS["muted"])
    stage_ax.text(0,-.68,f"另有 {len(report['unassigned_atomic_nodes'])} 个全局原子节点不属于上述阶段；无原子展开的阶段不伪造节点。",size=9,color=COLORS["muted"])

    fig.text(.045,.591,"02  数值端口 / 路径 yaw 状态 masks（与价值编码器相同）",size=14,weight="bold")
    mask_ax=fig.add_axes([.13,.349,.825,.213]);cols=report["mask_columns"]
    mask_ax.set(xlim=(0,len(cols)),ylim=(len(order),0))
    labels=[]
    for y,s in enumerate(order):
        labels.append(LABELS[s])
        for x,(known,deferred,present) in enumerate(report["masks"][s]):
            key="absent" if not present else "mixed" if known and deferred else "known" if known else "deferred" if deferred else "no_numeric"
            mask_ax.add_patch(Rectangle((x,y),1,1,facecolor=COLORS[key],edgecolor="white",linewidth=1.2))
            mask_ax.text(x+.5,y+.5,f"{known}/{deferred}/{present}",ha="center",va="center",size=10)
    mask_ax.set_xticks(np.arange(len(cols))+.5, [name.replace("minimum_insertion_depth_m","insert_depth").replace("height_offset","height").replace("path_yaw_","yaw#") for name in cols],rotation=25,ha="left",fontsize=9)
    mask_ax.xaxis.tick_top();mask_ax.tick_params(axis="both",length=0,pad=5)
    mask_ax.set_yticks(np.arange(len(order))+.5,labels,fontsize=10)
    for spine in mask_ax.spines.values():spine.set_visible(False)
    fig.text(.045,.323,"每格 K / D / P：K = 有已知数值；D = 至少一个延后绑定端口；P = 端口存在。缺失值不当作真实零。",size=10,color=COLORS["muted"])
    fig.text(.045,.303,"0/0/0 = 无该端口；0/0/1 = 端口存在但暂无数值（可为 None / 默认值）；1/1/1 = 同阶段同时有已知和延后值。",size=10,color=COLORS["muted"])

    detail=report["detail_nodes"]
    fig.text(.045,.272,f"03  {LABELS[report['detail_stage']]}阶段放大 · {len(detail)} 个真实原子节点（仅画其真实 execution 边）",size=14,weight="bold")
    atom_ax=fig.add_axes([.045,.072,.91,.181]);atom_ax.axis("off")
    perrow=7;nr=(len(detail)+perrow-1)//perrow
    atom_ax.set(xlim=(-.05,7),ylim=(-nr+.02,0.05))
    coords={}
    for i,node in enumerate(detail):
        row,col=divmod(i,perrow)
        if row%2:col=perrow-1-col
        x,y=col,-row-.8;coords[node["index"]]=(x,y)
        box(atom_ax,x,y,.9,.7,face="#f2f6f9")
        deferred=sum(p["status"]=="deferred" for p in node["ports"])
        atom_ax.text(x+.07,y+.50,f"#{node['index']:03d}  {node['skill']}",size=9,weight="bold")
        atom_ax.text(x+.07,y+.27,f"{len(node['ports'])} 端口 / {deferred} deferred",size=8.5,color=COLORS["muted"])
        atom_ax.text(x+.07,y+.08,node["kind"],size=8,color=COLORS["muted"])
    for edge in report["detail_execution_edges"]:
        x,y=coords[edge["source"]];tx,ty=coords[edge["target"]]
        if abs(y-ty)<1e-6:
            a,b=((x+.9,y+.36),(tx,ty+.36)) if tx>x else ((x,y+.36),(tx+.9,ty+.36))
        else:a,b=((x+.45,y),(tx+.45,ty+.7))
        atom_ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=8,linewidth=1,color="#617b8e"))
    fig.text(.045,.050,"这里没有画出的 data / object_state / scene / robot / holding / observations 边均保留在原始 JSON；不可把该放大视图当作完整原子图。",size=9,color=COLORS["muted"])
    fig.text(.045,.031,f"input_graph 文件 SHA256  {report['input_file_sha256']}",size=8.5,color=COLORS["muted"])
    fig.text(.045,.017,"图谱作用：状态与参数显式编码，依赖和延后绑定可校验，同一候选的价值输入与执行参数不再各写一份。此图不包含价值分数或结果标签。",size=9,color=COLORS["muted"])
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    fig.savefig(out/"stage_summary.png",dpi=180,facecolor="white")
    fig.savefig(out/"stage_summary.pdf",facecolor="white",metadata={"Title":"V12 archived skill graph stage summary", "Subject":report["input_file_sha256"]})
    plt.close(fig)
    report["display_label"]=label
    report["output_sha256"]={name:sha(out/name) for name in ("stage_summary.png","stage_summary.pdf")}
    (out/"graph_summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"README.md").write_text("# V12 技能图谱接口可视化\n\n"
        f"本图使用 **{label} / {report['candidate']}** 的真实归档初始图，未读取执行结果。\n\n"
        "这是价值编码器的八阶段摘要，不是完整原子技能图。清洁、推动与销保持验收以现有任务元数据表示，未伪造其原子展开。"
        "八阶段间虚线仅来自保存的 phase_edges；执行关系的阶段投影（含末尾验收回边）保存在摘要 JSON。下方只放大一个真实装配阶段，并仅显示其内部 execution 关系。\n\n"
        "K/D/P 表与现有编码器一致：已知数值、延后绑定和存在 mask。未数值化端口可能是 None/默认值，不等同于物体观测失败。"
        "图/PlanIR/当前技能接口和 full_plan 候选参数均经只读一致性校验；执行时仍需新观测绑定 deferred 值，本可视化不证明执行成功。\n\n"
        f"原始完整图：`{report['input_graph']}`\n\n"
        "可复核的实际节点、边索引、mask、输入/代码 SHA 与输出 SHA 均保存在 `graph_summary.json`。"
        "PNG/PDF 是同一幅图，无新增物理实验、模型预测或标签读取。\n",encoding="utf-8")
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--runtime",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out",type=Path,required=True,help="New evidence directory; never overwrite an archive")
    parser.add_argument("--label",required=True,help="Truthful archive context, e.g. development seed1604; not an outcome claim")
    parser.add_argument("--detail-stage",default="pin_left")
    args=parser.parse_args()
    if args.out.exists() or args.out.resolve().is_relative_to(args.input.resolve().parent):
        parser.error("output must be new and outside the input archive directory")
    report=load_interface(args.input,args.runtime,args.detail_stage)
    out=render(report,args.out,args.label)
    print(json.dumps(dict(output=str(out),atomic_nodes=report["actual_atomic_nodes"],atomic_edges=report["actual_atomic_edges"],
        input_sha256=report["input_file_sha256"],contract_validated=True),ensure_ascii=False))


if __name__=="__main__":
    main()
