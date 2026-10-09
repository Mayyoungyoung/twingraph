"""Export the prescribed nominal CAD body positions as a placement figure."""
import argparse
import json
from pathlib import Path


def plot(layout_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layout = json.loads(Path(layout_path).read_text(encoding="utf-8"))
    if layout["frame"] != "panda_base" or layout["units"] != "m":
        raise ValueError("need Panda-base positions in meters")
    figure, (axes, table_axes) = plt.subplots(1, 2, figsize=(13.2, 6.8),
                                             gridspec_kw={"width_ratios": [1.4, 1]})
    axes.axhline(0, color="#98a2b3", lw=.8)
    axes.axvline(0, color="#98a2b3", lw=.8)
    axes.scatter([0], [0], marker="*", s=150, color="#222222", zorder=4)
    axes.annotate("Panda base origin", (0, 0), (12, 10), textcoords="offset points", fontsize=9)
    offsets = {"guide_base": (8, 12), "carriage": (8, 8), "end_stop": (8, -16),
               "handle": (8, -14), "pin_left": (-5, 14), "pin_right": (8, -16), "wipe_tool": (8, 9)}
    rows = []
    for name, pose in layout["poses"].items():
        x, y, z = [1000 * value for value in pose["position_m"]]
        rows.append([name, f"{x:.1f}", f"{y:.1f}", f"{z:.1f}"])
        if name.endswith("holder"):
            continue
        color = "#c46a12" if name == "guide_base" else "#126078"
        axes.scatter([x], [y], s=55, color=color, zorder=4)
        label = name.replace("_", " ")
        if name.startswith("pin_"):
            label += " + holder"
        axes.annotate(label, (x, y), offsets[name], textcoords="offset points", fontsize=9,
                      ha="center" if name == "pin_left" else "left")
    axes.set(xlim=(-50, 810), ylim=(-420, 235), xlabel="Panda base X / mm", ylabel="Panda base Y / mm")
    axes.set_aspect("equal")
    axes.grid(alpha=.2)
    axes.set_title("Nominal initial placement: CAD body origins", loc="left", fontsize=12)
    table_axes.axis("off")
    table = table_axes.table(cellText=rows, colLabels=["Part", "X / mm", "Y / mm", "Z / mm"],
                             colWidths=[.47, .18, .18, .17], bbox=[0, .35, 1, .57])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#e1e5ea")
        if row == 0:
            cell.set_facecolor("#e8f1f4")
            cell.set_text_props(weight="bold")
    table_axes.text(0, .24, "Tabletop: base z = 0 mm\nGuide fixture: 6 mm spacer required\nPins share XY with their vertical holders\nYaw/orientation: see coordinate document", fontsize=10, va="top")
    table_axes.text(0, .06, "Prescribed nominal layout; not a physical measurement.\nCAD body positions are not robot EE/TCP commands.", fontsize=9, color="#626c76", va="top")
    figure.suptitle("Panda tabletop placement - seed 4000", x=.08, ha="left", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, .95))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, default=Path("real_robot/config/nominal_layout_4000.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/panda_layout_4000.png"))
    args = parser.parse_args()
    plot(args.layout, args.output)
