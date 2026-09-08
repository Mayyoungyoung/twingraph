"""Collect and audit isolated skill videos; no web frontend is generated."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .library import CATALOG


def video_audit(path):
    reader = imageio.get_reader(str(path))
    frames = reader.count_frames()
    meta = reader.get_meta_data()
    for frame_id in (0, frames // 2, frames - 1):
        im = reader.get_data(frame_id)
        assert im.shape == (1000, 1600, 3)
    reader.close()
    return dict(
        frames=frames,
        fps=meta["fps"],
        seconds=frames / meta["fps"],
        decoded_first_middle_last=True,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--out", default="results/standalone_skills/delivery")
    p.add_argument("--cube", default="results/standalone_skills/cube_example")
    a = p.parse_args()
    rows = {}
    for source in a.sources:
        source = Path(source)
        for row in json.loads((source / "verification.json").read_text()):
            row["source_directory"] = str(source)
            if "physics_steps" in row:
                # Early recordings named this field imprecisely; it counts
                # plotted trace samples, not MuJoCo integration steps.
                row["trace_samples"] = row.pop("physics_steps")
            rows[row["skill"]] = row
    assert set(rows) == set(CATALOG), (
        set(CATALOG) - set(rows),
        set(rows) - set(CATALOG),
    )
    out = Path(a.out)
    for folder in ("skills", "previews", "evidence"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    index = [
        "# 原子技能独立演示",
        "",
        "30 个原子技能，分别使用方块、障碍物、单孔插销座和短导轨。每段为固定 MuJoCo 视角，底部显示技能名称与标记说明。",
        "",
        "[小方块完整抓取示例](cube_grasp.mp4) · [实现与重录说明](IMPLEMENTATION.md)",
        "",
        "规划、检测与验收技能展示计算结果，不额外驱动机械臂。青线为计划，黄线为实测运动；绿色通常为目标或指垫接触，橙色为外部接触。检测目标与坐标轴采用各自的颜色标记，以视频底部说明为准。",
        "",
        "| 技能 | 独立场景 | 视频 | 时长 |",
        "|---|---|---|---|",
    ]
    scenes = {
        "perception.xml": "三个彩色方块",
        "cube.xml": "单个 40 mm 方块",
        "obstacle.xml": "方块与被动障碍物",
        "pin.xml": "单个插销与孔座",
        "rail.xml": "单个滑块与短导轨",
    }
    font = ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18
    )
    sheet = Image.new("RGB", (1600, 6 * 238), (242, 245, 249))
    draw = ImageDraw.Draw(sheet)
    for i, (name, spec) in enumerate(CATALOG.items()):
        row = rows[name]
        assert row["success"], row
        assert row["overlay_pixels"] > 20, (name, row["overlay_pixels"])
        source = Path(row["source_directory"])
        video = out / "skills" / (name + ".mp4")
        shutil.copy2(source / row["video"], video)
        preview = out / "previews" / (name + ".png")
        shutil.copy2(source / "previews" / (name + ".png"), preview)
        row["video_audit"] = video_audit(video)
        assert row["sha256"] == row["video_audit"]["sha256"]
        # Preserve machine-readable path/trace arrays separately from the index.
        (out / "evidence" / (name + ".json")).write_text(
            json.dumps(row, ensure_ascii=False, indent=2)
        )
        index.append(
            f"| {spec.label} | {scenes[row['scene']]} | [{name}](skills/{name}.mp4) | {row['seconds']:.1f} 秒 |"
        )
        x, y = (i % 5) * 320, (i // 5) * 238
        sheet.paste(Image.open(preview).resize((320, 200)), (x, y))
        draw.text((x + 5, y + 202), spec.label, font=font, fill=(30, 43, 58))
    cube = Path(a.cube)
    shutil.copy2(cube / "cube_grasp.mp4", out / "cube_grasp.mp4")
    shutil.copy2(cube / "cube_grasp.png", out / "cube_grasp.png")
    shutil.copy2(
        cube / "cube_grasp_verification.json", out / "evidence" / "cube_grasp.json"
    )
    cube_audit = video_audit(out / "cube_grasp.mp4")
    assert json.loads((cube / "cube_grasp_verification.json").read_text())["success"]
    sheet.save(out / "contact_sheet.jpg", quality=93)
    index += [
        "",
        "小方块完整抓取示例串联接近、闭爪和抬升三个原子技能，不计为新的原子技能。",
        "",
        "[30 个技能缩略图](contact_sheet.jpg) · [文件与执行检查记录](audit.json)",
        "",
    ]
    (out / "INDEX.md").write_text("\n".join(index), encoding="utf-8")
    audit = dict(
        skill_count=len(rows),
        all_skills_passed=True,
        all_overlays_read_only=True,
        cube_example=cube_audit,
        skills=[
            {
                k: v
                for k, v in rows[name].items()
                if k not in ("actual_trace", "planned_path", "result_steps")
            }
            for name in CATALOG
        ],
    )
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    guide = Path("docs/standalone-skills.md")
    if guide.exists():
        shutil.copy2(guide, out / "IMPLEMENTATION.md")
    print(
        json.dumps(
            dict(
                skill_videos=len(rows),
                bonus_videos=1,
                total_skill_seconds=sum(r["seconds"] for r in rows.values()),
                out=str(out),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
