"""A square table, Panda, passive parts and a bolted guide base. No fixture actuators."""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

HERE = Path(__file__).resolve().parent
SCENE = HERE / "tabletop.xml"
CENTER = np.array([0.075, 0.085])
TABLE_Z = 0.8


def fmt(values):
    return " ".join(f"{float(x):.7g}" for x in values)


def geom(parent, name, pos, size, color, kind="box", **kw):
    attrs = dict(
        name=name,
        type=kind,
        pos=fmt(pos),
        size=fmt(size),
        rgba=color,
        friction=".45 .01 .0001",
        condim="4",
        solref=".008 1",
        solimp=".95 .99 .001",
        density="700",
    )
    attrs.update({k: str(v) for k, v in kw.items()})
    return ET.SubElement(parent, "geom", attrs)


def ring(parent, prefix, center, inner, outer, half, color, n=20, **kw):
    # Tangential boxes form an actual through-hole, not a convex filled mesh.
    radius = (inner + outer) / 2
    tangent = outer * np.tan(np.pi / n)
    for i in range(n):
        angle = 2 * np.pi * i / n
        p = np.array(center) + [radius * np.cos(angle), radius * np.sin(angle), 0]
        geom(
            parent,
            f"{prefix}_{i}",
            p,
            [(outer - inner) / 2, tangent * 1.02, half],
            color,
            quat=fmt([np.cos(angle / 2), 0, 0, np.sin(angle / 2)]),
            **kw,
        )


def holed_plate(parent, prefix, halfxy, z, halfz, holes, color, holehalf=0.004):
    xs = sorted(
        set(
            [-halfxy[0], halfxy[0]]
            + [x + d for x, y in holes for d in (-holehalf, holehalf)]
        )
    )
    ys = sorted(
        set(
            [-halfxy[1], halfxy[1]]
            + [y + d for x, y in holes for d in (-holehalf, holehalf)]
        )
    )
    for i, (a, b) in enumerate(zip(xs[:-1], xs[1:])):
        for j, (c, d) in enumerate(zip(ys[:-1], ys[1:])):
            x, y = (a + b) / 2, (c + d) / 2
            if any(
                abs(x - hx) < holehalf and abs(y - hy) < holehalf for hx, hy in holes
            ):
                continue
            geom(
                parent,
                f"{prefix}_{i}_{j}",
                [x, y, z],
                [(b - a) / 2, (d - c) / 2, halfz],
                color,
            )


def build():
    root = ET.Element("mujoco", model="robot_only_sliding_stage")
    ET.SubElement(
        root, "compiler", angle="radian", meshdir="../assets/panda", autolimits="true"
    )
    ET.SubElement(
        root, "option", timestep=".002", cone="elliptic", impratio="10", iterations="80"
    )
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="1600", offheight="1000")
    ET.SubElement(vis, "quality", shadowsize="2048", offsamples="4")
    ET.SubElement(
        vis, "headlight", ambient=".3 .3 .3", diffuse=".45 .45 .45", specular=".1 .1 .1"
    )
    ET.SubElement(vis, "map", znear=".01")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1=".55 .59 .63",
        rgb2=".77 .79 .81",
        width="512",
        height="3072",
    )
    ET.SubElement(root, "include", file="../assets/panda/panda.xml")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        pos="-.3 -1 2.6",
        dir=".2 .4 -1",
        directional="true",
        diffuse=".6 .6 .6",
    )
    geom(
        world,
        "floor",
        [0, 0, -0.03],
        [3, 3, 0.03],
        ".46 .48 .5 1",
        contype="0",
        conaffinity="0",
    )
    geom(
        world,
        "table",
        [-0.21, 0, 0.775],
        [0.65, 0.65, 0.025],
        ".64 .58 .48 1",
        density="1000",
    )
    for x in (-0.79, 0.37):
        for y in (-0.58, 0.58):
            geom(
                world,
                f"leg_{x}_{y}",
                [x, y, 0.375],
                [0.03, 0.03, 0.375],
                ".28 .28 .29 1",
            )
    # This fixed base is the first product component, bolted to a passive assembly nest.
    base = ET.SubElement(world, "body", name="guide_base", pos=fmt([*CENTER, 0.812]))
    holes = [(-0.092, -0.032), (-0.092, 0.032)]
    holed_plate(base, "base", (0.118, 0.049), 0, 0.006, holes, ".28 .31 .35 1")
    for sign in (-1, 1):
        geom(
            base,
            f"rail_{sign}",
            [0.020, sign * 0.034, 0.012],
            [0.089, 0.007, 0.012],
            ".65 .68 .71 1",
            friction=".2 .005 .0001",
        )
        geom(
            base,
            f"lip_{sign}",
            [0.020, sign * 0.027, 0.023],
            [0.089, 0.006, 0.003],
            ".73 .75 .77 1",
            friction=".2 .005 .0001",
        )
    geom(base, "rear_stop", [0.108, 0, 0.014], [0.009, 0.026, 0.014], ".27 .30 .34 1")
    # Passive corner locating posts register the end stop independently of the
    # clearance pins. The central carriage insertion corridor stays open.
    for x in (-0.10723, -0.07677):
        for y in (-0.04623, 0.04623):
            geom(
                base,
                f"stop_locator_{x}_{y}",
                [x, y, 0.008],
                [0.004, 0.009],
                ".5 .53 .55 1",
                "cylinder",
            )
    # Loose carriage with a narrow raised grasp boss and a handle locating post.
    slider = ET.SubElement(world, "body", name="carriage", pos="-.25 -.20 .806")
    ET.SubElement(slider, "freejoint", name="carriage_free")
    geom(
        slider,
        "carriage_shoe",
        [0, 0, 0],
        [0.027, 0.023, 0.006],
        ".25 .45 .64 1",
        friction=".25 .005 .0001",
    )
    geom(
        slider,
        "carriage_boss",
        [0, 0, 0.025],
        [0.022, 0.014, 0.015],
        ".26 .49 .70 1",
        friction="1 .02 .001",
    )
    geom(
        slider,
        "handle_post",
        [0, 0, 0.050],
        [0.0055, 0.012],
        ".65 .67 .7 1",
        "cylinder",
    )
    cap = ET.SubElement(world, "body", name="end_stop", pos="-.12 -.25 .818")
    ET.SubElement(cap, "freejoint", name="end_stop_free")
    holed_plate(
        cap,
        "stop",
        (0.012, 0.043),
        0,
        0.018,
        [(0, -0.032), (0, 0.032)],
        ".74 .45 .22 1",
    )
    geom(
        cap,
        "stop_boss",
        [0, 0, 0.023],
        [0.011, 0.013, 0.008],
        ".78 .48 .24 1",
        friction="1 .02 .001",
    )
    handle = ET.SubElement(world, "body", name="handle", pos=".015 -.25 .808")
    ET.SubElement(handle, "freejoint", name="handle_free")
    ring(
        handle,
        "handle_ring",
        [0, 0, 0],
        0.0061,
        0.021,
        0.008,
        ".21 .23 .26 1",
        density="500",
        friction="1 .02 .001",
    )
    for name, y in [("pin_left", -0.22), ("pin_right", -0.34)]:
        p = [-0.36, y, 0.850]
        pin = ET.SubElement(world, "body", name=name, pos=fmt(p))
        ET.SubElement(pin, "freejoint", name=name + "_free")
        geom(
            pin,
            name + "_shaft",
            [0, 0, -0.0205],
            [0.0033, 0.0265],
            ".7 .72 .74 1",
            "cylinder",
            density="2200",
        )
        geom(
            pin,
            name + "_tip",
            [0, 0, -0.047],
            [0.0027, 0.001],
            ".7 .72 .74 1",
            "cylinder",
            density="2200",
        )
        geom(
            pin,
            name + "_head",
            [0, 0, 0.006],
            [0.009, 0.007],
            ".78 .62 .2 1",
            "cylinder",
            friction="1 .02 .001",
            density="900",
        )
        support = ET.SubElement(
            world, "body", name=name + "_holder", pos=fmt([p[0], p[1], 0.819])
        )
        ring(
            support,
            name + "_holder",
            [0, 0, 0],
            0.0044,
            0.010,
            0.019,
            ".44 .44 .43 1",
            n=12,
        )
    # Fixed camera: robot, all supply parts and the complete product remain visible.
    eye = np.array([0.94, -1.38, 1.98])
    target = np.array([-0.20, -0.01, 1.02])
    z = (eye - target) / np.linalg.norm(eye - target)
    x = np.cross([0, 0, 1], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(
        world,
        "camera",
        name="task_view",
        pos=fmt(eye),
        xyaxes=fmt(np.r_[x, y]),
        fovy="36",
    )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(SCENE, encoding="unicode")
    return SCENE


if __name__ == "__main__":
    print(build())
