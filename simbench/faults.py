"""Fault-injection model for simbench task runs.

``FailureModel`` is the single stochastic knob of a run: every failure
source (grasp-pose estimation error, detection miss/false positive,
path-planning unreachable/collision, move-to-pose residual, grasp slip,
incoming-part scatter) is sampled from its own ``np.random.default_rng``
drawn by (profile, seed).  Two runs with the same seed are bit-identical;
different seeds produce the realistic success/failure mix of the task.

Profiles scale the sources between the deterministic baseline (``none``)
and a deliberately degraded run (``strong``).  Every sampled event is
appended to ``log`` (JSON-safe) and counted per source, so a failed run
can always be attributed: the executor forwards the log to the
DataCollector (fault_log.json) and the run summary prints the counts.
"""
import numpy as np

# probability / magnitude defaults per profile.  ``default`` is tuned so
# that a ~70-step Task A run shows a mix of clean and failed seeds while
# the nominal physical chain (the ``none`` profile) stays untouched.
PROFILES = {
    # name: noise_std(m)  p_miss  p_false  p_unreach  p_coll  move_std(m)  p_slip  init_jitter(m)
    "none":    dict(noise_std=0.0,    p_miss=0.0,   p_false=0.0,
                    p_unreachable=0.0, p_collision=0.0,
                    move_std=0.0,     p_slip=0.0,   init_jitter=0.0),
    "mild":    dict(noise_std=0.0008, p_miss=0.02,  p_false=0.015,
                    p_unreachable=0.01, p_collision=0.015,
                    move_std=0.0004,  p_slip=0.02,  init_jitter=0.0005),
    "default": dict(noise_std=0.0015, p_miss=0.04,  p_false=0.03,
                    p_unreachable=0.02, p_collision=0.03,
                    move_std=0.0008,  p_slip=0.04,  init_jitter=0.001),
    "strong":  dict(noise_std=0.003,  p_miss=0.08,  p_false=0.06,
                    p_unreachable=0.05, p_collision=0.07,
                    move_std=0.0015,  p_slip=0.10,  init_jitter=0.002),
}

SOURCE_NAMES = {
    "detect_miss": "检测漏检",
    "detect_false": "检测误检",
    "detect_outlier": "检测离群误差",
    "plan_unreachable": "路径规划不可达",
    "plan_collision": "路径规划碰撞",
    "grasp_est_fail": "抓取位姿估计失败",
    "slip": "抓取滑脱",
}


class FailureModel:
    """Seeded sampling of every fault source + an attribution log."""

    def __init__(self, profile="default", seed=0, noise_override=None):
        if profile not in PROFILES:
            raise ValueError(f"unknown fault profile {profile!r}; "
                             f"known: {sorted(PROFILES)}")
        self.profile = profile
        self.seed = seed
        self.params = dict(PROFILES[profile])
        if noise_override is not None and noise_override > 0.0:
            # the legacy --noise CLI knob maps onto the estimation-noise
            # source only (kept for backward compatibility)
            self.params["noise_std"] = float(noise_override)
        self.rng = np.random.default_rng(seed)
        self.log = []        # list of {"source", "detail"} events
        self.counts = {}     # source -> occurrence count

    # ------------------------------------------------------------- logging
    def record(self, source, detail):
        """Append an attributed event (also usable by the executor for
        events this class does not sample directly)."""
        self.log.append({"source": source, "detail": str(detail)})
        self.counts[source] = self.counts.get(source, 0) + 1

    # ------------------------------------------------------------ sources
    def detect_outcome(self, step=None):
        """Per-detection outcome: 'ok', 'miss' or 'false'.

        'miss' = the part was not found (re-detection retries).
        'false' = the detector locked onto a wrong feature; the caller
        adds ``false_offset()`` to the reported position.
        """
        p_miss = self.params["p_miss"]
        p_false = self.params["p_false"]
        r = self.rng.random()
        if r < p_miss:
            self.record("detect_miss", f"step {step}: part not found")
            return "miss"
        if r < p_miss + p_false:
            self.record("detect_false",
                        f"step {step}: false feature lock")
            return "false"
        return "ok"

    def detect_noise(self):
        """(pos_noise(3), yaw_noise) -- the grasp-pose estimation error.

        Position std in m; yaw std x10 (rad), matching the old
        perception.detect semantics (3 mm error ~ 1.7 deg heading).
        A draw beyond 2 sigma is recorded as a 'detect_outlier' event
        so noise-driven downstream failures stay attributable."""
        s = self.params["noise_std"]
        if s <= 0.0:
            return np.zeros(3), 0.0
        n = self.rng.normal(0.0, s, 3)
        if np.linalg.norm(n) > 2.0 * s:
            self.record("detect_outlier",
                        f"estimation noise draw "
                        f"{np.linalg.norm(n) * 1000:.1f}mm > 2 sigma")
        return n, float(self.rng.normal(0.0, 10.0 * s))

    def false_offset(self):
        """Random 2-6 mm xy offset (the wrong-feature lock)."""
        ang = self.rng.uniform(0.0, 2.0 * np.pi)
        mag = self.rng.uniform(0.002, 0.006)
        return np.array([mag * np.cos(ang), mag * np.sin(ang), 0.0])

    def plan_path_outcome(self, step=None):
        """Per path-plan outcome: 'ok', 'unreachable' or 'collision'.

        'collision' here is the planner's own conservative prediction
        (the real geometric check runs separately in skills.planning);
        both failures trigger one replan with a fallback style."""
        p_u = self.params["p_unreachable"]
        p_c = self.params["p_collision"]
        r = self.rng.random()
        if r < p_u:
            self.record("plan_unreachable",
                        f"step {step}: goal unreachable")
            return "unreachable"
        if r < p_u + p_c:
            self.record("plan_collision",
                        f"step {step}: predicted collision")
            return "collision"
        return "ok"

    def move_error(self):
        """Move-to-pose residual (m), added to the final waypoint so the
        executed approach lands measurably off the planned pose."""
        s = self.params["move_std"]
        if s <= 0.0:
            return np.zeros(3)
        return self.rng.normal(0.0, s, 3)

    def slip_event(self, step=None):
        """Grasp-slip event during a carry: the executor weakens the
        finger bite, so the drop (if any) is real physics, not a flag."""
        if self.rng.random() < self.params["p_slip"]:
            self.record("slip", f"step {step}: weakened grip during "
                                f"carry")
            return True
        return False

    def init_jitter(self, n=2):
        """Incoming-part scatter (m) for the ``scatter_parts`` step."""
        s = self.params["init_jitter"]
        if s <= 0.0:
            return np.zeros(n)
        return self.rng.normal(0.0, s, n)

    def measure_noise(self):
        """Sensor noise on an inspection measurement (m, 3-vector)."""
        s = self.params["noise_std"]
        if s <= 0.0:
            return np.zeros(3)
        return self.rng.normal(0.0, s, 3)

    # ------------------------------------------------------------ summary
    def summary(self):
        """One-line per-source occurrence summary (Chinese labels)."""
        if not self.counts:
            return "no fault events"
        parts = []
        for src, n in sorted(self.counts.items()):
            label = SOURCE_NAMES.get(src, src)
            parts.append(f"{label}x{n}")
        return ", ".join(parts)
