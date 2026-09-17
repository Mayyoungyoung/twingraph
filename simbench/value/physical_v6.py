"""Robust v6 execution wrapper with paired sensing and dynamics noise."""
import copy
import numpy as np

from .physical import PhysicalRunner


class RobustPhysicalRunner(PhysicalRunner):
    """Apply one fully recorded perturbation, paired across all candidates."""
    def run(self, plan, trial, keep_trace=False):
        required = {"perception_seed", "position_noise_std_m", "mass_scale", "damping_scale"}
        if not required <= set(trial):
            raise ValueError("v6 trial is missing sensing/dynamics perturbations")
        if not (0 <= trial["position_noise_std_m"] <= .005
                and .5 <= trial["mass_scale"] <= 1.5
                and .5 <= trial["damping_scale"] <= 1.5):
            raise ValueError("v6 trial perturbation is outside the supported envelope")
        old_noise = self.session.noise
        old_rng = copy.deepcopy(self.snapshot["rng"])
        old_mass = self.snapshot["physics"]["model"]["body_mass"].copy()
        old_damping = self.snapshot["physics"]["model"]["dof_damping"].copy()
        rng = np.random.default_rng(int(trial["perception_seed"]))
        self.snapshot["rng"] = copy.deepcopy(rng.bit_generator.state)
        self.snapshot["physics"]["model"]["body_mass"] = old_mass * float(trial["mass_scale"])
        self.snapshot["physics"]["model"]["dof_damping"] = old_damping * float(trial["damping_scale"])
        self.session.noise = float(trial["position_noise_std_m"])
        try:
            result = super().run(plan, trial, keep_trace=keep_trace)
            result["robustness_target"] = "full-program success under recorded sensing and dynamics draw"
            return result
        finally:
            self.session.noise = old_noise
            self.snapshot["rng"] = old_rng
            self.snapshot["physics"]["model"]["body_mass"] = old_mass
            self.snapshot["physics"]["model"]["dof_damping"] = old_damping

