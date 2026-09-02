"""Adaptive settle skill: step until the physics is quiet.

Replaces the old fixed ``settle(60)`` wait between actions (a major
source of the slow demos): the sim stops as soon as
``max |qvel|`` drops below the threshold, capped at ``max_steps``.
"""
import numpy as np


def settle(ctx, thresh=2e-3, max_steps=200, verbose=False):
    """Step until quiet; returns the number of control steps used."""
    for i in range(max_steps):
        ctx.step()
        if float(np.max(np.abs(ctx.data.qvel))) < thresh:
            if verbose:
                print(f"  [settle] quiet after {i + 1} steps")
            return i + 1
    if verbose:
        print(f"  [settle] capped at {max_steps} steps "
              f"(max|qvel|={float(np.max(np.abs(ctx.data.qvel))):.2e})")
    return max_steps
