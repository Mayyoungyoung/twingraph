"""Shared stage feasibility and balanced within-layout ranking.

Linear is an explicit no-sharing baseline. Shared and graph heads receive
gradients from both observed-stage supervision and whole-plan ranking.
"""
import numpy as np
import torch
from torch import nn
from .graph_value_v12 import VALUE_RELATIONS

MODEL_KINDS = ("linear", "shared", "graph")
LOSS_CONFIG = dict(ranking_weight=1., local_weight=.25, classification="layout_mean_bce",
                   ranking="equal_positive_negative_within_layout_pairs",
                   local="equal_observed_stage_class_means", linear_local_weight=0.)


def collate(rows, device="cpu"):
    return {k: torch.as_tensor(np.stack([r[k] for r in rows]), device=device)
            for k in ("x", "relations", "active")}


class StageValueNet(nn.Module):
    def __init__(self, input_dim, width=48, relation_types=None, kind="graph"):
        super().__init__()
        if kind not in MODEL_KINDS:
            raise ValueError(f"unknown value model kind: {kind}")
        relation_types = len(VALUE_RELATIONS) if relation_types is None else relation_types
        self.kind = kind
        self.config = dict(input_dim=input_dim, width=width, relation_types=relation_types, kind=kind)
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("scale", torch.ones(input_dim))
        if kind == "linear":
            self.plan = nn.Linear(8*input_dim, 1)
            self.local = nn.Linear(input_dim, 1)
        else:
            self.embed = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.LayerNorm(width))
            if kind == "graph":
                # Each relation has incoming and outgoing transforms: later
                # requirements can condition early choices in the same pass.
                self.messages = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(2*relation_types)])
                self.update = nn.Sequential(nn.Linear(2*width, width), nn.GELU(), nn.LayerNorm(width))
            self.local = nn.Sequential(nn.Linear(width, width//2), nn.GELU(), nn.Linear(width//2, 1))
            self.plan = nn.Sequential(nn.Linear(8*width+16, width), nn.GELU(), nn.Dropout(.1), nn.Linear(width, 1))

    def forward(self, b):
        # Scale is fitted only to train data, with a floor in normalized units.
        # Clipping keeps constant-but-new ports finite without masking them.
        x = ((b["x"]-self.mean)/self.scale).clamp(-8., 8.)
        if self.kind == "linear":
            return dict(plan_logit=self.plan(x.flatten(1)).squeeze(-1), local_logits=self.local(x).squeeze(-1))
        h = self.embed(x)
        if self.kind == "graph":
            messages = torch.zeros_like(h)
            count = self.config["relation_types"]
            if b["relations"].shape[1] != count:
                raise ValueError("checkpoint relation count differs from encoded graph")
            for r in range(count):
                for direction in range(2):
                    a = b["relations"][:, r]
                    if direction == 0:
                        a = a.transpose(-1, -2)
                    a = a/a.sum(-1, keepdim=True).clamp_min(1.)
                    messages = messages + a@self.messages[2*r+direction](h)
            h = h+self.update(torch.cat((h, messages/len(self.messages)), -1))
        local = self.local(h).squeeze(-1)
        # The whole-plan head can learn dependence among stages, instead of
        # multiplying correlated local probabilities as independent events.
        z = torch.cat((h.flatten(1), local.sigmoid()*b["active"], b["active"]), -1)
        return dict(plan_logit=self.plan(z).squeeze(-1), local_logits=local)


def supervised_loss(pred, success, local_y, local_mask, ranking_weight=1., local_weight=.25):
    """One layout per call; no artificial cross-layout positive/negative pairs.

    The pair mean gives every positive and every negative equal participation,
    irrespective of class prevalence. Local losses first balance each observed
    stage's positive/negative classes, then average observed stages equally.
    The calibrated whole-plan BCE retains actual prevalence; ranking supplies
    the imbalance-resistant signal without turning probabilities into priors
    for a fictitious balanced deployment distribution.
    """
    full = nn.functional.binary_cross_entropy_with_logits(pred["plan_logit"], success)
    delta = success[:, None]-success[None, :]
    pairs = delta > .05
    difference = pred["plan_logit"][:, None]-pred["plan_logit"][None, :]
    ranking = (nn.functional.softplus(-difference[pairs])*delta[pairs]).mean() if pairs.any() else full*0.
    raw = nn.functional.binary_cross_entropy_with_logits(pred["local_logits"], local_y, reduction="none")
    weights = torch.stack((local_mask*local_y, local_mask*(1-local_y)))
    counts = weights.sum(1)
    class_means = (raw.unsqueeze(0)*weights).sum(1)/counts.clamp_min(1.e-8)
    observed_classes = (counts > 0).to(raw.dtype)
    stage_losses = class_means.sum(0)/observed_classes.sum(0).clamp_min(1.)
    local = stage_losses.sum()/(observed_classes.sum(0) > 0).sum().clamp_min(1.)
    return full + ranking_weight*ranking + local_weight*local
