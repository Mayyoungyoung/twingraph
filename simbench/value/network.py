"""Shared variable-length scene/plan Transformer with conditional value heads."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from .encode import VOCAB, NUMERIC


@dataclass
class ModelConfig:
    width: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.1
    vision: bool = True


class PlanValueNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.key = nn.Embedding(VOCAB, c.width, padding_idx=0)
        self.category = nn.Embedding(VOCAB, c.width, padding_idx=0)
        self.status = nn.Embedding(4, c.width, padding_idx=0)
        self.stage = nn.Embedding(4, c.width)
        self.number = nn.Sequential(
            nn.Linear(NUMERIC, c.width), nn.GELU(), nn.Linear(c.width, c.width)
        )
        self.visual = nn.Linear(512, c.width) if c.vision else None
        self.visual_type = nn.Parameter(torch.zeros(1, 1, c.width))
        self.cls = nn.Parameter(torch.randn(1, 1, c.width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            c.width,
            c.heads,
            c.width * 4,
            c.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, c.layers, norm=nn.LayerNorm(c.width), enable_nested_tensor=False
        )
        # TransformerEncoder clones initial weights; reinitialize independently.
        for module in self.encoder.layers:
            for parameter in module.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.input_norm = nn.LayerNorm(c.width)
        self.prefix_head = nn.Linear(c.width, 1)
        self.suffix_head = nn.Linear(c.width, 1)
        self.direct_head = nn.Linear(c.width, 1)

    def forward(self, batch):
        pos = batch["positions"].float().unsqueeze(-1)
        frequency = torch.exp(
            torch.arange(0, self.config.width, 2, device=pos.device)
            * (-math.log(10000.0) / self.config.width)
        )
        pe = torch.zeros((*pos.shape[:2], self.config.width), device=pos.device)
        pe[:, :, 0::2] = torch.sin(pos * frequency)
        pe[:, :, 1::2] = torch.cos(pos * frequency)
        # Context/object sets have no positional meaning.
        pe = pe * (batch["positions"] != 0).unsqueeze(-1)
        x = (
            self.key(batch["keys"])
            + self.category(batch["categories"])
            + self.status(batch["statuses"])
            + self.stage(batch["stages"])
            + self.number(batch["numbers"])
            + pe
        )
        padding = batch["padding"]
        prefix_padding = padding | (batch["stages"] >= 2)
        n = x.shape[0]
        if self.visual is not None:
            if "visual" not in batch:
                raise ValueError("checkpoint requires frozen visual features")
            v = self.visual(batch["visual"]) + self.visual_type
            x = torch.cat([v, x], 1)
            padding = torch.cat([batch["visual_padding"], padding], 1)
            prefix_padding = torch.cat([batch["visual_padding"], prefix_padding], 1)
        x = torch.cat([self.cls.expand(n, -1, -1), self.input_norm(x)], 1)
        false = torch.zeros((n, 1), device=x.device, dtype=torch.bool)
        all_mask = torch.cat([false, padding], 1)
        prefix_mask = torch.cat([false, prefix_padding], 1)
        # Two views share weights. The prefix head cannot attend to a suffix
        # directly OR indirectly via updated context tokens from another layer.
        full = self.encoder(x, src_key_padding_mask=all_mask)[:, 0]
        local = self.encoder(x, src_key_padding_mask=prefix_mask)[:, 0]
        a = self.prefix_head(local).squeeze(-1)
        b = self.suffix_head(full).squeeze(-1)
        direct = self.direct_head(full).squeeze(-1)
        return dict(
            prefix_logit=a,
            suffix_logit=b,
            direct_logit=direct,
            p_prefix=a.sigmoid(),
            p_suffix=b.sigmoid(),
            q=a.sigmoid() * b.sigmoid(),
        )
