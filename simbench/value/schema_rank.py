"""Deployable complete-plan graph scoring; no outcome files are read."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .collect import dump
from .graph_encode import encode_graphs, collate_graph
from .plan import digest
from .port_pool import vectorize, entries
from .schema_learning import load_model, sigmoid
from .skill_graph import compile_graph, validate_graph


class SchemaScorer:
    def __init__(self, checkpoint, device="cpu"):
        started = time.perf_counter()
        self.device = device
        self.model, self.saved = load_model(checkpoint, device)
        self.checkpoint_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.sync()
        self.load_seconds = time.perf_counter()-started

    def sync(self):
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def rank(self, graphs, k=4, calibrated=True):
        if not graphs or not 1 <= k <= len(graphs):
            raise ValueError("invalid candidate pool or K")
        started = time.perf_counter()
        plans = [validate_graph(g) for g in graphs]
        if len({p.id for p in plans}) != len(plans):
            raise ValueError("duplicate candidate identities")
        if len({digest(g["observation"]) for g in graphs}) != 1:
            raise ValueError("candidate pool must share the same observed task")
        check_seconds = time.perf_counter()-started
        t = time.perf_counter()
        encoded = encode_graphs(graphs, check=False)
        vocabulary = self.saved.get("vocabulary")
        x = None
        unseen = []
        if vocabulary is not None:
            x = np.stack([vectorize(e, vocabulary)[self.saved["columns"]] for e in encoded])
            known = set(vocabulary)
            unseen = [len(set(entries(e)) - known) for e in encoded]
        encoding_seconds = time.perf_counter()-t
        self.sync(); t = time.perf_counter()
        z = []
        for i in range(0, len(graphs), 8):
            batch = torch.as_tensor(x[i:i+8], device=self.device) if x is not None else collate_graph(encoded[i:i+8], self.device)
            z.extend(self.model(batch).cpu().tolist())
        self.sync(); inference_seconds = time.perf_counter()-t
        t = time.perf_counter()
        c = self.saved.get("calibration", dict(scale=1., bias=0.)) if calibrated else dict(scale=1., bias=0.)
        scores = sigmoid(c["scale"]*np.asarray(z)+c["bias"])
        # Sort logits to preserve rank even when sigmoid reaches finite precision limits.
        order = np.argsort(-np.asarray(z), kind="stable")
        output = dict(schema="twingraph.topk.schema.v1", checkpoint_sha256=self.checkpoint_sha256,
            kind=self.saved["kind"], calibrated=calibrated, scores=scores.tolist(), logits=z,
            order=[plans[i].id for i in order], top_k=[dict(candidate_id=plans[i].id,
                score=float(scores[i]), graph_sha256=digest(graphs[i]), graph=graphs[i], plan=plans[i].to_dict()) for i in order[:k]],
            unseen_vocabulary_entries=unseen,
            seconds=dict(integrity=check_seconds, encoding=encoding_seconds, inference=inference_seconds))
        output["seconds"]["export"] = time.perf_counter()-t
        output["seconds"]["total"] = time.perf_counter()-started
        return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True); p.add_argument("--k", type=int, default=4)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    inp = json.loads(Path(a.inputs).read_text())
    graphs = [compile_graph(inp["observation"], p) for p in inp["candidates"]]
    dump(a.out, SchemaScorer(a.checkpoint, a.device).rank(graphs, a.k))


if __name__ == "__main__":
    main()
