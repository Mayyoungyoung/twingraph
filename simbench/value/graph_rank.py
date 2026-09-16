"""Deployable score(graph) and complete Top-K graph/PlanIR export."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .plan import digest
from .skill_graph import compile_graph,validate_graph
from .graph_encode import encode_graphs,collate_graph
from .graph_learning import load_model
from .collect import dump
from .port_pool import vectorize


class GraphScorer:
    def __init__(self, checkpoint, device="cpu"):
        self.device=device
        t=time.perf_counter();self.model,self.kind,self.saved=load_model(checkpoint,device)
        if self.kind not in {"graph","sequence","port_mlp"}:raise ValueError("GraphScorer requires graph-derived input checkpoint")
        self.checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.sync();self.load_seconds=time.perf_counter()-t
        self.inference_calls=0

    def sync(self):
        if str(self.device).startswith("cuda"):torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def rank(self, graphs, k=4):
        if not graphs or not 1<=k<=len(graphs):raise ValueError("invalid K/pool")
        started=time.perf_counter()
        plans=[validate_graph(g) for g in graphs]
        if len({p.id for p in plans})!=len(plans):raise ValueError("duplicate candidate")
        checks=time.perf_counter()-started
        t=time.perf_counter();encoded=encode_graphs(graphs,check=False);encoding=time.perf_counter()-t
        t=time.perf_counter()
        vectors=np.stack([vectorize(e,self.saved["vocabulary"]) for e in encoded]) if self.kind=="port_mlp" else None
        encoding+=time.perf_counter()-t
        self.sync();t=time.perf_counter();scores=[]
        for i in range(0,len(encoded),32):
            batch=torch.as_tensor(vectors[i:i+32],device=self.device) if vectors is not None else collate_graph(encoded[i:i+32],self.device)
            scores.extend(self.model(batch).sigmoid().cpu().tolist())
        self.sync();inference=time.perf_counter()-t
        self.inference_calls+=1
        t=time.perf_counter();order=np.argsort(-np.asarray(scores),kind="stable")
        ranked=[dict(candidate_id=plans[i].id,score=scores[i],plan=plans[i].to_dict(),
                     input_graph_sha256=digest(graphs[i])) for i in order]
        top=[dict(**ranked[j],graph=graphs[order[j]]) for j in range(k)]
        export=time.perf_counter()-t
        return dict(schema="twingraph.graph_ranking.v1",n=len(graphs),k=k,ranked=ranked,top_k=top,
                    checkpoint_sha256=self.checkpoint_sha256,
                    timings=dict(graph_integrity_seconds=checks,typed_encoding_seconds=encoding,
                                 network_and_batch_seconds=inference,sort_export_seconds=export,
                                 call_total_seconds=time.perf_counter()-started,
                                 first_inference_after_load=self.inference_calls==1),
                    meaning="full suffix completion estimate, not a feasibility certificate")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--inputs",required=True)
    p.add_argument("--checkpoint",required=True);p.add_argument("--out",required=True)
    p.add_argument("--device",default="cpu");p.add_argument("--k",type=int,default=4);a=p.parse_args()
    inp=json.loads(Path(a.inputs).read_text());start=time.perf_counter()
    graphs=[compile_graph(inp["observation"],p) for p in inp["candidates"]]
    construction=time.perf_counter()-start
    scorer=GraphScorer(a.checkpoint,a.device);result=scorer.rank(graphs,a.k)
    result["timings"].update(graph_construction_seconds=construction,model_load_seconds=scorer.load_seconds)
    dump(a.out,result)
    print(json.dumps({k:result[k] for k in ("n","k","checkpoint_sha256","timings")}))


if __name__=="__main__":main()
