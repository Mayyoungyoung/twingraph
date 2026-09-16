"""Single-head value network over shared typed execution ports.

PIGINet-inspired typed fusion, compact call-level pooling, optional soft relation
bias. The no-relation control has identical inputs, width, depth and parameters.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .encode import VOCAB,NUMERIC
from .skill_graph import RELATIONS


@dataclass
class GraphConfig:
    width:int=64
    layers:int=2
    heads:int=4
    dropout:float=.1
    relations:bool=True


class RelationLayer(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c
        self.norm1=nn.LayerNorm(c.width);self.norm2=nn.LayerNorm(c.width)
        self.qkv=nn.Linear(c.width,3*c.width);self.out=nn.Linear(c.width,c.width)
        self.bias=nn.Linear(2*len(RELATIONS),c.heads,bias=False)
        nn.init.zeros_(self.bias.weight)
        self.ff=nn.Sequential(nn.Linear(c.width,4*c.width),nn.GELU(),nn.Dropout(c.dropout),nn.Linear(4*c.width,c.width))
        self.drop=nn.Dropout(c.dropout)

    def forward(self,x,padding,relations):
        b,n,w=x.shape;h=self.c.heads;d=w//h
        q,k,v=self.qkv(self.norm1(x)).reshape(b,n,3,h,d).permute(2,0,3,1,4).unbind(0)
        score=q@k.transpose(-1,-2)/math.sqrt(d)
        if self.c.relations:score=score+self.bias(relations).permute(0,3,1,2)
        score=score.masked_fill(padding[:,None,None,:],float("-inf"))
        attended=self.drop(score.softmax(-1))@v
        x=x+self.drop(self.out(attended.transpose(1,2).reshape(b,n,w)))
        return x+self.drop(self.ff(self.norm2(x)))


class GraphValueNet(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or GraphConfig();c=self.config;w=c.width
        self.key=nn.Embedding(VOCAB,w,padding_idx=0);self.category=nn.Embedding(VOCAB,w,padding_idx=0)
        self.status=nn.Embedding(4,w,padding_idx=0)
        self.number=nn.Sequential(nn.Linear(NUMERIC,w),nn.GELU(),nn.Linear(w,w))
        self.port=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w),nn.GELU(),nn.Linear(w,w))
        self.norm=nn.LayerNorm(w);self.cls=nn.Parameter(torch.randn(1,1,w)*.02)
        self.layers=nn.ModuleList([RelationLayer(c) for _ in range(c.layers)])
        self.head=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,1))

    def forward(self,b):
        x=self.key(b["keys"])+self.category(b["categories"])+self.status(b["statuses"])+self.number(b["numbers"])
        x=self.port(x);valid=(~b["padding"]).unsqueeze(-1);x=x*valid
        batch,nodes=b["positions"].shape;w=self.config.width
        pooled=x.new_zeros((batch,nodes,w));counts=x.new_zeros((batch,nodes,1))
        pooled.scatter_add_(1,b["owner"].unsqueeze(-1).expand(-1,-1,w),x)
        counts.scatter_add_(1,b["owner"].unsqueeze(-1),valid.to(x.dtype))
        pooled=self.norm(pooled/counts.clamp_min(1))
        pos=b["positions"].float().unsqueeze(-1)
        freq=torch.exp(torch.arange(0,w,2,device=x.device)*(-math.log(10000.)/w))
        pe=torch.zeros_like(pooled);pe[:,:,0::2]=torch.sin(pos*freq);pe[:,:,1::2]=torch.cos(pos*freq)
        pooled=pooled+pe*(pos!=0)
        x=torch.cat([self.cls.expand(batch,-1,-1),pooled],1)
        padding=torch.cat([torch.zeros(batch,1,dtype=torch.bool,device=x.device),b["node_padding"]],1)
        rel=torch.nn.functional.pad(b["relations"],(0,0,1,0,1,0))
        for layer in self.layers:x=layer(x,padding,rel)
        return self.head(x[:,0]).squeeze(-1)
