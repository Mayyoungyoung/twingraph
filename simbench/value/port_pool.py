"""Small set baseline generated from typed graph ports, not product features.

Vocabulary is fitted on training inputs only. Numerical columns are generic
typed-value channels and counts; names/IDs and geometric ranking proxies are
absent. Pooling sacrifices exact order; the graph Transformer retains it.
"""
import numpy as np
import torch
from torch import nn
from .encode import bucket,NUMERIC


def entries(encoded):
    # The first token of each node declares its context/skill type.
    owners=encoded["owner"];types={}
    for i,node in enumerate(owners):
        types.setdefault(int(node),int(encoded["categories"][i]))
    values={}
    for i,node in enumerate(owners):
        key=f"{types[int(node)]}:{encoded['keys'][i]}:{encoded['statuses'][i]}"
        category=int(encoded["categories"][i])
        if category:
            key+=f":cat{category}"
            v=np.zeros(NUMERIC+1,np.float32);v[-1]=1
        else:v=np.r_[encoded["numbers"][i],np.float32(1)]
        values.setdefault(key,[]).append(v)
    return {key:np.r_[np.mean(v,axis=0)[:NUMERIC],np.log1p(len(v))].astype(np.float32) for key,v in values.items()}


def fit_vocabulary(encoded):
    return sorted({key for row in encoded for key in entries(row)})


def vectorize(encoded,vocabulary):
    index={key:i for i,key in enumerate(vocabulary)};types={};indices=[];rows=[]
    for i,node in enumerate(encoded["owner"]):
        types.setdefault(int(node),int(encoded["categories"][i]))
        key=f"{types[int(node)]}:{encoded['keys'][i]}:{encoded['statuses'][i]}"
        category=int(encoded["categories"][i])
        if category:key+=f":cat{category}"
        if key in index:indices.append(index[key]);rows.append(i)
    count=np.zeros(len(vocabulary),np.float32);total=np.zeros((len(vocabulary),NUMERIC),np.float32)
    np.add.at(count,indices,1.)
    np.add.at(total,indices,encoded["numbers"][rows])
    return np.c_[total/np.maximum(count[:,None],1),np.log1p(count)].astype(np.float32).ravel()


class PortPoolNet(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.register_buffer("mean",torch.zeros(dim));self.register_buffer("std",torch.ones(dim))
        self.net=nn.Sequential(nn.Linear(dim,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))

    def forward(self,x):return self.net((x-self.mean)/self.std).squeeze(-1)
