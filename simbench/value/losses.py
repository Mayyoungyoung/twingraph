"""Binomial probability supervision plus optional near-optimal Top-K retention."""
import torch
import torch.nn.functional as F


def probability_loss(output, trials, prefix_successes, full_successes):
    n=trials.float()
    a=prefix_successes.float()
    b=full_successes.float()
    if torch.any((n<=0)|(a<0)|(b<0)|(a>n)|(b>a)):
        raise ValueError("inconsistent rollout counts")
    la=F.binary_cross_entropy_with_logits(output["prefix_logit"],a/n,reduction="none")
    lb=F.binary_cross_entropy_with_logits(output["suffix_logit"],b/a.clamp_min(1),reduction="none")
    # Prefix failures do not supervise the conditional suffix head.
    return (la*n).sum()/n.sum()+(lb*a).sum()/a.sum().clamp_min(1)


def keep_loss(scores, reference, group_ids, k=2, epsilon=.1, margin=.1):
    if k<1:
        raise ValueError("k must be positive")
    losses=[]
    for group in torch.unique(group_ids):
        mask=group_ids==group
        truth=reference[mask]
        z=scores[mask]
        if truth.max()<=0:
            continue
        good=truth>=truth.max()-epsilon
        bad=z[~good]
        if len(bad)>=k:
            losses.append(F.relu(margin+bad.topk(k).values[-1]-z[good].max()))
    return torch.stack(losses).mean() if losses else scores.sum()*0

