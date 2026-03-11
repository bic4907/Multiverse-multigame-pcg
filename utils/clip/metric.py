import torch

def topk_accuracy(logits, labels, ks=(1, 5)):
    max_k = max(ks)
    topk = torch.topk(logits, k=max_k, dim=1).indices

    res = {}
    for k in ks:
        correct = topk[:, :k].eq(labels.view(-1, 1))
        res[f"top{k}"] = correct.any(dim=1).float().mean().item()
    return res
