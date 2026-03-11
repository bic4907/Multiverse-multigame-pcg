import torch
import torch.nn.functional as F
from models.clip.positive_mask import PositiveMaskBuilder

class LossModule:
    def __init__(self, device):
        self.device = device

    def get_mask(self, batch):
        return None

    def loss(self, logits):
        raise NotImplementedError

    def topk_accuracy(self, logits, topk):
        raise NotImplementedError

class BaselineLoss(LossModule):
    name = "base"
    def __init__(self, device):
        super().__init__(device)
        
    def loss(self, logits):
        batch_size = logits.size(0)
        targets = torch.arange(batch_size, device=logits.device)

        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.T, targets)
        return loss_i2t, loss_t2i

    def topk_accuracy(self, logits, topk=(1, 5)):
        batch_size = logits.size(0)
        targets = torch.arange(batch_size, device=logits.device)

        acc = {}
        _, pred = logits.topk(max(topk), dim=1)

        for k in topk:
            correct = (pred[:, :k] == targets.unsqueeze(1)).any(dim=1).float()
            acc[f"top{k}"] = correct.mean().item()
        return acc

class MultiPositiveLoss(LossModule):
    def __init__(self, device):
        super().__init__(device)
        self.mask_builder = None
        self.mask = None

    def loss(self, logits):
        mask = self.mask
        eps = -1e9

        # i → t
        logp_i2t = F.log_softmax(logits, dim=1)
        masked_i2t = logp_i2t.masked_fill(~mask, eps)
        pos_logp_i2t = torch.logsumexp(masked_i2t, dim=1)
        valid_i = mask.sum(dim=1) > 0
        loss_i2t = -pos_logp_i2t[valid_i].mean()

        # t → i
        logp_t2i = F.log_softmax(logits, dim=0)
        masked_t2i = logp_t2i.masked_fill(~mask, eps)
        pos_logp_t2i = torch.logsumexp(masked_t2i, dim=0)
        valid_t = mask.sum(dim=0) > 0
        loss_t2i = -pos_logp_t2i[valid_t].mean()

        return loss_i2t, loss_t2i

    def topk_accuracy(self, logits, topk=(1, 5)):
        mask = self.mask.bool()

        acc = {}
        _, pred = logits.topk(max(topk), dim=1)

        for k in topk:
            hits = []
            for i in range(logits.size(0)):
                retrieved = pred[i, :k]
                hits.append(mask[i, retrieved].any().float())
            acc[f"top{k}"] = torch.stack(hits).mean().item()

        return acc

class SpecificInstructLoss(MultiPositiveLoss):
    name = "spec"
    def __init__(self, embedding_path, threshold, device):
        super().__init__(device)
        self.mask_builder = PositiveMaskBuilder(
            embedding_path=embedding_path,
            threshold=threshold,
            device=device,
        )

    def get_mask(self, batch):
        self.mask = self.mask_builder.get_positive_mask(batch["embedding_idx"])
        return self.mask

class GeneralInstructLoss(MultiPositiveLoss):
    name = "gen"
    def __init__(self, embedding_path, threshold, device):
        super().__init__(device)
        self.mask_builder = PositiveMaskBuilder(
            embedding_path=embedding_path,
            threshold=threshold,
            device=device,
        )

    def get_mask(self, batch):
        self.mask = self.mask_builder.get_positive_mask(batch["embedding_idx"])
        return self.mask

class DifficultyLoss(MultiPositiveLoss):
    name = "diff"
    def __init__(self, device):
        super().__init__(device)

    def get_mask(self, batch):
        difficulty = batch["difficulty"].to(self.device)
        self.mask = difficulty[:, None] == difficulty[None, :]
        return self.mask

def topk_accuracy(logits, ks=(1, 5)):
    B = logits.size(0)
    device = logits.device

    targets = torch.arange(B, device=device)

    max_k = max(ks)
    topk = torch.topk(logits, k=max_k, dim=1).indices

    res = {}
    for k in ks:
        correct = topk[:, :k].eq(targets.unsqueeze(1))
        res[f"top{k}"] = correct.any(dim=1).float().mean().item()
    return res