import torch
import torch.nn.functional as F

class PositiveMaskBuilder:
    def __init__(self, embedding_path, threshold=0.7, device="cpu"):
        self.text_embs = torch.load(embedding_path).to(device)
        self.text_embs = F.normalize(self.text_embs, dim=1)
        self.threshold = threshold

    def get_positive_mask(self, batch_indices):
        batch_embs = self.text_embs[batch_indices]

        sim = batch_embs @ batch_embs.T
        sim.fill_diagonal_(-1.0)
        mask = sim >= self.threshold

        return mask
