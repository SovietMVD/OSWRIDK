import torch
import torch.nn.functional as F


def logit_norm_scores(logits, temperature=0.1):
    normalized = F.normalize(logits, dim=1)
    probs = torch.softmax(normalized / temperature, dim=1)
    confidence, _ = probs.max(dim=1)
    return -confidence
