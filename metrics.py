import torch

def recall_at_k(similarity, k):
    n = similarity.size(0)
    k = min(k, n)
    target = torch.arange(n, device=similarity.device)
    top = similarity.topk(k, dim=1).indices
    return ((top == target[:, None]).any(dim=1).float().mean().item() * 100)

def retrieval_metrics(similarity):
    return {
        "R@1": recall_at_k(similarity, 1),
        "R@5": recall_at_k(similarity, 5),
        "R@10": recall_at_k(similarity, 10)
    }

def rsum(a, b):
    return sum(a.values()) + sum(b.values())
