import torch.nn.functional as F

def triplet_direction(similarity, margin=0.5):
    pos = similarity.diag()
    hard_text = similarity.max(dim=1).values
    hard_image = similarity.max(dim=0).values
    return F.relu(margin - pos + hard_text).mean() + F.relu(margin - pos + hard_image).mean()

def dual_triplet_loss(sim_f, sim_c, mu=8.0, eta=1.0, margin=0.5):
    l1 = triplet_direction(sim_f, margin)
    l2 = triplet_direction(sim_c, margin)
    return mu * l1 + eta * l2, l1, l2
