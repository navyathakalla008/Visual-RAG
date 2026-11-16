import torch.nn as nn
import torch.nn.functional as F

class VisionGuidedSemanticRebalancing(nn.Module):
    def __init__(self):
        super().__init__()

    def distance(self, a, b):
        return 1 - (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1).mean()

    def forward(self, text, aux, concepts, image):
        x1 = self.distance(text, image)
        x2 = self.distance(aux, image)
        x3 = self.distance(concepts, image)
        eps = 1e-8

        w1 = x1 / (x1 + x2 + eps)
        w2 = x2 / (x1 + x2 + eps)
        w3 = x1 / (x1 + x3 + eps)
        w4 = x3 / (x1 + x3 + eps)

        tf = w1 * (w1 * text + w2 * aux) + w2 * aux
        tc = w3 * (w3 * text + w4 * concepts) + w4 * concepts
        return tf, tc, (w1, w2, w3, w4)
