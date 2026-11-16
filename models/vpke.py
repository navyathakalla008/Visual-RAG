import torch
import torch.nn as nn
from .clip_encoder import CLIPEncoder
from .knowledge import DualConfidenceSelector
from .rebalancing import VisionGuidedSemanticRebalancing
from .msha import MultiScaleHierarchicalAttention

class VPKE(nn.Module):
    def __init__(self, clip_model="ViT-B-16", dim=512, similarity_dim=256,
                 heads=4, layers=6, Gk=10):
        super().__init__()
        self.clip = CLIPEncoder(clip_model, "openai")
        self.selector = DualConfidenceSelector(dim)
        self.rebalance = VisionGuidedSemanticRebalancing()
        self.image_msha = MultiScaleHierarchicalAttention(dim, heads, layers, similarity_dim)
        self.fused_msha = MultiScaleHierarchicalAttention(dim, heads, layers, similarity_dim)
        self.entity_msha = MultiScaleHierarchicalAttention(dim, heads, layers, similarity_dim)
        self.concept_projection = nn.Linear(dim, dim)
        self.Gk = Gk

    def forward(self, images, captions, auxiliary_captions=None, concept_features=None):
        image = self.clip.encode_image(images)
        text = self.clip.encode_text(captions)
        aux = text.detach() if auxiliary_captions is None else self.clip.encode_text(auxiliary_captions)

        if concept_features is None:
            concept_features = text.detach()

        concepts, _ = self.selector(text, concept_features, self.Gk)
        concepts = self.concept_projection(concepts)

        tf, tc, weights = self.rebalance(text, aux, concepts, image)

        image_e = self.image_msha(image)
        tf_e = self.fused_msha(tf)
        tc_e = self.entity_msha(tc)

        return {
            "image": image_e,
            "fused_text": tf_e,
            "entity_text": tc_e,
            "sim_f": image_e @ tf_e.T,
            "sim_c": image_e @ tc_e.T,
            "weights": weights
        }
