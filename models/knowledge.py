import re
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import spacy
except ImportError:
    spacy = None

ALLOWED_POS = {"NOUN", "PROPN", "ADJ"}

class ConceptExtractor:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = None
        if spacy:
            try:
                self.nlp = spacy.load(model)
            except Exception:
                pass

    def extract(self, caption):
        if self.nlp:
            doc = self.nlp(caption)
            concepts, current = [], []
            for token in doc:
                if token.pos_ in ALLOWED_POS and not token.is_stop and token.is_alpha:
                    current.append(token.text.lower())
                else:
                    if current:
                        concepts.append(" ".join(current))
                        current = []
            if current:
                concepts.append(" ".join(current))
            return list(dict.fromkeys(concepts))

        stop = {"a","an","the","of","and","in","on","to","with","is","are","for","at","by"}
        return [w for w in re.findall(r"[A-Za-z]+", caption.lower()) if w not in stop]

class ConceptMemoryBank:
    def __init__(self):
        self.concepts = []

    def build(self, extracted_caption_concepts):
        seen = set()
        for concepts in extracted_caption_concepts:
            for concept in concepts:
                if concept not in seen:
                    seen.add(concept)
                    self.concepts.append(concept)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.concepts))

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.concepts = [x.strip() for x in f if x.strip()]

class DualConfidenceSelector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.wm = nn.Linear(2 * dim + 1, 1)

    def forward(self, text_features, concept_features, Gk=10):
        t = F.normalize(text_features, dim=-1)
        c = F.normalize(concept_features, dim=-1)
        sim = t @ c.T

        te = t.unsqueeze(1).expand(-1, c.size(0), -1)
        ce = c.unsqueeze(0).expand(t.size(0), -1, -1)
        conf = torch.sigmoid(self.wm(torch.cat([te, ce, sim.unsqueeze(-1)], dim=-1))).squeeze(-1)
        score = sim * conf

        k = min(Gk, c.size(0))
        _, idx = torch.topk(score, k=k, dim=1)
        selected = torch.stack([c[idx[i]].mean(dim=0) for i in range(t.size(0))])
        return selected, score.gather(1, idx)
