import torch
import torch.nn as nn
import open_clip

class CLIPEncoder(nn.Module):
    def __init__(self, model_name="ViT-B-16", pretrained="openai"):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_image(self, images):
        return self.model.encode_image(images).float()

    def encode_text(self, texts):
        tokens = self.tokenizer(texts).to(next(self.parameters()).device)
        return self.model.encode_text(tokens).float()
