# Step 21 — Test the VisualRAG VPKE pipeline with one image

import torch
from PIL import Image

from models.vpke import VPKE
from models.knowledge import ConceptMemoryBank
from models.clip_encoder import CLIPEncoder


# Settings
IMAGE_PATH = r"D:\Flickr30K\Images\3359636318.jpg"

CAPTION = (
    "Two people are talking outside of the video game shop "
    "next door to the mobile phone store."
)

CONCEPT_FILE = "concept_features.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# Load VPKE
model = VPKE(
    clip_model="ViT-B-16",
    dim=512,
    similarity_dim=256,
    heads=4,
    layers=6,
    Gk=10
).to(device)

model.eval()


# Load concept memory
memory = torch.load(
    CONCEPT_FILE,
    map_location=device
)

concepts = memory["concepts"]
concept_features = memory["features"].to(device)

print("Concept memory:", len(concepts))
print("Concept feature shape:", concept_features.shape)


# Load image
image = Image.open(
    IMAGE_PATH
).convert("RGB")


# Prepare image using the same CLIP preprocessing
image_tensor = model.clip.preprocess(
    image
).unsqueeze(0).to(device)


# Run CLIP image encoder
with torch.no_grad():

    image_feature = model.clip.encode_image(
        image_tensor
    )

    image_feature = torch.nn.functional.normalize(
        image_feature,
        dim=-1
    )


# Run text encoder
with torch.no_grad():

    text_feature = model.clip.encode_text(
        [CAPTION]
    )

    text_feature = torch.nn.functional.normalize(
        text_feature,
        dim=-1
    )


# Select concepts related to the caption
with torch.no_grad():

    selected_concepts, scores = model.selector(
        text_feature,
        concept_features,
        Gk=10
    )


print("\nSelected concept feature shape:")
print(selected_concepts.shape)

print("\nTop concept scores:")
print(scores)


# Find the actual selected concept names
with torch.no_grad():

    t = torch.nn.functional.normalize(
        text_feature,
        dim=-1
    )

    c = torch.nn.functional.normalize(
        concept_features,
        dim=-1
    )

    similarity = t @ c.T

    _, indices = torch.topk(
        similarity,
        k=10,
        dim=1
    )


print("\nTop 10 VisualRAG concepts:")

for i in indices[0]:

    print("-", concepts[i.item()])


print("\nVisualRAG VPKE pipeline test completed successfully!")