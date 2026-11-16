# Step 19 — Convert VisualRAG concepts into CLIP feature vectors

import os
import torch
from models.clip_encoder import CLIPEncoder
from models.knowledge import ConceptMemoryBank


# Settings
MEMORY_FILE = "concept_memory.txt"
OUTPUT_FILE = "concept_features.pt"
BATCH_SIZE = 128

device = "cuda" if torch.cuda.is_available() else "cpu"


# Load concept memory
memory = ConceptMemoryBank()
memory.load(MEMORY_FILE)

print("Concepts loaded:", len(memory.concepts))


# Load CLIP
clip = CLIPEncoder(
    model_name="ViT-B-16",
    pretrained="openai"
).to(device)

clip.eval()


# Convert concepts to CLIP features
all_features = []

with torch.no_grad():

    for start in range(
        0,
        len(memory.concepts),
        BATCH_SIZE
    ):

        batch = memory.concepts[
            start:start + BATCH_SIZE
        ]

        features = clip.encode_text(
            batch
        )

        features = torch.nn.functional.normalize(
            features,
            dim=-1
        )

        all_features.append(
            features.cpu()
        )

        print(
            f"Processed {min(start + BATCH_SIZE, len(memory.concepts))}"
            f"/{len(memory.concepts)}"
        )


# Combine all features
concept_features = torch.cat(
    all_features,
    dim=0
)


# Save
torch.save(
    {
        "concepts": memory.concepts,
        "features": concept_features
    },
    OUTPUT_FILE
)


print("\nConcept feature memory created!")
print("Feature shape:", concept_features.shape)
print("Saved to:", os.path.abspath(OUTPUT_FILE))