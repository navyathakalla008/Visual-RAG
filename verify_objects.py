import torch
import open_clip
from PIL import Image
from ultralytics import YOLO

# Load YOLO
model = YOLO("yolo11n.pt")

# Load image
image_path = r"D:\Flickr30K\Images\1000268201.jpg"
img = Image.open(image_path).convert("RGB")

# Detect objects
results = model(img, conf=0.25)

# Get unique YOLO labels
labels = list(dict.fromkeys(
    [model.names[int(c)] for r in results for c in r.boxes.cls]
))

print("YOLO detected:")
for label in labels:
    print("-", label)

# Load CLIP
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Prepare image and text
image = preprocess(img).unsqueeze(0)
texts = tokenizer([f"a photo of a {label}" for label in labels])

# CLIP similarity
with torch.no_grad():
    image_features = clip_model.encode_image(image)
    text_features = clip_model.encode_text(texts)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    scores = (image_features @ text_features.T)[0]

print("\nCLIP verification:")

for label, score in zip(labels, scores):
    print(f"{label}: {float(score):.3f}")
