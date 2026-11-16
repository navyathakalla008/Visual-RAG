import torch
import open_clip
from PIL import Image
from ultralytics import YOLO

# Load YOLO
yolo = YOLO("yolo11n.pt")

# Load CLIP
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Load image
image_path = r"D:\Flickr30K\Images\1000268201.jpg"
img = Image.open(image_path).convert("RGB")

# YOLO detection
results = yolo(img, conf=0.25)

# Object synonyms that CLIP can consider valid
valid_groups = {
    "person": ["person", "child", "girl", "boy"],
    "bicycle": ["bicycle", "bike"],
    "car": ["car", "vehicle"],
    "chair": ["chair", "seat"],
    "dog": ["dog"],
    "cat": ["cat"],
    "house": ["house", "building"],
    "stairs": ["stairs", "staircase"],
    "door": ["door"],
    "window": ["window"],
}

print("\nFINAL VERIFIED OBJECTS")
print("=" * 50)

verified = []

for box in results[0].boxes:

    yolo_label = yolo.names[int(box.cls)]
    yolo_conf = float(box.conf)

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = img.crop((x1, y1, x2, y2))

    # If YOLO confidence is low, reject
    if yolo_conf < 0.50:
        continue

    # Candidate descriptions
    candidates = []
    candidate_group = {}

    for group, synonyms in valid_groups.items():
        for synonym in synonyms:
            candidates.append(synonym)
            candidate_group[synonym] = group

    texts = tokenizer(
        [f"a photo of {x}" for x in candidates]
    )

    image_tensor = preprocess(crop).unsqueeze(0)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        text_features = clip_model.encode_text(texts)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        scores = (image_features @ text_features.T)[0]

    best_index = scores.argmax().item()
    best_label = candidates[best_index]
    best_score = float(scores[best_index])

    # Accept only if CLIP gives reasonable evidence
    if best_score >= 0.23:
        final_label = candidate_group[best_label]

        if final_label not in verified:
            verified.append(final_label)

            print(
                f"{final_label} "
                f"(YOLO={yolo_conf:.3f}, CLIP={best_score:.3f})"
            )

print("\nObjects shown on website:")
print(verified)
