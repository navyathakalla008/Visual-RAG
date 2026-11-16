import torch
import open_clip
from PIL import Image
from ultralytics import YOLO

# Load models
yolo = YOLO("yolo11n.pt")

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

print("\nYOLO detections:")
print("-" * 50)

for i, box in enumerate(results[0].boxes):

    label = yolo.names[int(box.cls)]
    confidence = float(box.conf)

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Crop detected region
    crop = img.crop((x1, y1, x2, y2))

    # Compare the crop against candidate descriptions
    candidates = [
        label,
        "person",
        "bicycle",
        "car",
        "chair",
        "toilet",
        "object",
        "background"
    ]

    image_tensor = preprocess(crop).unsqueeze(0)
    text_tensor = tokenizer(
        [f"a photo of a {x}" for x in candidates]
    )

    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        text_features = clip_model.encode_text(text_tensor)

        image_features /= image_features.norm(
            dim=-1, keepdim=True
        )
        text_features /= text_features.norm(
            dim=-1, keepdim=True
        )

        scores = (image_features @ text_features.T)[0]

    best_index = scores.argmax().item()
    best_label = candidates[best_index]
    best_score = float(scores[best_index])

    print(f"\nDetection {i + 1}")
    print(f"YOLO label       : {label}")
    print(f"YOLO confidence  : {confidence:.3f}")
    print(f"CLIP best match  : {best_label}")
    print(f"CLIP score       : {best_score:.3f}")

    print("All CLIP scores:")
    for candidate, score in zip(candidates, scores):
        print(f"  {candidate:10s}: {float(score):.3f}")

