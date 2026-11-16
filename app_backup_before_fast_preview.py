# Step 1 — VisualRAG website
# Best Caption + Visual Image Contents

import os
import re
import threading
import uuid
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from ultralytics import YOLO

from flask import Flask, jsonify, redirect, render_template_string, request, send_from_directory
from markupsafe import Markup, escape
from PIL import Image
from werkzeug.utils import secure_filename

from video_pipeline_backup import (
    VIDEO_FRAME_DIR,
    VIDEO_UPLOAD_DIR,
    analyze_video,
    ensure_video_dirs,
    find_matching_index,
    list_video_indexes,
    load_video_index,
    update_index_transcript,
)
from video_search import run_search
from video_ui import VIDEO_HTML

from models.vpke import VPKE
from flickr_entities import (
    get_categorized_entities
)


# ============================================================
# Step 2 — Paths
# ============================================================

DATA_ROOT = r"D:\Flickr30K"

CAPTION_FILE = os.path.join(
    DATA_ROOT,
    "captions.txt"
)

CHECKPOINT = r"checkpoints\last.pt"

CONCEPT_FILE = "concept_features.pt"

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)
ensure_video_dirs()


# ============================================================
# Step 3 — Flask application
# ============================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 600 * 1024 * 1024

_video_jobs = {}
_video_jobs_lock = threading.Lock()
_video_analyze_lock = threading.Lock()


# ============================================================
# Step 4 — Device
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# Step 5 — Load VisualRAG model
# ============================================================

print("Loading VisualRAG model...")

model = VPKE(
    clip_model="ViT-B-16",
    dim=512,
    similarity_dim=256,
    heads=4,
    layers=6,
    Gk=10
)

model = model.to(device)

model.eval()

print("VisualRAG model loaded.")


# ============================================================
# Step 6 — Load YOLO object detector
# ============================================================

print("Loading YOLO object detector...")

yolo_model = YOLO("yolo11n.pt")

print("YOLO object detector loaded.")


# ============================================================
# Step 6A — Load YOLO pose detector safely
# ============================================================

pose_model = None

try:

    print("Loading YOLO pose detector...")
    pose_model = YOLO("yolo11n-pose.pt")
    print("YOLO pose detector loaded.")

except Exception as e:

    print("YOLO pose detector warning:", e)
    print("Pose/action detection will be skipped.")


# ============================================================
# Step 7 — Load checkpoint
# ============================================================

if os.path.exists(CHECKPOINT):

    try:

        checkpoint = torch.load(
            CHECKPOINT,
            map_location=device
        )

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                model.load_state_dict(
                    checkpoint["model_state_dict"],
                    strict=False
                )

            elif "state_dict" in checkpoint:

                model.load_state_dict(
                    checkpoint["state_dict"],
                    strict=False
                )

            else:

                model.load_state_dict(
                    checkpoint,
                    strict=False
                )

        else:

            model.load_state_dict(
                checkpoint,
                strict=False
            )

        print("VisualRAG checkpoint loaded.")

    except Exception as e:

        print(
            "Checkpoint loading warning:",
            e
        )

else:

    print(
        "No checkpoint found."
    )


# ============================================================
# Step 7 — Load concept memory
# ============================================================

concept_features = None

if os.path.exists(CONCEPT_FILE):

    try:

        memory = torch.load(
            CONCEPT_FILE,
            map_location="cpu"
        )

        print(
            "Concept memory loaded."
        )

        if isinstance(memory, dict):

            print(
                "Keys:",
                list(memory.keys())
            )

            if "features" in memory:

                concept_features = memory[
                    "features"
                ]

            elif "concept_features" in memory:

                concept_features = memory[
                    "concept_features"
                ]

        else:

            concept_features = memory

        if concept_features is not None:

            concept_features = (
                concept_features
                .float()
                .to(device)
            )

            print(
                "Concept memory shape:",
                concept_features.shape
            )

    except Exception as e:

        print(
            "Concept memory error:",
            e
        )


# ============================================================
# Step 8 — Visual concepts for WHAT'S IN THE IMAGE
# ============================================================

VISUAL_CONCEPTS = {

    "People": [

        "a man",
        "a woman",
        "a boy",
        "a girl",
        "a child",
        "a baby",
        "a person",
        "a group of people",
        "a crowd"

    ],

    "Food / Vegetables": [

        "vegetables",
        "onions",
        "tomatoes",
        "potatoes",
        "carrots",
        "cabbage",
        "lettuce",
        "spinach",
        "peppers",
        "pumpkin",
        "corn",
        "eggplant",
        "cucumber",
        "garlic",
        "ginger",
        "beans",
        "peas",
        "fruit",
        "apples",
        "bananas",
        "oranges",
        "watermelon",
        "food"

    ],

    "Objects": [

        "a basket",
        "a bag",
        "a box",
        "a table",
        "a chair",
        "a bench",
        "a ladder",
        "a window",
        "a door",
        "a car",
        "a truck",
        "a bus",
        "a bicycle",
        "a motorcycle",
        "a phone",
        "a camera",
        "an umbrella",
        "a cart",
        "a stall",
        "a bottle",
        "a container"

    ],

    "Animals": [

        "a dog",
        "a cat",
        "a horse",
        "a cow",
        "a sheep",
        "a goat",
        "a bird",
        "a chicken",
        "a duck",
        "an elephant",
        "a fish"

    ],

    "Clothing": [

        "a shirt",
        "a t-shirt",
        "jeans",
        "pants",
        "a dress",
        "a skirt",
        "shorts",
        "a jacket",
        "a coat",
        "a hat",
        "a cap",
        "shoes",
        "boots",
        "a saree",
        "a scarf"

    ],

    "Environment": [

        "the sky",
        "clouds",
        "the sun",
        "trees",
        "grass",
        "flowers",
        "plants",
        "mountains",
        "a river",
        "a lake",
        "water",
        "a beach",
        "sand",
        "a road",
        "a street",
        "a building",
        "a house",
        "a field"

    ],

    "Scene": [

        "an outdoor market",
        "a vegetable market",
        "a street market",
        "a market",
        "a shop",
        "a store",
        "a restaurant",
        "a kitchen",
        "a farm",
        "a park",
        "a beach",
        "a street",
        "a road",
        "a city",
        "a village",
        "a playground",
        "an office",
        "a classroom"

    ]
}


# ============================================================
# Step 9 — Get Flickr30K captions
# ============================================================

_CAPTIONS_BY_ID = None
_CAPTIONS_BY_FILENAME = None

PEOPLE_WORDS = [
    "man", "woman", "boy", "girl", "child", "children",
    "kid", "kids", "baby", "person", "people", "crowd",
    "couple", "family", "worker", "player"
]

ANIMAL_WORDS = [
    "dog", "cat", "horse", "cow", "sheep", "goat",
    "bird", "elephant", "bear", "zebra", "giraffe",
    "duck", "fish", "puppy", "kitten"
]

FOOD_WORDS = [
    "banana", "apple", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "sandwich",
    "vegetables", "fruit", "food", "bread", "ice cream"
]

OBJECT_WORDS = [
    "ladder", "window", "door", "staircase", "stairs",
    "table", "chair", "bench", "sofa", "couch", "bed",
    "desk", "shelf", "cabinet", "counter", "sink",
    "stove", "oven", "refrigerator", "microwave",
    "bowl", "plate", "cup", "bottle", "glass", "fork",
    "knife", "spoon", "pan", "pot", "basket", "bag",
    "backpack", "box", "book", "newspaper", "camera",
    "cell phone", "phone", "laptop", "computer",
    "keyboard", "mouse", "television", "tv", "clock",
    "umbrella", "bicycle", "bike", "motorcycle",
    "car", "truck", "bus", "train", "boat", "airplane",
    "traffic light", "fire hydrant", "sign", "fence",
    "wall", "building", "house", "tent", "bridge",
    "tree", "plant", "flower", "ball", "toy",
    "suitcase", "shopping cart", "cart", "stroller",
    "hat", "cap", "helmet", "guitar", "microphone",
    "fountain", "statue", "monument"
]

ENVIRONMENT_WORDS = [
    "sky", "clouds", "trees", "grass", "flowers",
    "plants", "mountains", "river", "lake", "water",
    "beach", "sand", "road", "street", "building",
    "house", "field", "snow", "ocean", "sea", "forest"
]

SCENE_WORDS = [
    "outdoor market", "vegetable market", "street market",
    "market", "shop", "store", "restaurant", "kitchen",
    "farm", "park", "beach", "street", "road", "city",
    "village", "playground", "office", "classroom",
    "forest", "mountain", "indoor", "outdoor",
    "living room", "bedroom", "stadium", "pool"
]

CLOTHING_WORDS = [
    "shirt", "t-shirt", "jeans", "pants", "dress",
    "skirt", "shorts", "jacket", "coat", "hat",
    "cap", "shoes", "boots", "saree", "scarf",
    "helmet", "uniform", "sweater", "hoodie"
]

MATERIAL_WORDS = [
    "wood", "wooden", "metal", "glass", "plastic",
    "fabric", "leather", "stone", "brick", "concrete",
    "steel", "paper"
]

SEMANTIC_WORDS = [
    "walking", "running", "sitting", "standing",
    "playing", "jumping", "riding", "eating",
    "cooking", "working", "talking", "smiling",
    "holding", "climbing", "swimming", "dancing",
    "shopping", "reading", "singing", "posing",
    "looking", "watching", "wearing", "carrying",
    "sitting", "leaning", "pointing", "laughing"
]

ATTRIBUTE_WORDS = [
    "red", "blue", "green", "yellow", "black", "white",
    "brown", "orange", "pink", "purple", "gray", "grey",
    "young", "old", "little", "small", "large", "big",
    "tall", "short", "long", "wooden", "metal", "glass",
    "striped", "colorful", "dark", "bright", "smiling"
]

RELATION_PHRASES = [
    "in front of", "on top of", "next to", "in front",
    "out of", "looking at", "playing with", "sitting on",
    "standing on", "standing in", "sitting in",
    "walking on", "walking in", "holding a", "wearing a",
    "surrounded by", "in the middle of"
]

RELATION_WORDS = [
    "near", "beside", "behind", "above", "below",
    "inside", "around", "between", "with", "over",
    "under", "onto", "across", "along", "against"
]

STOPWORDS = {
    "a", "an", "the", "of", "to", "for", "and", "or",
    "is", "are", "was", "were", "be", "this", "that",
    "it", "his", "her", "their", "they", "he", "she",
    "who", "whom", "whose", "from", "by", "as", "if",
    "then", "than", "so", "not", "no", "its", "at"
}


def extract_flickr_id(name):
    """Extract a Flickr30K numeric image id from any filename."""

    if not name:
        return None

    base = os.path.splitext(
        os.path.basename(str(name))
    )[0]

    if base.lower().startswith("temp_"):
        base = base[5:]

    match = re.search(r"(\d{6,})", base)

    if match:
        return match.group(1)

    return None


def _strip_caption_quotes(text):
    text = text.strip()
    if (
        len(text) >= 2
        and text[0] == text[-1]
        and text[0] in "\"'"
    ):
        text = text[1:-1].strip()
    return text


def _load_caption_index():
    """Load captions.txt once, keyed by Flickr id and filename."""

    global _CAPTIONS_BY_ID, _CAPTIONS_BY_FILENAME

    if _CAPTIONS_BY_ID is not None:
        return

    _CAPTIONS_BY_ID = {}
    _CAPTIONS_BY_FILENAME = {}

    if not os.path.exists(CAPTION_FILE):
        print("Flickr30K captions.txt not found:", CAPTION_FILE)
        return

    with open(
        CAPTION_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for raw_line in f:

            line = raw_line.strip()

            if not line:
                continue

            if "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(",", 1)

            if len(parts) != 2:
                continue

            filename = parts[0].strip().strip("\"'")
            filename = os.path.basename(filename)
            filename = re.sub(r"#\d+$", "", filename)

            caption = _strip_caption_quotes(parts[1])

            if not filename or not caption:
                continue

            _CAPTIONS_BY_FILENAME.setdefault(
                filename.lower(),
                []
            ).append(caption)

            flickr_id = extract_flickr_id(filename)

            if flickr_id:
                _CAPTIONS_BY_ID.setdefault(
                    flickr_id,
                    []
                ).append(caption)

    print(
        "Flickr30K caption index:",
        len(_CAPTIONS_BY_ID),
        "images"
    )


def get_captions(image_name):

    _load_caption_index()

    captions = []
    seen = set()

    def _add(items):
        for caption in items or []:
            key = caption.lower()
            if key not in seen:
                seen.add(key)
                captions.append(caption)

    filename = os.path.basename(
        str(image_name or "")
    )
    flickr_id = extract_flickr_id(filename)

    if flickr_id:
        _add(_CAPTIONS_BY_ID.get(flickr_id))

    _add(
        _CAPTIONS_BY_FILENAME.get(
            filename.lower()
        )
    )

    if flickr_id:
        _add(
            _CAPTIONS_BY_FILENAME.get(
                (flickr_id + ".jpg").lower()
            )
        )
        _add(
            _CAPTIONS_BY_FILENAME.get(
                (flickr_id + ".jpeg").lower()
            )
        )

    return captions[:5]


def _joined_caption_text(captions):
    return " ".join(
        captions or []
    ).lower()


def _label_mentioned(label, text):
    if not label or not text:
        return False

    label = str(label).lower().strip()
    label = re.sub(r"^(a|an|the)\s+", "", label)

    if not label:
        return False

    pattern = r"\b" + re.escape(label) + r"s?\b"
    return re.search(pattern, text) is not None


def labels_from_text(labels, text, min_hits=1):
    found = []

    if not text:
        return found

    for label in labels:
        if _label_mentioned(label, text):
            found.append(label)

    return found


def _merge_category(detected, category, items, limit=12):
    if not items:
        return

    combined = _unique_labels(
        list(detected.get(category, [])) + list(items)
    )[:limit]

    if combined:
        detected[category] = combined


def dataset_labels_from_captions(captions):
    """Pull people/objects/scene words that the Flickr captions name."""

    text = _joined_caption_text(captions)

    if not text:
        return {}

    found = {}

    people = labels_from_text(PEOPLE_WORDS, text)
    if people:
        found["People"] = ["person"]

    animals = labels_from_text(ANIMAL_WORDS, text)
    if animals:
        found["Animals"] = _unique_labels(animals)

    food = labels_from_text(FOOD_WORDS, text)
    if food:
        found["Food / Vegetables"] = _unique_labels(food)

    objects = labels_from_text(OBJECT_WORDS, text)
    if objects:
        found["Objects"] = _unique_labels(objects)

    environment = labels_from_text(ENVIRONMENT_WORDS, text)
    if environment:
        found["Environment"] = _unique_labels(environment)

    scene = labels_from_text(SCENE_WORDS, text)
    if scene:
        # Keep the most specific scene phrases first.
        scene = sorted(scene, key=len, reverse=True)
        found["Scene"] = _unique_labels(scene)[:2]

    clothing = labels_from_text(CLOTHING_WORDS, text)
    if clothing:
        found["Clothing"] = _unique_labels(clothing)

    return found


def materials_from_captions(captions):
    text = _joined_caption_text(captions)
    found = labels_from_text(MATERIAL_WORDS, text)

    cleaned = []
    for item in found:
        if item == "wooden":
            item = "wood"
        if item not in cleaned:
            cleaned.append(item)

    return cleaned[:4]


def semantics_from_captions(captions):
    text = _joined_caption_text(captions)
    return _unique_labels(
        labels_from_text(SEMANTIC_WORDS, text)
    )[:4]


# ============================================================
# Step 10 — Get CLIP image embedding
# ============================================================

@torch.no_grad()
def get_image_embedding(
    image_path,
    normalize=True
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = (
        model.clip.preprocess(image)
        .unsqueeze(0)
        .to(device)
    )

    image_features = (
        model.clip.model
        .encode_image(
            image_tensor
        )
        .float()
    )

    if normalize:
        image_features = F.normalize(
            image_features,
            dim=-1
        )

    return image_features


# ============================================================
# Step 11 — Get CLIP text embeddings
# ============================================================

@torch.no_grad()
def get_text_embeddings(
    texts,
    normalize=True
):

    tokens = (
        model.clip.tokenizer(
            texts
        )
        .to(device)
    )

    text_features = (
        model.clip.model
        .encode_text(
            tokens
        )
        .float()
    )

    if normalize:
        text_features = F.normalize(
            text_features,
            dim=-1
        )

    return text_features


def clip_logit_scale():
    """CLIP temperature used for image-text similarity logits."""
    clip_model = model.clip.model
    if hasattr(clip_model, "logit_scale"):
        return float(
            clip_model.logit_scale.exp().item()
        )
    return 100.0


@torch.no_grad()
def get_pil_embedding(image, normalize=True):
    """CLIP embedding for a PIL image or crop."""
    if image is None:
        return None

    image = image.convert("RGB")

    if image.width < 8 or image.height < 8:
        return None

    image_tensor = (
        model.clip.preprocess(image)
        .unsqueeze(0)
        .to(device)
    )

    image_features = (
        model.clip.model
        .encode_image(image_tensor)
        .float()
    )

    if normalize:
        image_features = F.normalize(
            image_features,
            dim=-1
        )

    return image_features


@torch.no_grad()
def score_embedding_against_labels(
    image_raw,
    labels,
    prompt_template="a photograph of {}"
):
    """
    Score each label against one CLIP image embedding.

    Returns cosine similarity, raw dot product, Euclidean
    distance, and CLIP image-text similarity (scaled cosine).
    """
    unique = _unique_labels(labels)

    if image_raw is None or not unique:
        return []

    texts = [
        prompt_template.format(label)
        for label in unique
    ]

    text_raw = get_text_embeddings(
        texts,
        normalize=False
    )

    image_unit = F.normalize(
        image_raw,
        dim=-1
    )
    text_unit = F.normalize(
        text_raw,
        dim=-1
    )

    cosine_scores = (
        image_unit @ text_unit.T
    )[0]

    dot_scores = (
        image_raw @ text_raw.T
    )[0]

    euclidean_scores = torch.norm(
        image_raw.unsqueeze(1) - text_raw.unsqueeze(0),
        p=2,
        dim=-1
    )[0]

    scale = clip_logit_scale()
    results = []

    for i, label in enumerate(unique):
        cosine = float(cosine_scores[i].item())
        results.append(
            {
                "label": label,
                "cosine": cosine,
                "dot_product": float(
                    dot_scores[i].item()
                ),
                "euclidean": float(
                    euclidean_scores[i].item()
                ),
                "image_text_similarity": cosine * scale
            }
        )

    results.sort(
        key=lambda x: x["cosine"],
        reverse=True
    )

    return results


def _content_labels(items):
    """Labels from What's-in-the-image entries (strings or scored dicts)."""
    labels = []

    for item in items or []:
        if isinstance(item, dict):
            label = item.get("label")
            if label:
                labels.append(label)
        else:
            text = str(item).strip()
            if text:
                labels.append(text)

    return labels


def _merge_scored(bucket, items, index_map):
    """Add scored items, keeping the stronger cosine if duplicated."""
    for item in items or []:
        if not isinstance(item, dict):
            continue

        key = str(
            item.get("label", "")
        ).lower().strip()

        if not key:
            continue

        if key in index_map:
            i = index_map[key]
            if item["cosine"] > bucket[i]["cosine"]:
                bucket[i] = item
            continue

        index_map[key] = len(bucket)
        bucket.append(item)


def _filter_scored(items, min_cosine=0.0, limit=None):
    kept = [
        item for item in items
        if item.get("cosine", 0.0) >= min_cosine
    ]
    if limit is not None:
        kept = kept[:limit]
    return kept


# ============================================================
# Step 12 — Find highest-related caption
# ============================================================

@torch.no_grad()
def find_best_caption(
    image_path,
    captions
):

    if not captions:

        return (
            "No Flickr30K captions found.",
            0.0,
            [],
            None,
            None,
            None
        )

    print()
    print(
        "Comparing image with all 5 captions..."
    )

    # Raw CLIP embeddings so cosine, dot product, and Euclidean
    # distance are three different image-text scores.
    image_raw = get_image_embedding(
        image_path,
        normalize=False
    )

    text_raw = get_text_embeddings(
        captions,
        normalize=False
    )

    image_unit = F.normalize(
        image_raw,
        dim=-1
    )

    text_unit = F.normalize(
        text_raw,
        dim=-1
    )

    cosine_scores = (
        image_unit @ text_unit.T
    )[0]

    dot_scores = (
        image_raw @ text_raw.T
    )[0]

    euclidean_scores = torch.norm(
        image_raw.unsqueeze(1) - text_raw.unsqueeze(0),
        p=2,
        dim=-1
    )[0]

    results = []

    for i, caption in enumerate(
        captions
    ):

        cosine = float(
            cosine_scores[i].item()
        )
        dot_product = float(
            dot_scores[i].item()
        )
        euclidean = float(
            euclidean_scores[i].item()
        )
        image_text_similarity = (
            cosine * clip_logit_scale()
        )

        results.append(
            {
                "number": i + 1,
                "caption": caption,
                "image_text_similarity": image_text_similarity,
                "cosine": cosine,
                "dot_product": dot_product,
                "euclidean": euclidean,
                "score": cosine
            }
        )

    ranked_results = sorted(
        results,
        key=lambda x: x["cosine"],
        reverse=True
    )

    best_cosine = max(
        results,
        key=lambda x: x["cosine"]
    )

    best_dot = max(
        results,
        key=lambda x: x["dot_product"]
    )

    best_euclidean = min(
        results,
        key=lambda x: x["euclidean"]
    )

    print()
    print(
        "========================================"
    )

    print(
        "IMAGE-TEXT MATCHING RESULTS"
    )

    print(
        "========================================"
    )

    for item in ranked_results:

        print(
            f"Caption {item['number']}: "
            f"cosine={item['cosine']:.4f}  "
            f"dot={item['dot_product']:.4f}  "
            f"euclidean={item['euclidean']:.4f}"
        )

        print(
            item["caption"]
        )

        print()

    print(
        "BEST BY COSINE SIMILARITY:"
    )
    print(best_cosine["caption"])
    print(f"{best_cosine['cosine']:.4f}")

    print(
        "BEST BY DOT PRODUCT:"
    )
    print(best_dot["caption"])
    print(f"{best_dot['dot_product']:.4f}")

    print(
        "BEST BY EUCLIDEAN DISTANCE:"
    )
    print(best_euclidean["caption"])
    print(f"{best_euclidean['euclidean']:.4f}")

    print(
        "========================================"
    )

    return (
        best_cosine["caption"],
        best_cosine["cosine"],
        ranked_results,
        best_cosine,
        best_dot,
        best_euclidean
    )


# ============================================================
# Step 13 — Detect image contents with grounded YOLO + conservative CLIP
# ============================================================

@torch.no_grad()
def get_yolo_detections(image_path, confidence=0.35):
    """Return real YOLO detections with bounding boxes."""
    image = Image.open(image_path).convert("RGB")
    results = yolo_model(
        image,
        conf=confidence,
        iou=0.50,
        verbose=False
    )

    detections = []

    if not results or len(results[0].boxes) == 0:
        return detections

    for box in results[0].boxes:
        conf = float(box.conf.item())

        if conf < confidence:
            continue

        x1, y1, x2, y2 = [
            int(v)
            for v in box.xyxy[0].tolist()
        ]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "label": yolo_model.names[
                    int(box.cls.item())
                ].lower(),
                "confidence": conf,
                "box": (x1, y1, x2, y2),
                "area_ratio":
                    ((x2 - x1) * (y2 - y1))
                    / float(image.width * image.height),
                "center":
                    ((x1 + x2) / 2.0,
                     (y1 + y2) / 2.0)
            }
        )

    detections.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return detections


def _unique_labels(labels):
    """Remove duplicate labels without changing order."""
    return list(
        dict.fromkeys(
            str(x).strip()
            for x in labels
            if str(x).strip()
        )
    )


@torch.no_grad()
def clip_rank_image(
    image_features,
    labels,
    prompt_template,
    threshold=0.28,
    margin=0.012,
    top_k=3
):
    """
    Conservative whole-image CLIP classification.

    A label is returned only when its score is high enough and is
    sufficiently separated from the next alternatives.
    """
    if not labels:
        return []

    text_features = get_text_embeddings(
        [
            prompt_template.format(label)
            for label in labels
        ]
    )

    scores = (
        image_features @ text_features.T
    )[0]

    ranked = sorted(
        [
            (
                float(scores[i].item()),
                labels[i]
            )
            for i in range(len(labels))
        ],
        reverse=True
    )

    if not ranked:
        return []

    if top_k <= 1:
        best_score, best_label = ranked[0]
        second_score = (
            ranked[1][0]
            if len(ranked) > 1
            else -1.0
        )

        if best_score < threshold:
            return []

        if len(ranked) > 1 and (
            best_score - second_score < margin
        ):
            return []

        return [best_label]

    accepted = []
    for score, label in ranked:
        if score < threshold:
            break
        accepted.append(label)
        if len(accepted) >= top_k:
            break

    return accepted


@torch.no_grad()
def clip_rank_crop(
    crop,
    labels,
    prompt_template,
    threshold=0.27,
    margin=0.012
):
    """
    Conservative CLIP classification on a real detected region.
    This is used only after YOLO has supplied an actual image region.
    """
    if crop is None or crop.width < 10 or crop.height < 10:
        return None, 0.0

    image_tensor = (
        model.clip.preprocess(crop)
        .unsqueeze(0)
        .to(device)
    )

    image_features = (
        model.clip.model
        .encode_image(image_tensor)
        .float()
    )

    image_features = F.normalize(
        image_features,
        dim=-1
    )

    text_features = get_text_embeddings(
        [
            prompt_template.format(label)
            for label in labels
        ]
    )

    scores = (
        image_features @ text_features.T
    )[0]

    ranked = sorted(
        [
            (
                float(scores[i].item()),
                labels[i]
            )
            for i in range(len(labels))
        ],
        reverse=True
    )

    if not ranked:
        return None, 0.0

    best_score, best_label = ranked[0]

    second_score = (
        ranked[1][0]
        if len(ranked) > 1
        else -1.0
    )

    if best_score < threshold:
        return None, best_score

    if (
        len(ranked) > 1
        and best_score - second_score < margin
    ):
        return None, best_score

    return best_label, best_score


def _crop_from_detection(image, detection):
    x1, y1, x2, y2 = detection["box"]

    return image.crop(
        (x1, y1, x2, y2)
    )


@torch.no_grad()
def detect_image_contents(image_path, reference_captions=None):
    """
    WHAT'S IN THE IMAGE

    Inventory of what is actually visible: YOLO boxes first,
    then CLIP only to name person/clothing crops and to confirm
    caption or Flickr phrases against the picture. Similarity
    scores are used internally and are not shown.
    """

    detected = {}
    people_map, clothing_map = {}, {}
    animals_map, objects_map = {}, {}
    food_map, environment_map, scene_map = {}, {}, {}
    people, clothing, animals = [], [], []
    objects, food, environment, scene = [], [], [], []

    image = Image.open(
        image_path
    ).convert("RGB")

    flickr_id = extract_flickr_id(image_path)
    caption_text = _joined_caption_text(reference_captions)

    entity_contents = {}
    if flickr_id:
        try:
            entity_contents = get_categorized_entities(
                flickr_id
            ) or {}
        except Exception as e:
            print("Flickr30K Entities warning:", e)
            entity_contents = {}

    dataset_image = bool(reference_captions) or bool(
        entity_contents
    )

    caption_contents = dataset_labels_from_captions(
        reference_captions
    )

    detections = get_yolo_detections(
        image_path,
        confidence=0.25 if dataset_image else 0.45
    )

    food_labels = {
        "banana", "apple", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "sandwich"
    }

    animal_labels = {
        "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe"
    }

    people_detail = [
        "man", "woman", "boy", "girl", "child",
        "baby", "person", "group of people", "crowd"
    ]

    clothing_labels = [
        "shirt", "t-shirt", "jeans", "pants", "dress",
        "skirt", "shorts", "jacket", "coat", "hat",
        "cap", "shoes", "boots", "saree", "scarf",
        "helmet", "uniform", "sweater", "hoodie"
    ]

    environment_labels = [
        "sky", "clouds", "trees", "grass", "flowers",
        "plants", "mountains", "river", "lake", "water",
        "beach", "sand", "road", "street", "building",
        "house", "field", "snow", "ocean", "forest"
    ]

    scene_labels = [
        "outdoor market", "vegetable market", "street market",
        "market", "shop", "store", "restaurant", "kitchen",
        "farm", "park", "beach", "street", "road", "city",
        "village", "playground", "office", "classroom",
        "forest", "mountain", "indoor scene", "outdoor scene",
        "living room", "bedroom", "stadium", "pool"
    ]

    image_raw = get_image_embedding(
        image_path,
        normalize=False
    )

    best_crop_by_label = {}
    person_crops = []

    for detection in detections:
        label = detection["label"]
        conf = float(detection.get("confidence", 0.0))
        crop = _crop_from_detection(image, detection)

        mentioned = (
            _label_mentioned(label, caption_text)
            or any(
                _label_mentioned(
                    label,
                    " ".join(items).lower()
                )
                for items in entity_contents.values()
            )
        )

        if dataset_image and conf < 0.40 and not mentioned:
            continue

        if label == "person":
            person_crops.append(crop)
            continue

        previous = best_crop_by_label.get(label)
        if previous is None or conf > previous[0]:
            best_crop_by_label[label] = (conf, crop)

    # --------------------------------------------------------
    # People — classify each person crop
    # --------------------------------------------------------

    for crop in person_crops:
        crop_raw = get_pil_embedding(crop, normalize=False)
        ranked = score_embedding_against_labels(
            crop_raw,
            people_detail,
            "a photograph of a {}"
        )
        if ranked and ranked[0]["cosine"] >= 0.18:
            _merge_scored(people, [ranked[0]], people_map)
        elif ranked:
            person_only = [
                item for item in ranked
                if item["label"] == "person"
            ]
            _merge_scored(
                people,
                person_only or ranked[:1],
                people_map
            )

    if not people and dataset_image:
        ranked = score_embedding_against_labels(
            image_raw,
            people_detail,
            "a photograph of a {}"
        )
        _merge_scored(
            people,
            _filter_scored(ranked, min_cosine=0.28, limit=2),
            people_map
        )

    # --------------------------------------------------------
    # YOLO animals / food / objects, scored on their crops
    # --------------------------------------------------------

    for label, (_conf, crop) in best_crop_by_label.items():
        crop_raw = get_pil_embedding(crop, normalize=False)
        ranked = score_embedding_against_labels(
            crop_raw,
            [label],
            "a photograph of a {}"
        )
        if not ranked:
            continue

        if label in animal_labels:
            _merge_scored(animals, ranked, animals_map)
        elif label in food_labels:
            _merge_scored(food, ranked, food_map)
        else:
            _merge_scored(objects, ranked, objects_map)

    # --------------------------------------------------------
    # Open-vocabulary objects + caption/entity details
    # --------------------------------------------------------

    extra_object_labels = _unique_labels(
        caption_contents.get("Objects", [])
        + entity_contents.get("Objects", [])
    )

    existing_object = {
        item["label"].lower() for item in objects
    }
    extra_object_labels = [
        label for label in extra_object_labels
        if label.lower() not in existing_object
    ]

    if extra_object_labels:
        ranked = score_embedding_against_labels(
            image_raw,
            extra_object_labels,
            "a photograph containing a {}"
        )
        accepted = [
            item for item in ranked
            if item["cosine"] >= 0.24
        ]
        _merge_scored(
            objects,
            accepted[:12],
            objects_map
        )

    extra_animals = _unique_labels(
        caption_contents.get("Animals", [])
        + entity_contents.get("Animals", [])
    )
    extra_animals = [
        label for label in extra_animals
        if label.lower() not in {
            item["label"].lower() for item in animals
        }
    ]
    if extra_animals:
        ranked = score_embedding_against_labels(
            image_raw,
            extra_animals,
            "a photograph of a {}"
        )
        accepted = [
            item for item in ranked
            if item["cosine"] >= 0.24
        ]
        _merge_scored(animals, accepted[:8], animals_map)

    extra_food = _unique_labels(
        caption_contents.get("Food / Vegetables", [])
    )
    extra_food = [
        label for label in extra_food
        if label.lower() not in {
            item["label"].lower() for item in food
        }
    ]
    if extra_food:
        ranked = score_embedding_against_labels(
            image_raw,
            extra_food,
            "a photograph of {}"
        )
        accepted = [
            item for item in ranked
            if item["cosine"] >= 0.24
        ]
        _merge_scored(food, accepted[:8], food_map)

    # Flickr entity phrases give finer names (for example
    # "young boy" instead of only "person").
    for category, phrases in entity_contents.items():
        if not phrases:
            continue
        ranked = score_embedding_against_labels(
            image_raw,
            phrases,
            "a photograph of {}"
        )
        accepted = _filter_scored(
            ranked,
            min_cosine=0.24
        )
        if category == "People":
            _merge_scored(people, accepted, people_map)
        elif category == "Clothing":
            _merge_scored(clothing, accepted, clothing_map)
        elif category == "Animals":
            _merge_scored(animals, accepted, animals_map)
        elif category == "Environment":
            _merge_scored(
                environment,
                accepted,
                environment_map
            )
        else:
            _merge_scored(objects, accepted, objects_map)

    # --------------------------------------------------------
    # Environment and scene
    # --------------------------------------------------------

    env_candidates = _unique_labels(
        environment_labels
        + caption_contents.get("Environment", [])
        + entity_contents.get("Environment", [])
    )
    ranked_env = score_embedding_against_labels(
        image_raw,
        env_candidates,
        "a photograph containing {}"
    )
    _merge_scored(
        environment,
        _filter_scored(ranked_env, min_cosine=0.26, limit=5),
        environment_map
    )

    scene_candidates = _unique_labels(
        scene_labels
        + caption_contents.get("Scene", [])
    )
    ranked_scene = score_embedding_against_labels(
        image_raw,
        scene_candidates,
        "a photograph of {}"
    )
    _merge_scored(
        scene,
        _filter_scored(ranked_scene, min_cosine=0.26, limit=3),
        scene_map
    )

    # --------------------------------------------------------
    # Clothing from person crops, then the whole image
    # --------------------------------------------------------

    for crop in person_crops:
        crop_raw = get_pil_embedding(crop, normalize=False)
        ranked = score_embedding_against_labels(
            crop_raw,
            clothing_labels,
            "a photograph of a person wearing {}"
        )
        _merge_scored(
            clothing,
            _filter_scored(ranked, min_cosine=0.22, limit=2),
            clothing_map
        )

    extra_clothing = _unique_labels(
        caption_contents.get("Clothing", [])
        + entity_contents.get("Clothing", [])
    )
    if extra_clothing:
        ranked_clothing = score_embedding_against_labels(
            image_raw,
            extra_clothing,
            "a photograph of a person wearing {}"
        )
        _merge_scored(
            clothing,
            _filter_scored(
                ranked_clothing,
                min_cosine=0.24,
                limit=6
            ),
            clothing_map
        )
    elif people and not clothing:
        ranked_clothing = score_embedding_against_labels(
            image_raw,
            clothing_labels,
            "a photograph of a person wearing {}"
        )
        _merge_scored(
            clothing,
            _filter_scored(
                ranked_clothing,
                min_cosine=0.24,
                limit=3
            ),
            clothing_map
        )

    people.sort(key=lambda x: x["cosine"], reverse=True)
    clothing.sort(key=lambda x: x["cosine"], reverse=True)
    objects.sort(key=lambda x: x["cosine"], reverse=True)
    animals.sort(key=lambda x: x["cosine"], reverse=True)
    food.sort(key=lambda x: x["cosine"], reverse=True)
    environment.sort(key=lambda x: x["cosine"], reverse=True)
    scene.sort(key=lambda x: x["cosine"], reverse=True)

    if people:
        detected["People"] = people[:12]
    if clothing:
        detected["Clothing"] = clothing[:10]
    if objects:
        detected["Objects"] = objects[:20]
    if animals:
        detected["Animals"] = animals[:10]
    if food:
        detected["Food / Vegetables"] = food[:10]
    if environment:
        detected["Environment"] = environment[:8]
    if scene:
        detected["Scene"] = scene[:4]

    print()
    print("========================================")
    print("WHAT'S IN THE IMAGE")
    print("========================================")

    for category, items in detected.items():
        names = _content_labels(items)
        print(category + ":", "; ".join(names))

    print("========================================")

    content_order = [
        "People",
        "Clothing",
        "Objects",
        "Animals",
        "Food / Vegetables",
        "Environment",
        "Scene"
    ]

    ordered_contents = {
        category: _unique_labels(
            _content_labels(detected[category])
        )
        for category in content_order
        if detected.get(category)
    }

    for category, items in detected.items():
        if category not in ordered_contents:
            ordered_contents[category] = _unique_labels(
                _content_labels(items)
            )

    return ordered_contents


# ============================================================
# Step 14 — Accurate visual feature analysis
# ============================================================

# Visual features are measured from the uploaded image and/or
# actual detected regions. A feature is not invented when there
# is insufficient visual evidence.


def detect_dominant_colors(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:
        return []

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (320, 320),
        interpolation=cv2.INTER_AREA
    )

    pixels = image.reshape(
        -1,
        3
    ).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1.0
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        10,
        None,
        criteria,
        6,
        cv2.KMEANS_PP_CENTERS
    )

    counts = np.bincount(
        labels.ravel(),
        minlength=len(centers)
    )

    ratios = (
        counts
        / max(1, counts.sum())
    )

    # Wider palette so common natural colors are not forced
    # into a wrong basic color.
    palette = {
        "Black": np.array([25, 25, 25]),
        "White": np.array([245, 245, 245]),
        "Gray": np.array([145, 145, 145]),
        "Red": np.array([210, 45, 45]),
        "Orange": np.array([235, 125, 35]),
        "Yellow": np.array([225, 195, 45]),
        "Green": np.array([55, 165, 75]),
        "Cyan": np.array([40, 175, 190]),
        "Blue": np.array([55, 110, 205]),
        "Purple": np.array([125, 75, 180]),
        "Pink": np.array([220, 105, 155]),
        "Brown": np.array([120, 75, 45]),
        "Beige": np.array([205, 185, 145])
    }

    result = []

    for idx in np.argsort(
        ratios
    )[::-1]:

        ratio = float(
            ratios[idx]
        )

        if ratio < 0.06:
            continue

        color = centers[idx]

        distances = {
            name: float(
                np.linalg.norm(
                    color - reference
                )
            )
            for name, reference
            in palette.items()
        }

        name = min(
            distances,
            key=distances.get
        )

        # Do not report the same color repeatedly.
        if name not in result:
            result.append(name)

        if len(result) >= 6:
            break

    return result


def get_grounded_yolo_detections(image_path):

    return get_yolo_detections(
        image_path,
        confidence=0.45
    )


@torch.no_grad()
def detect_shapes_from_objects(
    image_path,
    detections
):
    """
    Shape is determined from real detected regions.
    CLIP is used only as a secondary check.
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    shape_labels = [
        "round",
        "circular",
        "square",
        "rectangular",
        "triangular",
        "elongated",
        "irregular"
    ]

    features = []

    for detection in detections[:8]:

        x1, y1, x2, y2 = detection["box"]

        width = max(
            1,
            x2 - x1
        )

        height = max(
            1,
            y2 - y1
        )

        aspect = (
            width
            / float(height)
        )

        # Strong geometric evidence.
        if (
            0.90 <= aspect <= 1.10
        ):
            geometric_hint = "square"

        elif (
            aspect >= 1.80
            or aspect <= 0.56
        ):
            geometric_hint = "elongated"

        else:
            geometric_hint = None

        crop = _crop_from_detection(
            image,
            detection
        )

        label, score = clip_rank_crop(
            crop,
            shape_labels,
            "a close-up photograph of a {} shaped object",
            threshold=0.275,
            margin=0.008
        )

        if label is not None:

            if label == "circular":
                label = "round"

            if label not in features:
                features.append(label)

        elif geometric_hint is not None:

            if geometric_hint not in features:
                features.append(
                    geometric_hint
                )

    return features[:6]


def detect_texture(
    image_path,
    detections
):
    """
    Texture combines measurable local contrast/gradient evidence
    with CLIP only on real detected regions.
    """

    image_gray = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image_gray is None:
        return []

    crops = []

    for detection in detections[:6]:

        x1, y1, x2, y2 = detection["box"]

        crop = image_gray[
            y1:y2,
            x1:x2
        ]

        if crop.size > 150:
            crops.append(crop)

    if not crops:
        crops = [
            image_gray
        ]

    texture_scores = []

    for crop in crops:

        crop = cv2.resize(
            crop,
            (160, 160),
            interpolation=cv2.INTER_AREA
        )

        laplacian = float(
            cv2.Laplacian(
                crop,
                cv2.CV_64F
            ).var()
        )

        gradient_x = cv2.Sobel(
            crop,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        gradient_y = cv2.Sobel(
            crop,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        gradient = float(
            np.mean(
                np.sqrt(
                    gradient_x ** 2
                    + gradient_y ** 2
                )
            )
        )

        texture_scores.append(
            (
                laplacian,
                gradient
            )
        )

    median_lap = float(
        np.median(
            [
                x[0]
                for x in texture_scores
            ]
        )
    )

    median_gradient = float(
        np.median(
            [
                x[1]
                for x in texture_scores
            ]
        )
    )

    features = []

    if (
        median_lap < 45
        and median_gradient < 18
    ):
        features.append(
            "smooth"
        )

    elif (
        median_lap > 350
        or median_gradient > 45
    ):
        features.append(
            "rough"
        )

    else:
        features.append(
            "moderate"
        )

    # Semantic texture labels are checked only on real objects.
    if detections:

        image = Image.open(
            image_path
        ).convert("RGB")

        texture_labels = [
            "smooth",
            "rough",
            "furry",
            "woven"
        ]

        for detection in detections[:4]:

            crop = _crop_from_detection(
                image,
                detection
            )

            label, score = clip_rank_crop(
                crop,
                texture_labels,
                "a close-up photograph showing a {} texture",
                threshold=0.29,
                margin=0.012
            )

            if label is not None:

                if label not in features:
                    features.append(
                        label
                    )

    return _unique_labels(
        features
    )[:4]


def _periodic_score(profile):
    """
    Measures whether a 1-D edge profile contains a strong
    repeated frequency.
    """

    profile = (
        profile.astype(np.float32)
        - np.mean(profile)
    )

    if np.std(profile) < 1e-5:
        return 0.0

    autocorr = np.correlate(
        profile,
        profile,
        mode="full"
    )[len(profile)-1:]

    autocorr = autocorr / (
        autocorr[0] + 1e-8
    )

    # Ignore the zero-lag peak.
    search = autocorr[
        4:
        min(
            len(autocorr),
            len(profile) // 2
        )
    ]

    if len(search) == 0:
        return 0.0

    return float(
        np.max(search)
    )


def detect_patterns(image_path):
    """
    Pattern detector designed to distinguish:
      - striped / linear
      - checked / grid
      - dotted
      - repeated geometric
      - floral (only with strong CLIP evidence)
      - plain / no strong regular pattern

    The two supplied example images are handled as:
      first  -> linear / striped when repeated lines are present
      second -> checked / grid when both directions repeat.
    """

    image = cv2.imread(
        image_path
    )

    if image is None:
        return []

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (384, 384),
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    edges = cv2.Canny(
        gray,
        45,
        130
    )

    row_profile = (
        edges.mean(axis=1)
    )

    col_profile = (
        edges.mean(axis=0)
    )

    row_period = _periodic_score(
        row_profile
    )

    col_period = _periodic_score(
        col_profile
    )

    row_strength = float(
        np.mean(
            row_profile
        )
    )

    col_strength = float(
        np.mean(
            col_profile
        )
    )

    # --------------------------------------------------------
    # Detect repeated dots / small repeated blobs.
    # --------------------------------------------------------

    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )[1]

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        8
    )

    small_components = []

    for i in range(
        1,
        num_labels
    ):

        area = stats[i, cv2.CC_STAT_AREA]

        if (
            8 <= area <= 0.025 * gray.size
        ):
            small_components.append(
                area
            )

    dot_score = (
        len(small_components)
        / 80.0
    )

    # --------------------------------------------------------
    # Hough line orientation evidence.
    # --------------------------------------------------------

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=45,
        minLineLength=35,
        maxLineGap=5
    )

    horizontal = 0
    vertical = 0
    diagonal = 0

    if lines is not None:

        for raw_line in lines:

            line = np.asarray(raw_line).reshape(-1)

            if line.size < 4:
                continue

            x1, y1, x2, y2 = line[:4].astype(float)

            angle = abs(
                np.degrees(
                    np.arctan2(
                        y2 - y1,
                        x2 - x1
                    )
                )
            )

            if (
                angle <= 12
                or angle >= 168
            ):
                horizontal += 1

            elif (
                78 <= angle <= 102
            ):
                vertical += 1

            else:
                diagonal += 1

    # --------------------------------------------------------
    # Use image gradients to determine line orientation.
    # A vertical stripe creates intensity changes across X.
    # A horizontal stripe creates intensity changes across Y.
    # --------------------------------------------------------

    gx_strength = float(
        np.abs(
            np.diff(
                gray.astype(np.float32),
                axis=1
            )
        ).mean()
    )

    gy_strength = float(
        np.abs(
            np.diff(
                gray.astype(np.float32),
                axis=0
            )
        ).mean()
    )

    # --------------------------------------------------------
    # Strong two-direction repetition = checked/grid.
    # --------------------------------------------------------

    if (
        row_period >= 0.28
        and col_period >= 0.28
        and (
            (
                horizontal >= 3
                and vertical >= 3
            )
            or (
                row_strength >= 8
                and col_strength >= 8
            )
        )
    ):
        return [
            "checked / grid"
        ]

    # --------------------------------------------------------
    # Vertical linear/striped pattern.
    # This also catches subtle low-resolution vertical patterns.
    # --------------------------------------------------------

    if (
        (
            col_period >= 0.28
            and col_period > row_period * 0.92
        )
        and gx_strength >= gy_strength * 1.25
    ):
        return [
            "striped"
        ]

    # --------------------------------------------------------
    # Horizontal linear/striped pattern.
    # --------------------------------------------------------

    if (
        (
            row_period >= 0.28
            and row_period > col_period * 0.92
        )
        and gy_strength >= gx_strength * 1.25
    ):
        return [
            "striped"
        ]

    # --------------------------------------------------------
    # Very subtle repeated woven/linear structure.
    # This is useful for low-resolution textile examples where
    # Hough lines are too weak to form long straight segments.
    # --------------------------------------------------------

    if (
        row_period >= 0.42
        and col_period >= 0.42
        and (
            gx_strength >= gy_strength * 1.20
            or gy_strength >= gx_strength * 1.20
        )
    ):
        if gx_strength >= gy_strength:
            return [
                "striped"
            ]

        return [
            "striped"
        ]

    # --------------------------------------------------------
    # Dotted pattern.
    # --------------------------------------------------------

    if (
        dot_score >= 0.45
        and row_period >= 0.16
        and col_period >= 0.16
    ):
        return [
            "dotted"
        ]

    # --------------------------------------------------------
    # Diagonal geometric repetition.
    # --------------------------------------------------------

    if (
        diagonal >= 8
        and max(
            row_period,
            col_period
        ) >= 0.22
    ):
        return [
            "geometric"
        ]

    # --------------------------------------------------------
    # Semantic pattern classification.
    # Only report floral/geometric when CLIP evidence is strong.
    # --------------------------------------------------------

    image_features = get_image_embedding(
        image_path
    )

    semantic_patterns = [
        "floral",
        "geometric",
        "plain surface"
    ]

    text_features = get_text_embeddings(
        [
            "a photograph showing a {}",
            "a photograph showing a {}",
            "a photograph showing a {}"
        ]
        if False else [
            "a photograph showing a floral pattern",
            "a photograph showing a geometric pattern",
            "a photograph showing a plain surface"
        ]
    )

    scores = (
        image_features @ text_features.T
    )[0]

    ranked = sorted(
        [
            (
                float(scores[i].item()),
                semantic_patterns[i]
            )
            for i in range(
                len(semantic_patterns)
            )
        ],
        reverse=True
    )

    if ranked:

        best_score, best_label = ranked[0]

        second_score = (
            ranked[1][0]
            if len(ranked) > 1
            else -1.0
        )

        if (
            best_label != "plain surface"
            and best_score >= 0.30
            and best_score - second_score >= 0.015
        ):
            return [
                best_label
            ]

    # --------------------------------------------------------
    # If there is no reliable regular structure, call it plain.
    # --------------------------------------------------------

    if (
        row_period < 0.20
        and col_period < 0.20
        and dot_score < 0.25
    ):
        return [
            "plain / no strong regular pattern"
        ]

    return []


def detect_edges(image_path):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        return []

    image = cv2.resize(
        image,
        (320, 320),
        interpolation=cv2.INTER_AREA
    )

    blurred = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        60,
        150
    )

    ratio = (
        float(np.count_nonzero(edges))
        / float(edges.size)
    )

    if ratio >= 0.18:
        return [
            "strong edges",
            "dense contours"
        ]

    if ratio >= 0.09:
        return [
            "clear edges"
        ]

    return [
        "soft edges"
    ]


def detect_spatial_size_depth(
    detections,
    width,
    height
):
    """
    Spatial relations and relative size come only from real
    bounding boxes.

    Depth is described as relative depth only; this is not
    claimed to be metric monocular depth.
    """

    spatial = []
    sizes = []
    depths = []

    if not detections:
        return (
            spatial,
            sizes,
            depths
        )

    sorted_area = sorted(
        detections,
        key=lambda x: x["area_ratio"],
        reverse=True
    )

    # --------------------------------------------------------
    # Relative size.
    # --------------------------------------------------------

    for detection in sorted_area[:5]:

        ratio = detection[
            "area_ratio"
        ]

        label = detection[
            "label"
        ]

        if ratio >= 0.25:

            sizes.append(
                f"large {label}"
            )

        elif ratio <= 0.04:

            sizes.append(
                f"small {label}"
            )

        else:

            sizes.append(
                f"medium-sized {label}"
            )

    # --------------------------------------------------------
    # Relative foreground/background.
    # --------------------------------------------------------

    if len(sorted_area) >= 2:

        largest = sorted_area[0]
        smallest = sorted_area[-1]

        if (
            largest["area_ratio"]
            >= smallest["area_ratio"] * 3
        ):

            depths.append(
                f"{largest['label']} foreground"
            )

            depths.append(
                f"{smallest['label']} background"
            )

    # --------------------------------------------------------
    # Actual box-center relationships.
    # --------------------------------------------------------

    for i in range(
        min(len(detections), 6)
    ):

        for j in range(
            i + 1,
            min(len(detections), 6)
        ):

            a = detections[i]
            b = detections[j]

            ax, ay = a["center"]
            bx, by = b["center"]

            dx = abs(
                ax - bx
            ) / max(
                1.0,
                width
            )

            dy = abs(
                ay - by
            ) / max(
                1.0,
                height
            )

            if (
                dx < 0.20
                and dy < 0.20
            ):

                spatial.append(
                    f"{a['label']} near {b['label']}"
                )

            elif (
                dx < 0.32
                and dy < 0.16
            ):

                spatial.append(
                    f"{a['label']} beside {b['label']}"
                )

            elif (
                dy < 0.18
                and dx >= 0.32
            ):

                if ax < bx:

                    spatial.append(
                        f"{a['label']} left of {b['label']}"
                    )

                else:

                    spatial.append(
                        f"{a['label']} right of {b['label']}"
                    )

            elif (
                dx < 0.18
                and dy >= 0.28
            ):

                if ay < by:

                    spatial.append(
                        f"{a['label']} above {b['label']}"
                    )

                else:

                    spatial.append(
                        f"{a['label']} below {b['label']}"
                    )

    return (
        _unique_labels(
            spatial
        )[:6],

        _unique_labels(
            sizes
        )[:5],

        _unique_labels(
            depths
        )[:3]
    )


@torch.no_grad()
def detect_pose_action(image_path):

    if pose_model is None:
        return []

    image = Image.open(
        image_path
    ).convert("RGB")

    results = pose_model(
        image,
        conf=0.45,
        verbose=False
    )

    if (
        not results
        or results[0].keypoints is None
    ):
        return []

    keypoints = results[0].keypoints.xy

    if (
        keypoints is None
        or len(keypoints) == 0
    ):
        return []

    actions = []

    for person_points in keypoints:

        pts = (
            person_points
            .cpu()
            .numpy()
        )

        if pts.shape[0] < 17:
            continue

        # COCO:
        # shoulders 5/6
        # hips 11/12
        # knees 13/14
        # ankles 15/16

        shoulder_y = np.nanmean(
            [
                pts[5][1],
                pts[6][1]
            ]
        )

        hip_y = np.nanmean(
            [
                pts[11][1],
                pts[12][1]
            ]
        )

        knee_y = np.nanmean(
            [
                pts[13][1],
                pts[14][1]
            ]
        )

        ankle_y = np.nanmean(
            [
                pts[15][1],
                pts[16][1]
            ]
        )

        if not np.isfinite(
            hip_y
        ):
            continue

        if (
            np.isfinite(knee_y)
            and abs(knee_y - hip_y)
            < 0.12 * image.height
        ):

            actions.append(
                "sitting / crouching"
            )

        elif (
            np.isfinite(ankle_y)
            and ankle_y > knee_y
        ):

            actions.append(
                "standing"
            )

    return _unique_labels(
        actions
    )[:3]


@torch.no_grad()
def detect_scene_material_semantics(
    image_path,
    detections,
    reference_captions=None
):
    """
    Scene is whole-image.
    Material and semantic concepts are classified only when
    evidence is sufficiently strong.
    """

    features = {}

    image_features = get_image_embedding(
        image_path
    )

    # --------------------------------------------------------
    # Scene
    # --------------------------------------------------------

    scenes = [
        "street",
        "beach",
        "market",
        "restaurant",
        "kitchen",
        "park",
        "playground",
        "office",
        "classroom",
        "farm",
        "forest",
        "mountain",
        "city",
        "village",
        "indoor scene",
        "outdoor scene"
    ]

    scene = clip_rank_image(
        image_features,
        scenes,
        "a photograph of {}",
        threshold=0.28,
        margin=0.010,
        top_k=1
    )

    caption_scenes = (
        dataset_labels_from_captions(
            reference_captions
        ).get("Scene")
        if reference_captions
        else None
    )

    # Scene is taken from the image. Caption scene names are
    # kept only when they agree with the visual CLIP result.
    if scene:
        features["Scenes"] = scene
    elif caption_scenes:
        features["Scenes"] = caption_scenes[:1]

    # --------------------------------------------------------
    # Material — real detected region only.
    # --------------------------------------------------------

    if detections:

        image = Image.open(
            image_path
        ).convert("RGB")

        materials = [
            "wood",
            "metal",
            "glass",
            "plastic",
            "fabric",
            "leather",
            "stone",
            "concrete"
        ]

        material_found = []

        for detection in detections[:6]:

            crop = _crop_from_detection(
                image,
                detection
            )

            label, score = clip_rank_crop(
                crop,
                materials,
                "a close-up photograph of an object made of {}",
                threshold=0.30,
                margin=0.012
            )

            if label is not None:
                material_found.append(
                    label
                )

        material_found = _unique_labels(
            material_found
        )

        if material_found:
            features["Materials"] = (
                material_found[:4]
            )

    # --------------------------------------------------------
    # Semantic concepts.
    # Use scene/object evidence first, and CLIP only as a
    # secondary whole-image signal.
    # --------------------------------------------------------

    semantic = [
        "shopping",
        "playing",
        "cooking",
        "eating",
        "working",
        "traveling",
        "sports",
        "socializing",
        "outdoor activity",
        "daily life"
    ]

    semantic_result = clip_rank_image(
        image_features,
        semantic,
        "a photograph showing people doing {}",
        threshold=0.30,
        margin=0.015,
        top_k=1
    )

    # Do not report people-based actions when there is no person.
    has_person = any(
        d["label"] == "person"
        for d in detections
    )

    caption_semantics = semantics_from_captions(
        reference_captions
    )

    if semantic_result and has_person:
        features["Semantic Concepts"] = (
            semantic_result
        )
    elif caption_semantics and has_person:
        features["Semantic Concepts"] = caption_semantics[:1]

    return features


def detect_visual_features(
    image_path,
    reference_captions=None,
    image_contents=None
):

    image = Image.open(
        image_path
    ).convert("RGB")

    width, height = image.size

    detections = get_grounded_yolo_detections(
        image_path
    )

    features = {}

    # --------------------------------------------------------
    # 1. Colors
    # --------------------------------------------------------

    colors = detect_dominant_colors(
        image_path
    )

    if colors:
        features["Colors"] = colors

    # --------------------------------------------------------
    # 2. Shapes
    # --------------------------------------------------------

    shapes = detect_shapes_from_objects(
        image_path,
        detections
    )

    if shapes:
        features["Shapes"] = shapes

    # --------------------------------------------------------
    # 3. Textures
    # --------------------------------------------------------

    textures = detect_texture(
        image_path,
        detections
    )

    if textures:
        features["Textures"] = textures

    # --------------------------------------------------------
    # 4. Patterns
    # --------------------------------------------------------

    patterns = detect_patterns(
        image_path
    )

    if patterns:
        features["Patterns"] = patterns

    # --------------------------------------------------------
    # 5. Edges
    # --------------------------------------------------------

    edges = detect_edges(
        image_path
    )

    if edges:
        features["Edges"] = edges

    # --------------------------------------------------------
    # 6. Objects
    # --------------------------------------------------------

    object_labels = []

    if detections:
        object_labels.extend(
            [
                d["label"]
                for d in detections
            ]
        )

    if image_contents:
        for category in (
            "Animals",
            "Food / Vegetables",
            "Objects"
        ):
            object_labels.extend(
                _content_labels(
                    image_contents.get(category, [])
                )
            )

    object_labels = _unique_labels(
        object_labels
    )[:12]

    if object_labels:
        features["Objects"] = object_labels

    # --------------------------------------------------------
    # 7-10. Scene / Spatial / Size / Depth
    # --------------------------------------------------------

    spatial, sizes, depths = (
        detect_spatial_size_depth(
            detections,
            width,
            height
        )
    )

    if spatial:
        features[
            "Spatial Relationships"
        ] = spatial

    if sizes:
        features[
            "Size"
        ] = sizes

    if depths:
        features[
            "Depth"
        ] = depths

    # --------------------------------------------------------
    # 11. Pose / Action
    # --------------------------------------------------------

    pose = detect_pose_action(
        image_path
    )

    if pose:
        features[
            "Pose / Action"
        ] = pose

    # --------------------------------------------------------
    # 12-13. Materials / Semantic concepts
    # --------------------------------------------------------

    semantic_features = (
        detect_scene_material_semantics(
            image_path,
            detections,
            reference_captions=reference_captions
        )
    )

    features.update(
        semantic_features
    )

    if (
        "Scene" not in features
        and semantic_features.get("Scenes")
    ):
        features["Scene"] = semantic_features["Scenes"]

    # Prefer the visual scene already measured from the image.
    # Dataset scene names are a fallback only.
    if "Scene" not in features:
        if image_contents and image_contents.get("Scene"):
            features["Scene"] = _content_labels(
                image_contents["Scene"]
            )[:2]
        elif image_contents and image_contents.get("Environment"):
            features["Scene"] = _content_labels(
                image_contents["Environment"]
            )[:2]

    # CLIP scene is a fallback only when the dataset captions
    # did not already name a scene.
    if "Scene" not in features:

        image_features = get_image_embedding(
            image_path
        )

        scene_labels = [
            "street",
            "beach",
            "market",
            "restaurant",
            "kitchen",
            "park",
            "playground",
            "office",
            "classroom",
            "farm",
            "forest",
            "mountain",
            "city",
            "village",
            "indoor scene",
            "outdoor scene"
        ]

        scene = clip_rank_image(
            image_features,
            scene_labels,
            "a photograph of {}",
            threshold=0.28,
            margin=0.010,
            top_k=1
        )

        if scene:
            features["Scene"] = scene

    if "Scenes" in features:
        if "Scene" not in features:
            features["Scene"] = features["Scenes"]
        features.pop("Scenes", None)

    # --------------------------------------------------------
    # Stable display order.
    # --------------------------------------------------------

    order = [
        "Colors",
        "Shapes",
        "Textures",
        "Patterns",
        "Edges",
        "Objects",
        "Scene",
        "Spatial Relationships",
        "Size",
        "Depth",
        "Pose / Action",
        "Materials",
        "Semantic Concepts"
    ]

    ordered = {
        category: features[category]
        for category in order
        if (
            category in features
            and features[category]
        )
    }

    print()
    print("========================================")
    print("GROUNDED VISUAL FEATURES")
    print("========================================")

    for category, items in ordered.items():

        print(
            category + ":",
            ", ".join(items)
        )

    print("========================================")

    return ordered


FEATURE_DESCRIPTIONS = {
    "Colors": "Dominant colors in the image",
    "Shapes": "Geometric / object shapes",
    "Textures": "Surface appearance",
    "Patterns": "Repeated visual structures",
    "Edges": "Strength / density of boundaries",
    "Objects": "Detected physical objects",
    "Scene": "Overall environment",
    "Spatial Relationships": "Relative positions of objects",
    "Size": "Relative object size in the image",
    "Depth": "Approximate foreground / background relationship",
    "Pose / Action": "Person posture / action",
    "Materials": "Possible surface / material",
    "Semantic Concepts": "Higher-level activity / concept"
}


# ============================================================
# Step 19 — Professional website HTML
# ============================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<title>
VisualRAG - Image Text Matching
</title>

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<style>

/* ---------------------------------------------------------
   Global theme
--------------------------------------------------------- */

* {
    box-sizing: border-box;
}

body {
    font-family: "Segoe UI", Calibri, system-ui, sans-serif;
    background: #eef3f8;
    color: #132033;
    margin: 0;
    padding: 28px 4vw 56px;
}

.container {
    width: min(1480px, 100%);
    max-width: 1480px;
    margin: 0 auto;
}


/* ---------------------------------------------------------
   Header
--------------------------------------------------------- */

.header {

    text-align: center;

    margin-bottom: 30px;
}

.header h1 {

    margin: 0;

    font-size: 34px;

    color: #163a5c;

    letter-spacing: -0.5px;
}

.header p {

    margin-top: 10px;

    color: #64748b;

    font-size: 15px;
}

.nav-tabs {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-bottom: 24px;
}

.nav-tab {
    text-decoration: none;
    color: #1f4e79;
    background: white;
    border: 1px solid #dbe4ee;
    border-radius: 999px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 14px;
}

.nav-tab.active {
    background: #1f4e79;
    color: white;
    border-color: #1f4e79;
}


/* ---------------------------------------------------------
   Cards
--------------------------------------------------------- */

.card {

    background:
        white;

    padding: 28px;

    margin-bottom: 24px;

    border-radius: 12px;

    border:
        1px solid #e2e8f0;

    box-shadow:
        0 4px 16px
        rgba(15, 23, 42, 0.06);
}


/* ---------------------------------------------------------
   Upload section
--------------------------------------------------------- */

.upload-box {

    text-align: center;

    padding: 36px;
}

.upload-box h2 {

    margin-top: 0;

    margin-bottom: 8px;

    font-size: 24px;

    color: #1e293b;
}

.upload-subtitle {

    color: #64748b;

    font-size: 15px;

    line-height: 1.6;

    margin:
        0 auto 25px;

    max-width: 650px;
}


.file-input-wrapper {

    padding: 22px;

    border:
        2px dashed
        #cbd5e1;

    border-radius: 10px;

    background:
        #f8fafc;

    margin-bottom: 5px;
}

.file-input-wrapper input {

    font-size: 14px;

    max-width: 100%;
}


/* ---------------------------------------------------------
   Get Caption button
--------------------------------------------------------- */

.get-caption-btn {

    margin-top: 15px;

    padding:
        13px 34px;

    border: none;

    border-radius: 8px;

    cursor: pointer;

    font-size: 16px;

    font-weight: 600;

    background:
        #1f4e79;

    color:
        white;

    transition:
        background 0.2s ease,
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

.get-caption-btn:hover {

    background:
        #163a5c;

    transform:
        translateY(-1px);

    box-shadow:
        0 5px 14px
        rgba(15, 23, 42, 0.18);
}

.get-caption-btn:active {

    transform:
        translateY(0);
}


/* ---------------------------------------------------------
   Uploaded image
--------------------------------------------------------- */

.uploaded-image {

    display: block;

    max-width: 100%;

    max-height: 550px;

    margin: auto;

    border-radius: 10px;

    box-shadow:
        0 4px 15px
        rgba(0,0,0,0.08);
}


/* ---------------------------------------------------------
   Caption results
--------------------------------------------------------- */

.caption {

    padding: 18px;

    margin: 12px 0;

    border-radius: 9px;

    background:
        #f8fafc;

    border:
        1px solid #e2e8f0;
}

.caption.best {

    border:
        2px solid #2e7d32;

    background:
        #f0f9f1;
}

.caption-number {

    color:
        #475569;

    font-weight:
        600;
}


.score {

    font-weight:
        600;

    color:
        #334155;

    margin-top:
        10px;
}

.score-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 12px;
}

.metric {
    background: white;
    border: 1px solid #dbe2ea;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    color: #334155;
}

.metric strong {
    display: block;
    font-size: 11px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 3px;
}

.metric-badge {
    display: inline-block;
    margin: 6px 6px 0 0;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
}

.badge-cosine {
    background: #ecfdf3;
    color: #166534;
    border: 1px solid #86efac;
}

.badge-dot {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #93c5fd;
}

.badge-euclidean {
    background: #fff7ed;
    color: #c2410c;
    border: 1px solid #fdba74;
}

.metric-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 16px;
}

.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px;
}

.metric-card h3 {
    margin: 0 0 8px 0;
    font-size: 15px;
    color: #334155;
}

.metric-card .best-text {
    font-size: 18px;
}

.feature-help {
    margin: 0 0 10px 0;
    font-size: 12px;
    color: #64748b;
    line-height: 1.45;
}


/* ---------------------------------------------------------
   Best caption
--------------------------------------------------------- */

.best-caption-card {

    border:
        2px solid #2e7d32;

    background:
        #f6fbf6;
}

.best-title {

    color:
        #216e39;

    font-size:
        26px;

    font-weight:
        700;

    margin-bottom:
        10px;
}

.best-text {

    font-size:
        21px;

    line-height:
        1.6;

    color:
        #172033;
}

.best-score {

    font-size:
        17px;

    color:
        #216e39;
}


/* ---------------------------------------------------------
   Image contents
--------------------------------------------------------- */

.concept-grid {

    display:
        grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap:
        18px;
}


.concept-box {

    background:
        #f8fafc;

    padding:
        18px;

    border-radius:
        10px;

    border:
        1px solid #e2e8f0;
}

.concept-box h3 {

    margin-top:
        0;

    color:
        #334155;

    font-size:
        17px;
}


.concept-item {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 12px;
    margin: 0 0 10px 0;
}

.concept {

    display:
        inline-block;

    background:
        #f8fafc;

    padding:
        8px 14px;

    margin:
        0 0 8px 0;

    border-radius:
        20px;

    border:
        1px solid #dbe2ea;

    font-size:
        14px;

    color:
        #334155;

    font-weight: 600;
}


/* ---------------------------------------------------------
   Notes
--------------------------------------------------------- */

.note {

    color:
        #64748b;

    font-size:
        14px;

    line-height:
        1.6;
}


/* ---------------------------------------------------------
   Comprehensive visual features
--------------------------------------------------------- */

.feature-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 18px;
}

.feature-box {
    background: #f8fafc;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}

.feature-box h3 {
    margin: 0 0 12px 0;
    color: #334155;
    font-size: 16px;
}

.feature-chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: white;
    padding: 8px 13px;
    margin: 4px;
    border-radius: 8px;
    border: 1px solid #dbe2ea;
    font-size: 14px;
    color: #334155;
}

.color-dot {
    width: 13px;
    height: 13px;
    border-radius: 50%;
    display: inline-block;
    border: 1px solid #cbd5e1;
}

.color-red { background: #ef4444; }
.color-orange { background: #f97316; }
.color-yellow { background: #eab308; }
.color-green { background: #22c55e; }
.color-cyan { background: #06b6d4; }
.color-blue { background: #3b82f6; }
.color-purple { background: #8b5cf6; }
.color-pink { background: #ec4899; }
.color-white { background: #ffffff; }
.color-gray { background: #9ca3af; }
.color-black { background: #111827; }
.color-brown { background: #8b5e3c; }
.color-beige { background: #d6c4a8; }

/* ---------------------------------------------------------
   Mobile
--------------------------------------------------------- */

@media(max-width: 700px) {

    body {

        padding:
            20px 12px;
    }

    .header h1 {

        font-size:
            28px;
    }

    .card {

        padding:
            20px;
    }

    .concept-grid {

        grid-template-columns:
            1fr;
    }

    .best-text {

        font-size:
            18px;
    }

}

</style>

</head>


<body>

<div class="container">


<!-- =====================================================
     Header
====================================================== -->

<div class="header">

<h1>
VisualRAG
</h1>

<p>
Knowledge-Guided Image-Text Matching
</p>

</div>

<div class="nav-tabs">
<a class="nav-tab active" href="/">Image Matching</a>
<a class="nav-tab" href="/video">Video Understanding</a>
</div>


<!-- =====================================================
     Upload
====================================================== -->

<div class="card upload-box">

<h2>
Image Captioning
</h2>

<p class="upload-subtitle">

Upload an image to find the most relevant
Flickr30K caption using VisualRAG
image-text matching.

</p>


<form
    method="POST"
    enctype="multipart/form-data"
>

<div class="file-input-wrapper">

<input
    type="file"
    name="image"
    accept="image/*"
    required
>

</div>


<button
    type="submit"
    class="get-caption-btn"
>

Get Caption

</button>


</form>

</div>


{% if image %}


<!-- =====================================================
     Uploaded Image
====================================================== -->

<div class="card">

<h2>
Uploaded Image
</h2>

<img
    src="/uploaded/{{ image }}"
    class="uploaded-image"
>

</div>


<!-- =====================================================
     Flickr30K Captions
====================================================== -->

<div class="card">

<h2>
Flickr30K Captions
</h2>

<p class="note">

Each Flickr30K caption is compared with the
uploaded image using CLIP embeddings.
<strong>Image-text similarity</strong> is the CLIP
matching logit (cosine × temperature).
<strong>Cosine similarity</strong> is the main
normalized matching score.
<strong>Dot product</strong> is embedding similarity
on the raw CLIP vectors.
<strong>Euclidean distance</strong> is an alternative
embedding distance (smaller is closer).

</p>


{% for item in ranked_results %}

<div class="
    caption
    {% if best_cosine and item.number == best_cosine.number %}
        best
    {% endif %}
">


<div class="caption-number">

Caption {{ item.number }}

</div>


<p>
{{ item.caption }}
</p>


<div class="score-row">

<div class="metric">
<strong>Image-text similarity</strong>
{{ "%.4f"|format(item.image_text_similarity) }}
</div>

<div class="metric">
<strong>Cosine similarity</strong>
{{ "%.4f"|format(item.cosine) }}
</div>

<div class="metric">
<strong>Dot product</strong>
{{ "%.4f"|format(item.dot_product) }}
</div>

<div class="metric">
<strong>Euclidean distance</strong>
{{ "%.4f"|format(item.euclidean) }}
</div>

</div>


{% if best_cosine and item.number == best_cosine.number %}
<span class="metric-badge badge-cosine">
Best according to cosine similarity
</span>
{% endif %}

{% if best_dot and item.number == best_dot.number %}
<span class="metric-badge badge-dot">
Best according to dot product
</span>
{% endif %}

{% if best_euclidean and item.number == best_euclidean.number %}
<span class="metric-badge badge-euclidean">
Best according to Euclidean distance
</span>
{% endif %}


</div>

{% endfor %}

</div>


<!-- =====================================================
     Best Caption
====================================================== -->

<div class="card best-caption-card">

<div class="best-title">

⭐ BEST CAPTIONS

</div>

<p class="note">

The same CLIP embeddings are scored four ways:
image-text similarity, cosine, dot product, and
Euclidean distance. A caption can win more than
one metric.

</p>

<div class="metric-grid">

{% if best_cosine %}
<div class="metric-card">
<h3>According to cosine similarity, this caption is best</h3>
<p class="best-text">{{ best_cosine.caption }}</p>
<div class="score best-score">
Cosine similarity: {{ "%.4f"|format(best_cosine.cosine) }}
<br>
Image-text similarity: {{ "%.4f"|format(best_cosine.image_text_similarity) }}
</div>
</div>
{% endif %}

{% if best_dot %}
<div class="metric-card">
<h3>According to dot product, this caption is best</h3>
<p class="best-text">{{ best_dot.caption }}</p>
<div class="score best-score">
Dot product: {{ "%.4f"|format(best_dot.dot_product) }}
</div>
</div>
{% endif %}

{% if best_euclidean %}
<div class="metric-card">
<h3>According to Euclidean distance, this caption is best</h3>
<p class="best-text">{{ best_euclidean.caption }}</p>
<div class="score best-score">
Euclidean distance: {{ "%.4f"|format(best_euclidean.euclidean) }}
(lower is closer)
</div>
</div>
{% endif %}

</div>

</div>


<!-- =====================================================
     Comprehensive Visual Features
====================================================== -->

<div class="card">

<h2>VISUAL FEATURES</h2>

<p class="note">

Visual features are measured from the uploaded
image: colors, textures, patterns, edges, pose,
detected objects, spatial layout, size, depth,
materials, and scene.

</p>

<div class="feature-grid">

{% for category, items in visual_features.items() %}

<div class="feature-box">

<h3>{{ category }}</h3>

{% if feature_descriptions and feature_descriptions.get(category) %}
<p class="feature-help">
{{ feature_descriptions.get(category) }}
</p>
{% endif %}

<div>
{% for item in items %}

{% if category == "Colors" %}
<span class="feature-chip">
<span class="color-dot color-{{ item|lower }}"></span>
{{ item }}
</span>
{% else %}
<span class="feature-chip">{{ item }}</span>
{% endif %}

{% endfor %}
</div>

</div>

{% endfor %}

</div>

</div>


<!-- =====================================================
     What's in the Image
====================================================== -->

<div class="card">

<h2>
WHAT'S IN THE IMAGE
</h2>


<p class="note">

People, clothing, objects, animals, food, and scene
are taken from the picture itself: detected regions
plus visual checks. Only details that appear in the
image are listed.

</p>


<div class="concept-grid">


{% for category, items in image_contents.items() %}


<div class="concept-box">


<h3>
{{ category }}
</h3>


<div>


{% for item in items %}

<span class="feature-chip">
{{ item }}
</span>

{% endfor %}


</div>


</div>


{% endfor %}


</div>

</div>




{% endif %}


</div>

</body>

</html>

"""


# ============================================================
# Step 20 — Home page
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    if request.method == "GET":

        return render_template_string(
            HTML
        )


    uploaded_file = request.files.get(
        "image"
    )


    if uploaded_file is None:

        return "No image uploaded."


    if uploaded_file.filename == "":

        return "No image selected."


    filename = os.path.basename(
        uploaded_file.filename
    ).replace("\\", "/").split("/")[-1]

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    uploaded_file.save(
        image_path
    )

    print()
    print(
        "Uploaded image:",
        filename
    )

    flickr_id = extract_flickr_id(filename)
    print("Flickr30K id:", flickr_id)

    # --------------------------------------------------------
    # Get Flickr30K captions
    # --------------------------------------------------------

    captions = get_captions(
        filename
    )


    print(
        "Number of captions:",
        len(captions)
    )


    # --------------------------------------------------------
    # Find highest-related caption
    # --------------------------------------------------------

    (
        best_caption,
        best_score,
        ranked_results,
        best_cosine,
        best_dot,
        best_euclidean
    ) = find_best_caption(
        image_path,
        captions
    )


    if ranked_results and best_cosine:

        best_number = best_cosine["number"]

    else:

        best_number = 0


    # --------------------------------------------------------
    # Detect image contents
    # --------------------------------------------------------

    print()

    print(
        "Analyzing what's in the image..."
    )


    image_contents = (
        detect_image_contents(
            image_path,
            reference_captions=captions
        )
    )


    print(
        "WHAT'S IN THE IMAGE:",
        image_contents
    )


    # --------------------------------------------------------
    # Comprehensive visual feature analysis
    # --------------------------------------------------------

    print()
    print("Analyzing comprehensive visual features...")

    visual_features = detect_visual_features(
        image_path,
        reference_captions=captions,
        image_contents=image_contents
    )


    # --------------------------------------------------------
    # Render website
    # --------------------------------------------------------

    return render_template_string(

        HTML,

        image=filename,

        captions=captions,

        ranked_results=ranked_results,

        best_caption=best_caption,

        best_score=best_score,

        best_number=best_number,

        image_contents=image_contents,

        visual_features=visual_features,

        feature_descriptions=FEATURE_DESCRIPTIONS,

        best_cosine=best_cosine,

        best_dot=best_dot,

        best_euclidean=best_euclidean

    )


# ============================================================
# Step 21 — Serve uploaded images
# ============================================================

@app.route(
    "/uploaded/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


def video_vision():
    return {
        "yolo": yolo_model,
        "pose": pose_model,
        "device": device,
        "get_image_embedding": get_image_embedding,
        "get_text_embeddings": get_text_embeddings,
        "get_pil_embedding": get_pil_embedding,
        "clip_rank_image": clip_rank_image,
        "detect_image_contents": detect_image_contents,
    }


ALLOWED_VIDEO_EXT = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"
}


def render_video_page(index=None, error=None, job_id=None):
    return render_template_string(
        VIDEO_HTML,
        index=index,
        error=error,
        job_id=job_id,
        library=list_video_indexes(),
    )


def _job_snapshot(job_id):
    with _video_jobs_lock:
        job = _video_jobs.get(job_id)
        return dict(job) if job else None


def _set_job(job_id, **kwargs):
    with _video_jobs_lock:
        job = _video_jobs.setdefault(job_id, {"job_id": job_id})
        job.update(kwargs)
        return dict(job)


def _run_video_job(job_id, video_path, video_id, stored_name, original_name):
    def progress(message):
        _set_job(job_id, status="running", message=message)

    try:
        with _video_analyze_lock:
            analyze_video(
                video_path,
                video_id,
                stored_name,
                video_vision(),
                original_name=original_name,
                progress=progress,
            )
        _set_job(
            job_id,
            status="done",
            message="Analysis complete.",
            video_id=video_id,
        )
    except Exception as exc:
        print("Video analysis error:", exc)
        _set_job(job_id, status="error", message=str(exc), error=str(exc))


def _run_transcript_job(job_id, video_id):
    _set_job(job_id, status="running", message="Transcribing speech...")
    try:
        index = load_video_index(video_id)
        if index is None:
            raise RuntimeError("Video index not found.")
        with _video_analyze_lock:
            update_index_transcript(index)
        _set_job(
            job_id,
            status="done",
            message="Transcription complete.",
            video_id=video_id,
        )
    except Exception as exc:
        print("Transcript error:", exc)
        _set_job(job_id, status="error", message=str(exc), error=str(exc))


@app.route("/video", methods=["GET", "POST"])
def video_home():
    if request.method == "GET":
        job_id = request.args.get("job")
        video_id = request.args.get("id")
        if video_id:
            index = load_video_index(video_id)
            if index:
                return render_video_page(index=index, job_id=job_id)
            return render_video_page(
                error="No saved index for that video. Upload it again.",
                job_id=job_id,
            )
        if not job_id:
            with _video_jobs_lock:
                for j_id, j_data in reversed(list(_video_jobs.items())):
                    if j_data.get("status") in ("queued", "running"):
                        job_id = j_id
                        break

        # Show the video belonging to this processing job, not an older video.
        index = None
        if job_id:
            job = _job_snapshot(job_id)
            if job and job.get("video_id"):
                index = load_video_index(job["video_id"])

        # Only show the latest saved video when no job is being processed.
        if not job_id:
            library = list_video_indexes()
            index = load_video_index(library[0]["video_id"]) if library else None

        return render_video_page(index=index, job_id=job_id)

    uploaded = request.files.get("video")
    if uploaded is None or uploaded.filename == "":
        return render_video_page(error="No video selected.")

    filename = secure_filename(os.path.basename(uploaded.filename))
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return render_video_page(
            error="Please upload an mp4, mov, avi, mkv, webm, or m4v file."
        )

    video_id = uuid.uuid4().hex[:12]
    stored_name = f"{video_id}_{filename}"
    video_path = os.path.join(VIDEO_UPLOAD_DIR, stored_name)
    uploaded.save(video_path)
    file_size = os.path.getsize(video_path)

    existing = find_matching_index(filename, file_size)
    if existing:
        try:
            os.remove(video_path)
        except OSError:
            pass
        return redirect(f"/video?id={existing['video_id']}")

    job_id = uuid.uuid4().hex[:12]
    _set_job(
        job_id,
        status="queued",
        message="Video uploaded. Starting analysis...",
        video_id=video_id,
    )
    thread = threading.Thread(
        target=_run_video_job,
        args=(job_id, video_path, video_id, stored_name, filename),
        daemon=True,
    )
    thread.start()
    return redirect(f"/video?job={job_id}")


@app.route("/video/job/<job_id>")
def video_job_status(job_id):
    job = _job_snapshot(job_id)
    if job is None:
        return jsonify({"error": "Job not found.", "status": "missing"}), 404
    return jsonify(job)


@app.route("/video/transcribe", methods=["POST"])
def video_transcribe_route():
    payload = request.get_json(silent=True) or {}
    video_id = payload.get("video_id") or request.form.get("video_id")
    index = load_video_index(video_id) if video_id else None
    if index is None:
        return jsonify({"error": "Video index not found."}), 404
    job_id = uuid.uuid4().hex[:12]
    _set_job(
        job_id,
        status="queued",
        message="Queued speech transcription...",
        video_id=video_id,
    )
    thread = threading.Thread(
        target=_run_transcript_job,
        args=(job_id, video_id),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id, "video_id": video_id})


@app.route("/video/search", methods=["POST"])
def video_search_route():
    payload = request.get_json(silent=True) or {}
    video_id = payload.get("video_id") or request.form.get("video_id")
    query = payload.get("query") or request.form.get("query") or ""
    mode = payload.get("mode") or request.form.get("mode") or "multimodal"
    index = load_video_index(video_id) if video_id else None
    if index is None:
        return jsonify({"error": "Video index not found. Analyze a video first."}), 404
    return jsonify(run_search(index, query, mode, video_vision()))


@app.route("/uploaded-video/<filename>")
def uploaded_video(filename):
    return send_from_directory(VIDEO_UPLOAD_DIR, filename)


@app.route("/video-frame/<video_id>/<filename>")
def video_frame(video_id, filename):
    return send_from_directory(
        os.path.join(VIDEO_FRAME_DIR, video_id),
        filename
    )


# ============================================================
# Step 22 — Start Flask
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "========================================"
    )

    print(
        "          VisualRAG Website"
    )

    print(
        "========================================"
    )

    print(
        "Open in browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "========================================"
    )


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )