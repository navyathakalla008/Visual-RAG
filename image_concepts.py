# Step 32 — Understand what is actually visible in the image using BLIP-2

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


# -----------------------------
# Load BLIP-2
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading BLIP-2 for image understanding...")
print("Image understanding device:", DEVICE)

processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b"
)

blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=(
        torch.float16
        if DEVICE == "cuda"
        else torch.float32
    )
).to(DEVICE)

blip_model.eval()

print("BLIP-2 image understanding loaded!")


# -----------------------------
# Get objects and visual details
# -----------------------------

def get_image_concepts(image_path, top_k=10):

    # Open image
    image = Image.open(
        image_path
    ).convert("RGB")


    # Ask BLIP-2 specifically about visible content
    question = (
        "What objects, people, animals, clothing, "
        "places, and important visible things are "
        "clearly present in this image? "
        "List only things that are visibly present. "
        "Do not guess."
    )


    # Prepare BLIP-2 input
    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt"
    )


    # Move inputs to correct device
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
        if hasattr(value, "to")
    }


    # Generate answer
    with torch.no_grad():

        generated_ids = blip_model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=5
        )


    # Convert generated answer to text
    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()


    # -----------------------------
    # Convert answer into items
    # -----------------------------

    # Remove question if model repeats it
    if answer.lower().startswith(
        question.lower()
    ):
        answer = answer[
            len(question):
        ].strip()


    # Split generated answer
    answer = answer.replace(
        "\n",
        ","
    )

    raw_items = answer.split(",")


    concepts = []


    for item in raw_items:

        item = item.strip()

        # Remove numbering
        item = item.lstrip(
            "0123456789.-) "
        )

        if not item:
            continue


        # Avoid very long sentences
        if len(item.split()) > 6:
            continue


        # Remove duplicates
        if item.lower() not in [
            x.lower()
            for x in concepts
        ]:

            concepts.append(item)


        if len(concepts) >= top_k:
            break


    return concepts