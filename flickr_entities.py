# Step 62 — Flickr30K Entities image understanding

import os
import re
import xml.etree.ElementTree as ET


# Flickr30K Entities folders

ENTITY_ROOT = r"D:\Flickr30K"

SENTENCES_DIR = os.path.join(
    ENTITY_ROOT,
    "Sentences"
)

ANNOTATIONS_DIR = os.path.join(
    ENTITY_ROOT,
    "Annotations"
)


# -----------------------------
# Get visible entities from XML
# -----------------------------

def get_visible_entity_ids(image_id):

    xml_file = os.path.join(
        ANNOTATIONS_DIR,
        image_id + ".xml"
    )

    if not os.path.isfile(xml_file):

        print(
            "XML annotation not found:",
            xml_file
        )

        return set()


    visible_ids = set()

    tree = ET.parse(xml_file)

    root = tree.getroot()


    for obj in root.findall(".//object"):

        name = obj.find("name")

        if name is None or not name.text:
            continue


        entity_id = name.text.strip()


        bbox = obj.find("bndbox")

        scene = obj.find("scene")

        no_box = obj.find("nobndbox")


        has_box = bbox is not None

        is_scene = (
            scene is not None
            and scene.text
            and scene.text.strip() == "1"
        )

        has_no_box = (
            no_box is not None
            and no_box.text
            and no_box.text.strip() == "1"
        )


        if (has_box or is_scene) and not has_no_box:

            visible_ids.add(entity_id)


    return visible_ids


# -----------------------------
# Get annotated phrases
# -----------------------------

def get_annotated_phrases(
    image_id,
    visible_ids
):

    txt_file = os.path.join(
        SENTENCES_DIR,
        image_id + ".txt"
    )


    if not os.path.isfile(txt_file):

        print(
            "Sentence annotation not found:",
            txt_file
        )

        return []


    phrases = []


    # Flickr30K Entities annotation format

    pattern = re.compile(
        r"\[/EN#(\d+)/[^ ]+(?:/[^ ]+)*\s+([^\]]+)\]"
    )


    with open(
        txt_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            matches = pattern.findall(line)


            for entity_id, phrase in matches:

                if entity_id not in visible_ids:
                    continue


                phrase = phrase.strip()


                phrase = re.sub(
                    r"^(a|an|the)\s+",
                    "",
                    phrase,
                    flags=re.IGNORECASE
                )


                phrase = phrase.strip(
                    " .,;:!?\"'"
                )


                if not phrase:
                    continue


                if len(phrase.split()) > 6:
                    continue


                if phrase.lower() not in [
                    p.lower()
                    for p in phrases
                ]:

                    phrases.append(
                        phrase
                    )


    return phrases


# -----------------------------
# Categorize visible entities
# -----------------------------

_ENTITY_TYPE_MAP = {
    "people": "People",
    "animals": "Animals",
    "clothing": "Clothing",
    "vehicles": "Objects",
    "instruments": "Objects",
    "other": "Objects",
    "scene": "Environment"
}

_ENTITY_PHRASE_PATTERN = re.compile(
    r"\[/EN#(\d+)/([^]\s]+)\s+([^\]]+)\]"
)


def get_categorized_entities(image_id):
    """
    Return Flickr30K Entities phrases grouped by type.
    Body-part mentions are skipped because they are not
    useful as image-content objects.
    """

    categorized = {}

    if not image_id:
        return categorized

    visible_ids = get_visible_entity_ids(
        image_id
    )

    if not visible_ids:
        return categorized

    txt_file = os.path.join(
        SENTENCES_DIR,
        image_id + ".txt"
    )

    if not os.path.isfile(txt_file):
        return categorized

    seen = set()

    with open(
        txt_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            for entity_id, type_field, phrase in (
                _ENTITY_PHRASE_PATTERN.findall(line)
            ):

                if entity_id not in visible_ids:
                    continue

                type_parts = [
                    part.lower()
                    for part in type_field.split("/")
                    if part
                ]

                if "bodyparts" in type_parts:
                    continue

                category = None

                for part in type_parts:
                    if part in _ENTITY_TYPE_MAP:
                        category = _ENTITY_TYPE_MAP[part]
                        break

                if category is None:
                    category = "Objects"

                phrase = phrase.strip()
                phrase = re.sub(
                    r"^(a|an|the)\s+",
                    "",
                    phrase,
                    flags=re.IGNORECASE
                )
                phrase = phrase.strip(" .,;:!?\"'")

                if not phrase or len(phrase.split()) > 6:
                    continue

                key = (
                    category,
                    phrase.lower()
                )

                if key in seen:
                    continue

                seen.add(key)
                categorized.setdefault(
                    category,
                    []
                ).append(phrase)

    return categorized


# -----------------------------
# Get what's in the image
# -----------------------------

def get_image_entities(
    image_path,
    top_k=10
):

    # Step 62 — Handle temporary uploaded filename

    image_id = os.path.splitext(
        os.path.basename(image_path)
    )[0]


    # Website temporarily saves:
    # temp_1000268201.jpg
    #
    # Actual Flickr30K ID:
    # 1000268201

    if image_id.startswith("temp_"):

        image_id = image_id[5:]

    match = re.search(r"(\d{6,})", image_id)

    if match:
        image_id = match.group(1)


    print(
        "Reading Flickr30K Entities:",
        image_id
    )


    visible_ids = get_visible_entity_ids(
        image_id
    )


    print(
        "Visible entity IDs:",
        len(visible_ids)
    )


    if not visible_ids:
        return []


    phrases = get_annotated_phrases(
        image_id,
        visible_ids
    )


    print(
        "Annotated image phrases:",
        phrases
    )


    return phrases[:top_k]