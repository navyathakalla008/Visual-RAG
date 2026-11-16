# Step 15 — Build VisualRAG concept memory from Flickr30K captions

import argparse
import csv

from models.knowledge import ConceptExtractor, ConceptMemoryBank


parser = argparse.ArgumentParser()

parser.add_argument(
    "--captions",
    required=True
)

parser.add_argument(
    "--output",
    default="concept_memory.txt"
)

args = parser.parse_args()


# Create concept extractor
extractor = ConceptExtractor()

all_concepts = []


# Read Flickr30K CSV captions
with open(
    args.captions,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        caption = row["caption"].strip()

        concepts = extractor.extract(
            caption
        )

        all_concepts.append(
            concepts
        )


# Build memory bank
bank = ConceptMemoryBank()

bank.build(
    all_concepts
)


# Save memory bank
bank.save(
    args.output
)


print(
    f"Saved {len(bank.concepts)} concepts to {args.output}"
)