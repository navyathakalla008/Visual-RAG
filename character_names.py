"""Extract and conservatively associate names spoken or shown in video scenes."""

_GENERIC_NAMES = {
    "dad",
    "didi",
    "father",
    "grandma",
    "grandmother",
    "grandpa",
    "grandfather",
    "ma",
    "mom",
    "mother",
    "mr",
    "mrs",
    "ms",
    "sir",
}


def extract_character_mentions(scenes, nlp):
    mentions = {}
    pending_proper_nouns = {}
    for scene in scenes:
        scene_id = int(scene.get("id", 0))
        if not scene_id:
            continue
        contents = scene.get("contents") or {}
        text_sources = [("transcript", str(scene.get("transcript") or ""))]
        on_screen = contents.get("On-screen Text") or contents.get("on_screen_text") or []
        if isinstance(on_screen, str):
            on_screen = [on_screen]
        text_sources.extend(("on-screen text", str(text)) for text in on_screen)

        for source, text in text_sources:
            if not text.strip():
                continue
            doc = nlp(text)
            names_in_source = set()
            person_spans = {
                (entity.start, entity.end): entity
                for entity in doc.ents
                if entity.label_ == "PERSON"
            }
            for entity in person_spans.values():
                name = " ".join(entity.text.strip().split())
                if _valid_name(name):
                    names_in_source.add(name)

            proper_noun_counts = {}
            for token in doc:
                if token.pos_ != "PROPN":
                    continue
                name = token.text.strip()
                if not _valid_name(name):
                    continue
                proper_noun_counts[name.casefold()] = (
                    proper_noun_counts.get(name.casefold(), 0) + 1
                )
                pending_proper_nouns.setdefault(name.casefold(), {
                    "name": name,
                    "scene_ids": set(),
                    "mention_count": 0,
                    "scene_counts": {},
                    "sources": set(),
                })
                pending = pending_proper_nouns[name.casefold()]
                pending["scene_ids"].add(scene_id)
                pending["mention_count"] += 1
                pending["scene_counts"][scene_id] = (
                    pending["scene_counts"].get(scene_id, 0) + 1
                )
                pending["sources"].add(source)
                if token.ent_type_ == "PERSON":
                    names_in_source.add(name)

            for name in names_in_source:
                key = name.casefold()
                row = mentions.setdefault(key, {
                    "name": name,
                    "scene_ids": set(),
                    "mention_count": 0,
                    "scene_counts": {},
                    "sources": set(),
                })
                row["scene_ids"].add(scene_id)
                count = max(1, proper_noun_counts.get(key, 0))
                row["mention_count"] += count
                row["scene_counts"][scene_id] = (
                    row["scene_counts"].get(scene_id, 0) + count
                )
                row["sources"].add(source)

    for key, row in pending_proper_nouns.items():
        if row["mention_count"] < 2:
            continue
        mention = mentions.setdefault(key, {
            "name": row["name"],
            "scene_ids": set(),
            "mention_count": 0,
            "scene_counts": {},
            "sources": set(),
        })
        mention["scene_ids"].update(row["scene_ids"])
        mention["mention_count"] = max(
            mention["mention_count"], row["mention_count"]
        )
        for scene_id, count in row["scene_counts"].items():
            mention["scene_counts"][scene_id] = max(
                mention["scene_counts"].get(scene_id, 0),
                count,
            )
        mention["sources"].update(row["sources"])

    return {
        key: {
            **row,
            "scene_ids": sorted(row["scene_ids"]),
            "sources": sorted(row["sources"]),
        }
        for key, row in mentions.items()
    }


def associate_character_names(characters, mentions):
    scene_character_counts = {}
    for character in characters:
        for scene_id in character.get("scene_ids") or []:
            scene_character_counts[scene_id] = (
                scene_character_counts.get(scene_id, 0) + 1
            )

    name_scene_map = {}
    for key, mention in mentions.items():
        for scene_id in mention.get("scene_ids") or []:
            name_scene_map.setdefault(scene_id, set()).add(key)

    name_scores = {}
    for key, mention in mentions.items():
        for scene_id in mention.get("scene_ids") or []:
            if (
                scene_character_counts.get(scene_id) != 1
                or name_scene_map.get(scene_id) != {key}
            ):
                continue
            character_index = next(
                index for index, character in enumerate(characters)
                if scene_id in (character.get("scene_ids") or [])
            )
            score = int(mention.get("scene_counts", {}).get(scene_id, 1))
            name_scores.setdefault(key, {})[character_index] = (
                name_scores.setdefault(key, {}).get(character_index, 0) + score
            )

    assigned = {}
    for key, scores in name_scores.items():
        if not scores:
            continue
        top_score = max(scores.values())
        winners = [
            character_index
            for character_index, score in scores.items()
            if score == top_score
        ]
        if len(winners) == 1:
            assigned[key] = winners[0]

    names_by_character = {}
    for key, character_index in assigned.items():
        names_by_character.setdefault(character_index, []).append(key)

    for character_index, character in enumerate(characters):
        names = names_by_character.get(character_index, [])
        character_name = names[0] if len(names) == 1 else None
        character["character_name"] = (
            mentions[character_name]["name"] if character_name else None
        )
        character["name_source"] = (
            ", ".join(mentions[character_name]["sources"])
            if character_name else None
        )

    return characters


def _valid_name(name):
    if not name or name.casefold() in _GENERIC_NAMES:
        return False
    return all(
        len(part) >= 2
        and any(character.isalpha() for character in part)
        and all(character.isalpha() or character in "'’-"
                for character in part)
        for part in name.split()
    )
