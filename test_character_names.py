from character_names import associate_character_names, extract_character_mentions


class _Token:
    def __init__(self, text):
        self.text = text
        self.pos_ = "PROPN" if text[:1].isupper() else "NOUN"
        self.ent_type_ = "PERSON" if text in {"Shradha", "Disha"} else ""


class _Entity:
    def __init__(self, text, start):
        self.text = text
        self.start = start
        self.end = start + 1
        self.label_ = "PERSON"


class _Document(list):
    def __init__(self, text):
        super().__init__(
            _Token(token) for token in text.replace(".", "").split()
        )
        self.ents = [
            _Entity(token.text, position)
            for position, token in enumerate(self)
            if token.ent_type_ == "PERSON"
        ]


def _fake_nlp(text):
    return _Document(text)


def test_extract_character_names_from_repeated_proper_nouns_and_person_entities():
    mentions = extract_character_mentions([
        {"id": 1, "transcript": "Shradha met Disha."},
        {"id": 2, "transcript": "Shradha spoke with Disha."},
    ], _fake_nlp)

    assert mentions["shradha"]["scene_ids"] == [1, 2]
    assert mentions["disha"]["scene_ids"] == [1, 2]
    assert "mom" not in mentions


def test_associate_name_only_when_scene_has_one_character_and_one_name():
    mentions = {
        "alice": {
            "name": "Alice",
            "scene_ids": [1],
            "sources": ["transcript"],
        }
    }
    characters = [{"scene_ids": [1, 2]}]

    associate_character_names(characters, mentions)

    assert characters[0]["character_name"] == "Alice"
    assert characters[0]["name_source"] == "transcript"
    assert "mentioned_names" not in characters[0]
    assert "mentioned_names_by_scene" not in characters[0]


def test_ambiguous_scene_keeps_names_as_mentions_without_claiming_identity():
    mentions = {
        "alice": {"name": "Alice", "scene_ids": [1], "sources": ["transcript"]}
    }
    characters = [{"scene_ids": [1]}, {"scene_ids": [1]}]

    associate_character_names(characters, mentions)

    assert all(character["character_name"] is None for character in characters)
    assert all("mentioned_names" not in character for character in characters)


def test_character_with_multiple_scene_name_candidates_is_left_unnamed():
    mentions = {
        "alice": {"name": "Alice", "scene_ids": [1], "sources": ["transcript"]},
        "bea": {"name": "Bea", "scene_ids": [1], "sources": ["transcript"]},
    }
    characters = [{"scene_ids": [1]}]

    associate_character_names(characters, mentions)

    assert characters[0]["character_name"] is None


def test_one_name_is_assigned_to_only_the_strongest_scene_candidate():
    mentions = {
        "shradha": {
            "name": "Shradha",
            "scene_ids": [1, 2],
            "scene_counts": {1: 1, 2: 3},
            "sources": ["transcript"],
        }
    }
    characters = [{"scene_ids": [1]}, {"scene_ids": [2]}]

    associate_character_names(characters, mentions)

    assert characters[0]["character_name"] is None
    assert characters[1]["character_name"] == "Shradha"


if __name__ == "__main__":
    test_extract_character_names_from_repeated_proper_nouns_and_person_entities()
    test_associate_name_only_when_scene_has_one_character_and_one_name()
    test_ambiguous_scene_keeps_names_as_mentions_without_claiming_identity()
    test_character_with_multiple_scene_name_candidates_is_left_unnamed()
    test_one_name_is_assigned_to_only_the_strongest_scene_candidate()
    print("Character name tests passed.")
