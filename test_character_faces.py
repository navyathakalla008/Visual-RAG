import numpy as np
import torch

import app
from character_faces import crop_character_portrait, select_face_for_person


def _face(x, y, width, height, confidence):
    detection = np.zeros(15, dtype=np.float32)
    detection[:4] = [x, y, width, height]
    detection[14] = confidence
    return detection


def test_select_face_for_person_uses_confident_face_inside_person_box():
    faces = [
        (_face(30, 20, 12, 12, 0.95), np.array([1.0, 0.0])),
        (_face(45, 18, 20, 20, 0.9), np.array([0.0, 1.0])),
    ]

    face, embedding = select_face_for_person(faces, (20, 10, 80, 100))

    assert face is faces[1][0]
    assert np.array_equal(embedding, faces[1][1])


def test_character_portraits_use_face_or_upper_person_crop_within_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detected_face = _face(-4, 3, 25, 30, 0.9)

    face_crop = crop_character_portrait(
        frame, (10, 0, 50, 90), detected_face
    )
    fallback_crop = crop_character_portrait(
        frame, (40, 10, 60, 90)
    )

    assert face_crop is not None
    assert 0 < face_crop.shape[0] <= 100
    assert 0 < face_crop.shape[1] <= 100
    assert fallback_crop is not None
    assert fallback_crop.shape[0] < 80


def test_character_clustering_prefers_face_identity_over_visual_similarity():
    tracks = [
        {"track_id": 1, "scene_ids": [1], "start": 0.0},
        {"track_id": 2, "scene_ids": [2], "start": 5.0},
        {"track_id": 3, "scene_ids": [3], "start": 10.0},
    ]
    visual_embeddings = {
        track_id: torch.tensor([1.0, 0.0]) for track_id in (1, 2, 3)
    }
    face_embeddings = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.95, 0.31]),
        3: np.array([0.0, 1.0]),
    }

    characters = app._cluster_character_tracks(
        tracks, visual_embeddings, [], face_embeddings
    )

    assert len(characters) == 2
    assert sorted(character["scene_ids"] for character in characters) == [[1, 2], [3]]


def test_cartoon_face_match_combines_moderate_sface_and_clip_similarity():
    tracks = [
        {"track_id": 1, "scene_ids": [1], "start": 0.0},
        {"track_id": 2, "scene_ids": [2], "start": 1.0},
    ]
    visual_embeddings = {
        1: torch.tensor([1.0, 0.0]),
        2: torch.tensor([0.86, 0.51]),
    }
    face_embeddings = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.46, 0.89]),
    }

    characters = app._cluster_character_tracks(
        tracks, visual_embeddings, [], face_embeddings
    )

    assert len(characters) == 1
    assert characters[0]["scene_ids"] == [1, 2]


def test_cooccurrence_safeguard_does_not_split_duplicate_boxes_but_separates_people():
    scene = [{"id": 1, "start": 0.0, "end": 1.0}]
    first = {
        "scene_ids": [1],
        "timestamps": [0.2],
        "boxes": [[10, 10, 50, 100]],
    }
    duplicate = {
        "scene_ids": [1],
        "timestamps": [0.2],
        "boxes": [[12, 12, 48, 98]],
    }
    distinct = {
        "scene_ids": [1],
        "timestamps": [0.2],
        "boxes": [[70, 10, 110, 100]],
    }

    assert not app._person_tracks_conflict(first, duplicate, scene)
    assert app._person_tracks_conflict(first, distinct, scene)


if __name__ == "__main__":
    test_select_face_for_person_uses_confident_face_inside_person_box()
    test_character_portraits_use_face_or_upper_person_crop_within_frame()
    test_character_clustering_prefers_face_identity_over_visual_similarity()
    test_cartoon_face_match_combines_moderate_sface_and_clip_similarity()
    test_cooccurrence_safeguard_does_not_split_duplicate_boxes_but_separates_people()
    print("Character face tests passed.")
