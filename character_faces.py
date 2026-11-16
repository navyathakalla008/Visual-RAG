"""Face-assisted character portrait and embedding helpers."""

import numpy as np


def create_face_models(cv2, model_dir):
    detector_path = model_dir / "face_detection_yunet_2026may.onnx"
    recognizer_path = model_dir / "face_recognition_sface_2021dec.onnx"
    missing = [str(path) for path in (detector_path, recognizer_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Character face models are missing: " + ", ".join(missing)
        )
    detector = cv2.FaceDetectorYN.create(
        str(detector_path), "", (320, 320), 0.5, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
    return detector, recognizer


def detect_character_faces(cv2, detector, recognizer, frame):
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _status, detections = detector.detect(frame)
    if detections is None:
        return []

    faces = []
    for detection in detections:
        x, y, face_width, face_height = (float(value) for value in detection[:4])
        if detection[14] < 0.55 or min(face_width, face_height) < 12:
            continue
        aligned = recognizer.alignCrop(frame, detection)
        embedding = recognizer.feature(aligned).reshape(-1).astype(np.float32)
        faces.append((detection, embedding))
    return faces


def select_face_for_person(faces, person_box):
    x1, y1, x2, y2 = (float(value) for value in person_box)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    candidates = []
    for detection, embedding in faces:
        face_x, face_y, face_width, face_height = (
            float(value) for value in detection[:4]
        )
        center_x = face_x + face_width / 2
        center_y = face_y + face_height / 2
        if not (
            x1 - width * 0.05 <= center_x <= x2 + width * 0.05
            and y1 - height * 0.08 <= center_y <= y1 + height * 0.62
        ):
            continue
        size_score = min(face_height / height, 0.25)
        score = float(detection[14]) + size_score
        candidates.append((score, detection, embedding))
    if not candidates:
        return None, None
    _score, detection, embedding = max(candidates, key=lambda row: row[0])
    return detection, embedding


def crop_character_portrait(frame, person_box, face=None):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in person_box)
    person_width = max(1.0, x2 - x1)
    person_height = max(1.0, y2 - y1)

    if face is not None:
        face_x, face_y, face_width, face_height = (float(value) for value in face[:4])
        padding_x = max(6.0, face_width * 0.55)
        left = int(face_x - padding_x)
        right = int(face_x + face_width + padding_x)
        top = int(face_y - face_height * 0.45)
        bottom = int(face_y + face_height * 2.2)
    else:
        portrait_height = max(24.0, person_height * 0.45)
        portrait_width = max(person_width * 1.15, portrait_height * 0.78)
        center_x = (x1 + x2) / 2
        left = int(center_x - portrait_width / 2)
        right = int(center_x + portrait_width / 2)
        top = int(y1 - person_height * 0.04)
        bottom = int(top + portrait_height)

    left = max(0, min(frame_width, left))
    right = max(0, min(frame_width, right))
    top = max(0, min(frame_height, top))
    bottom = max(0, min(frame_height, bottom))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


def cosine_similarity(first, second):
    first = np.asarray(first, dtype=np.float32).reshape(-1)
    second = np.asarray(second, dtype=np.float32).reshape(-1)
    if first.size != second.size or not first.size:
        return -1.0
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return -1.0
    return float(np.dot(first, second) / denominator)
