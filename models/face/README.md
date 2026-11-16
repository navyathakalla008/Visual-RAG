# Character face models

The character gallery uses these local OpenCV Zoo ONNX models:

- `face_detection_yunet_2026may.onnx` for face detection (MIT; see
  [LICENSE_YUNET](./LICENSE_YUNET)).
- `face_recognition_sface_2021dec.onnx` for face embeddings (Apache-2.0; see
  [LICENSE_SFACE](./LICENSE_SFACE)).

Upstream model sources:

- [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)

The models run locally on sampled video frames. Face matches are combined with
CLIP appearance features and a scene co-occurrence check; stylized, small, or
occluded faces can still be missed or split into separate cards.
