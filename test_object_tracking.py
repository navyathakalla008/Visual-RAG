import numpy as np
import unittest

from video_pipeline import track_objects


class TrackObjectsTests(unittest.TestCase):
    def test_uses_persistent_tracker_and_resets_at_scene_cuts(self):
        model = FakeModel(
            [
                [_detection(0)],
                [_detection(0, box=[12.0, 20.0, 42.0, 80.0])],
                [_detection(0)],
            ]
        )
        samples = [
            {"time": 0.0, "frame": np.zeros((2, 2, 3))},
            {"time": 1.0, "frame": np.zeros((2, 2, 3))},
            {"time": 2.0, "frame": np.zeros((2, 2, 3))},
        ]
        scenes = [
            {"id": 1, "start": 0.0, "end": 2.0},
            {"id": 2, "start": 2.0, "end": 3.0},
        ]

        detections, tracks = track_objects(samples, model, scenes)

        self.assertEqual(
            [item["detections"][0]["track_id"] for item in detections],
            [1, 1, 2],
        )
        self.assertEqual([track["scene_ids"] for track in tracks], [[1], [2]])
        self.assertEqual([call["tracker"] for call in model.calls], ["botsort.yaml"] * 3)
        self.assertTrue(all(call["persist"] for call in model.calls))
        self.assertEqual(model.predictor.trackers[0].reset_count, 4)

    def test_keeps_untracked_detections_without_creating_tracks(self):
        model = FakeModel([[_detection(1, track_id=None)]])
        samples = [{"time": 0.0, "frame": np.zeros((2, 2, 3))}]

        detections, tracks = track_objects(samples, model, [])

        self.assertEqual(detections[0]["detections"][0]["label"], "car")
        self.assertIsNone(detections[0]["detections"][0]["track_id"])
        self.assertEqual(tracks, [])

    def test_fuses_supplemental_detections_without_duplicate_tracks(self):
        primary = FakeModel(
            [
                [_detection(0)],
                [_detection(0, box=[12.0, 20.0, 42.0, 80.0])],
            ]
        )
        supplemental = FakeModel(
            [
                [
                    _detection(0, box=[11.0, 20.0, 41.0, 80.0], track_id=7),
                    _detection(1, box=[100.0, 100.0, 150.0, 150.0], track_id=8),
                ],
                [
                    _detection(0, box=[13.0, 20.0, 43.0, 80.0], track_id=7),
                    _detection(1, box=[101.0, 100.0, 151.0, 150.0], track_id=8),
                ],
            ]
        )
        samples = [
            {"time": 0.0, "frame": np.zeros((2, 2, 3))},
            {"time": 1.0, "frame": np.zeros((2, 2, 3))},
        ]
        scenes = [{"id": 1, "start": 0.0, "end": 2.0}]

        detections, tracks = track_objects(
            samples,
            primary,
            scenes,
            supplemental_model=supplemental,
        )

        self.assertEqual([len(item["detections"]) for item in detections], [2, 2])
        self.assertEqual(
            [item["detections"][0]["track_id"] for item in detections],
            [1, 1],
        )
        self.assertEqual(
            [item["detections"][1]["track_id"] for item in detections],
            [3, 3],
        )
        self.assertEqual(
            [(track["label"], track["hits"]) for track in tracks],
            [("person", 2), ("car", 2)],
        )
        self.assertEqual(primary.predictor.trackers[0].reset_count, 3)
        self.assertEqual(supplemental.predictor.trackers[0].reset_count, 3)

    def test_keeps_single_frame_tracks(self):
        model = FakeModel([[_detection(1, confidence=0.4)]])
        samples = [{"time": 0.0, "frame": np.zeros((2, 2, 3))}]

        _, tracks = track_objects(samples, model, [])

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["hits"], 1)

    def test_surfaces_detector_errors_and_resets_tracker(self):
        class BrokenModel(FakeModel):
            def track(self, frame, **kwargs):
                raise RuntimeError("detector failed")

        model = BrokenModel([])
        samples = [{"time": 0.0, "frame": np.zeros((2, 2, 3))}]

        with self.assertRaisesRegex(RuntimeError, "detector failed"):
            track_objects(samples, model, [])

        self.assertEqual(model.predictor.trackers[0].reset_count, 3)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, detections):
        self.cls = FakeTensor([item[0] for item in detections])
        self.conf = FakeTensor([item[1] for item in detections])
        self.xyxy = FakeTensor([item[2] for item in detections])
        ids = [item[3] for item in detections]
        self.id = FakeTensor(ids) if all(value is not None for value in ids) else None
        self._length = len(detections)

    def __len__(self):
        return self._length


class FakeTracker:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class FakeModel:
    names = {0: "person", 1: "car"}

    def __init__(self, results):
        self.results = iter(results)
        self.predictor = type("Predictor", (), {"trackers": [FakeTracker()]})()
        self.calls = []

    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [type("Result", (), {"boxes": FakeBoxes(next(self.results))})()]


def _detection(class_id, confidence=0.9, box=None, track_id=1):
    return (
        class_id,
        confidence,
        box or [10.0, 20.0, 40.0, 80.0],
        track_id,
    )


if __name__ == "__main__":
    unittest.main()
