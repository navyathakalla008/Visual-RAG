"""Focused tests for AudioSet sound-event inference and event timing."""
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

import audio_events


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self):
        self.window_lengths = []

    def __call__(self, audio, sampling_rate, return_tensors):
        self.window_lengths.append(len(audio))
        return FakeInputs()


class FakeAudioModel:
    config = SimpleNamespace(id2label={0: "Speech", 1: "Music", 2: "Silence"})

    def __init__(self, logits):
        self.logits = iter(logits)

    def __call__(self, **inputs):
        return SimpleNamespace(logits=next(self.logits))


class AudioEventTests(unittest.TestCase):
    def test_audio_is_scaled_and_resampled_for_model(self):
        source = np.array([0, 16384, -32768], dtype=np.int16)
        converted = audio_events._audio_as_float(source, 8000)
        self.assertEqual(converted.dtype, np.float32)
        self.assertEqual(converted.size, 6)
        self.assertAlmostEqual(float(converted.min()), -1.0)
        self.assertTrue(np.isfinite(converted).all())

    def test_model_predictions_merge_repeated_windows_and_keep_timestamps(self):
        processor = FakeProcessor()
        model = FakeAudioModel([
            torch.tensor([[5.0, 4.0, -5.0]]),
            torch.tensor([[5.0, 4.0, -5.0]]),
            torch.tensor([[5.0, -5.0, -5.0]]),
        ])
        audio = np.zeros(int(7.5 * audio_events.SAMPLE_RATE), dtype=np.float32)
        with mock.patch.dict("os.environ", {"AUDIO_EVENT_MIN_CONFIDENCE": "0.25"}):
            events = audio_events._classify_audio(
                audio, processor, model, torch.device("cpu"), "test-model"
            )

        self.assertEqual(processor.window_lengths, [
            5 * audio_events.SAMPLE_RATE,
            5 * audio_events.SAMPLE_RATE,
            int(2.5 * audio_events.SAMPLE_RATE),
        ])
        by_label = {event["label"]: event for event in events}
        self.assertEqual(set(by_label), {"Speech", "Music"})
        self.assertEqual((by_label["Speech"]["start"], by_label["Speech"]["end"]), (0.0, 7.5))
        self.assertEqual((by_label["Music"]["start"], by_label["Music"]["end"]), (0.0, 7.5))
        self.assertGreater(by_label["Speech"]["confidence"], 0.9)
        self.assertEqual(by_label["Speech"]["evidence"], "AudioSet AST multi-label model")

    def test_silence_is_not_reported_as_a_sound_event(self):
        events = audio_events._classify_audio(
            np.zeros(audio_events.SAMPLE_RATE, dtype=np.float32),
            FakeProcessor(),
            FakeAudioModel([torch.tensor([[-5.0, -5.0, 5.0]])]),
            torch.device("cpu"),
            "test-model",
        )
        self.assertEqual(events, [])

    def test_stereo_downmix_avoids_phase_cancellation(self):
        source = np.tile(
            np.array([[12000, -12000]], dtype=np.int16),
            (audio_events.SAMPLE_RATE, 1),
        )

        converted = audio_events._audio_as_float(source, audio_events.SAMPLE_RATE)

        self.assertGreater(float(np.sqrt(np.mean(converted ** 2))), 0.3)

    def test_extraction_error_is_returned_to_caller(self):
        with mock.patch.object(
            audio_events, "_extract_audio", return_value=(False, "No audio track was found.")
        ):
            events, error = audio_events.detect_audio_events("video.mp4")
        self.assertEqual(events, [])
        self.assertEqual(error, "No audio track was found.")


if __name__ == "__main__":
    unittest.main()
