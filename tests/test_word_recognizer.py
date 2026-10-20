from pathlib import Path
from types import SimpleNamespace
import ast
import unittest

import numpy as np

from sign2text.recognizers.words import (
    StableWordText,
    WordRecognizer,
    WordReleaseGate,
    WordSequenceBuffer,
    has_detected_hand,
)


REPOSITORY = Path(__file__).resolve().parents[1].parent / "asl-recognition-lstm"
CLASSES = (
    "bath", "bye", "dad", "drink", "food", "happy", "hello", "hungry",
    "mad", "milk", "mom", "no", "please", "sad", "see", "sick", "sleep",
    "thankyou", "thirsty", "wait", "water", "where", "yes", "yourself",
)


class FakeModel:
    input_shape = (None, 40, 444)
    output_shape = (None, 24)

    def __init__(self, scores=None):
        self.scores = scores if scores is not None else np.eye(1, 24, 6, dtype=np.float32)
        self.inputs = []

    def predict(self, batch, verbose=0):
        self.inputs.append(batch.copy())
        return self.scores


class FakeSource:
    def __init__(self):
        self.model = FakeModel()
        self.actions = np.asarray(CLASSES)
        self.mean = np.zeros(444, dtype=np.float32)
        self.std = np.ones(444, dtype=np.float32)
        self.normalized_input = None
        self.extract_calls = 0

    def normalize_sequence(self, sequence):
        self.normalized_input = sequence.copy()
        return (sequence - self.mean) / self.std

    def extract_keypoints(self, _results):
        self.extract_calls += 1
        return np.arange(444, dtype=np.float32)


class WordRecognizerTests(unittest.TestCase):
    def setUp(self):
        self.source = FakeSource()
        self.recognizer = WordRecognizer(REPOSITORY, source_module=self.source)

    def test_artifacts_and_verified_label_mapping(self):
        self.assertEqual(self.recognizer.num_classes, 24)
        self.assertEqual(self.recognizer.class_names[6], "hello")
        self.assertNotIn("world", self.recognizer.class_names)
        for relative in ("model/best.keras", "model/actions.npy", "model/mean.npy", "model/std.npy"):
            self.assertTrue((REPOSITORY / relative).is_file())

    def test_extract_frame_delegates_to_source_order_and_validates_shape(self):
        results = SimpleNamespace(left_hand_landmarks=[object()], right_hand_landmarks=None)
        features = self.recognizer.extract_frame(results)
        self.assertEqual(features.shape, (444,))
        self.assertTrue(np.array_equal(features, np.arange(444, dtype=np.float32)))
        self.assertEqual(self.source.extract_calls, 1)

    def test_normalization_batch_shape_and_label_mapping(self):
        scores = np.zeros((1, 24), dtype=np.float32)
        scores[0, 6] = 0.82
        scores[0, 0] = 0.10
        scores[0, 23] = 0.08
        self.source.model.scores = scores
        sequence = np.arange(40 * 444, dtype=np.float32).reshape(40, 444)

        result = self.recognizer.predict_sequence(sequence)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.best.label, "hello")
        self.assertEqual(result.best.index, 6)
        self.assertAlmostEqual(result.best.confidence, 0.82)
        self.assertEqual([item.label for item in result.predictions], ["hello", "bath", "yourself"])
        self.assertTrue(np.array_equal(self.source.normalized_input, sequence))
        model_input = self.source.model.inputs[0]
        self.assertEqual(model_input.shape, (1, 40, 444))
        self.assertEqual(model_input.dtype, np.float32)

    def test_incomplete_wrong_shape_nonfinite_and_bad_model_output(self):
        self.assertEqual(self.recognizer.predict_sequence(np.zeros((39, 444))).status,
                         "insufficient frames")
        self.assertEqual(self.recognizer.predict_sequence(np.zeros((40, 443))).status,
                         "invalid input")
        invalid = np.zeros((40, 444), dtype=np.float32)
        invalid[0, 0] = np.nan
        self.assertEqual(self.recognizer.predict_sequence(invalid).status, "invalid input")
        self.source.model.scores = np.zeros((1, 23), dtype=np.float32)
        failed = self.recognizer.predict_sequence(np.zeros((40, 444), dtype=np.float32))
        self.assertEqual(failed.status, "inference error")

    def test_missing_artifacts_are_reported(self):
        with self.assertRaisesRegex(FileNotFoundError, "artifact"):
            WordRecognizer(Path("/tmp/missing-word-model"), source_module=self.source)

    def test_missing_hand_detection_is_distinguished(self):
        self.assertFalse(has_detected_hand(SimpleNamespace(left_hand_landmarks=None,
                                                           right_hand_landmarks=None)))
        self.assertTrue(has_detected_hand(SimpleNamespace(left_hand_landmarks=[object()],
                                                          right_hand_landmarks=None)))

    def test_word_adapter_is_separate_from_alphabet_and_gru(self):
        source = Path(__import__("sign2text.recognizers.words.adapter", fromlist=["__file__"]).__file__)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("tensorflow", imports)
        self.assertNotIn("launch_demo", imports)
        self.assertNotIn("alphabet", imports)


class WordBufferTests(unittest.TestCase):
    def test_release_gate_suppresses_held_and_repeated_labels_until_reset(self):
        gate = WordReleaseGate()
        self.assertTrue(gate.accept("hello"))
        self.assertFalse(gate.accept("hello"))
        self.assertTrue(gate.accept("bye"))
        self.assertFalse(gate.accept("hello"))
        gate.reset()  # integrated worker resets this only when no hand is detected
        self.assertTrue(gate.accept("hello"))

    def test_sequence_waits_for_exact_40_and_slides(self):
        buffer = WordSequenceBuffer(sequence_length=40, feature_size=444)
        frame = np.ones(444, dtype=np.float32)
        for _ in range(39):
            buffer.append(frame)
        self.assertFalse(buffer.ready)
        self.assertIsNone(buffer.array())
        buffer.append(frame * 2)
        self.assertTrue(buffer.ready)
        self.assertEqual(buffer.array().shape, (40, 444))
        buffer.append(frame * 3)
        self.assertEqual(buffer.array().shape, (40, 444))
        self.assertTrue(np.all(buffer.array()[-1] == 3))
        buffer.clear()
        self.assertEqual(buffer.count, 0)

    def test_hello_stability_low_confidence_clear_and_no_world(self):
        buffer = StableWordText(CLASSES, stable_updates=3, confidence_threshold=0.60)
        self.assertFalse(buffer.update("hello", 0.9))
        self.assertFalse(buffer.update("hello", 0.9))
        self.assertTrue(buffer.update("hello", 0.9))
        self.assertEqual(buffer.text, "HELLO")
        for _ in range(4):
            buffer.update("hello", 0.99)
        self.assertEqual(buffer.text, "HELLO")
        buffer.update("bye", 0.5)
        buffer.update("bye", 0.9)
        buffer.update("bye", 0.9)
        self.assertEqual(buffer.text, "HELLO")
        self.assertFalse(buffer.update("world", 0.99))

    def test_releasing_hand_then_clearing_stability_rearms_same_word(self):
        buffer = StableWordText(CLASSES, stable_updates=2, confidence_threshold=0.60)
        self.assertFalse(buffer.update("hello", 0.9))
        self.assertTrue(buffer.update("hello", 0.9))
        self.assertFalse(buffer.update("hello", 0.99))
        buffer.clear()  # Integrated worker calls this after no-hand detection.
        self.assertFalse(buffer.update("hello", 0.9))
        self.assertTrue(buffer.update("hello", 0.9))
        self.assertEqual(buffer.text, "HELLO")
        buffer.clear()
        self.assertEqual(buffer.text, "")


if __name__ == "__main__":
    unittest.main()
