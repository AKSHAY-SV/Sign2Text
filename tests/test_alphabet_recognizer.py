from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import ast

import numpy as np

from sign2text.recognizers.alphabet import AlphabetRecognizer, StableTextBuffer
from sign2text.recognizers.alphabet.adapter import read_class_names


class FakeSession:
    def __init__(self, logits=None):
        self.logits = logits if logits is not None else np.arange(26, dtype=np.float32)[None, :]
        self.inputs = [SimpleNamespace(name="input_image", shape=[None, 96, 96, 3])]
        self.outputs = [SimpleNamespace(name="logits")]
        self.calls = []

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, output_names, inputs):
        self.calls.append((output_names, inputs))
        return [self.logits]


def fake_hand():
    return SimpleNamespace(landmark=[
        SimpleNamespace(x=float(i % 5) / 5 + 0.1, y=float(i // 5) / 5 + 0.1)
        for i in range(21)
    ])


class AlphabetAdapterTests(unittest.TestCase):
    def setUp(self):
        self.repository = Path(__file__).resolve().parents[1].parent / "MediaPipe-ASL-sign-language-recognition"
        self.session = FakeSession()
        self.recognizer = AlphabetRecognizer(self.repository, session=self.session)

    def tearDown(self):
        self.recognizer.close()

    def test_checkpoint_and_authoritative_class_order_are_used(self):
        self.assertEqual(self.recognizer.class_names, tuple("abcdefghijklmnopqrstuvwxyz"))
        self.assertTrue(self.recognizer.model_path.is_file())

    def test_landmarks_render_to_model_contract_and_ranked_predictions(self):
        result = self.recognizer.process_hand(fake_hand(), 640, 480)
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.predictions), 3)
        self.assertEqual(result.best.label, "z")
        self.assertTrue(0 <= result.best.confidence <= 1)
        self.assertEqual(len(self.session.calls), 1)
        _names, inputs = self.session.calls[0]
        model_input = inputs["input_image"]
        self.assertEqual(model_input.shape, (1, 96, 96, 3))
        self.assertEqual(model_input.dtype, np.float32)
        self.assertGreater(float(model_input.max()), 1.0)  # ONNX performs /255 rescaling.

    def test_no_hand_and_invalid_landmarks_are_safe(self):
        self.assertEqual(self.recognizer.process_hand(None, 640, 480).status, "no hand detected")
        invalid = SimpleNamespace(landmark=fake_hand().landmark[:-1])
        result = self.recognizer.process_hand(invalid, 640, 480)
        self.assertEqual(result.status, "inference error")
        self.assertIn("Expected 21", result.error)

    def test_refresh_skip_keeps_tracking_but_does_not_run_inference(self):
        result = self.recognizer.process_hand(fake_hand(), 640, 480, infer=False)
        self.assertFalse(result.updated)
        self.assertEqual(self.session.calls, [])
        result = self.recognizer.process_hand(fake_hand(), 640, 480, infer=True)
        self.assertTrue(result.updated)
        self.assertEqual(len(self.session.calls), 1)

    def test_wrong_output_shape_is_reported_as_inference_error(self):
        self.session.logits = np.zeros((1, 25), dtype=np.float32)
        result = self.recognizer.process_hand(fake_hand(), 640, 480)
        self.assertEqual(result.status, "inference error")
        self.assertIn("(1,26)", result.error)

    def test_invalid_class_list_and_missing_checkpoint_are_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            classes = root / "classes.txt"
            classes.write_text("a,b,c\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verified a-z"):
                read_class_names(classes)
            classes.write_text(",".join("abcdefghijklmnopqrstuvwxyz"), encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint"):
                AlphabetRecognizer(root, classes_path=classes)

    def test_missing_onnx_runtime_is_reported_without_environment_fallback(self):
        from unittest.mock import patch

        with patch("sign2text.recognizers.alphabet.adapter.importlib.import_module",
                   side_effect=ImportError("runtime unavailable")):
            with self.assertRaisesRegex(RuntimeError, "ONNX Runtime is required"):
                AlphabetRecognizer(self.repository)

    def test_alphabet_adapter_does_not_import_or_change_gru_pipeline(self):
        source = Path(__import__("sign2text.recognizers.alphabet.adapter", fromlist=["__file__"]).__file__)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("tensorflow", imports)
        self.assertNotIn("launch_demo", imports)


class StableTextBufferTests(unittest.TestCase):
    def test_stability_confidence_repeat_and_edits(self):
        buffer = StableTextBuffer(commit_frames=3, confidence_threshold=0.75)
        self.assertFalse(buffer.update("a", 0.9))
        self.assertFalse(buffer.update("a", 0.9))
        self.assertTrue(buffer.update("a", 0.9))
        self.assertEqual(buffer.text, "A")
        for _ in range(4):
            buffer.update("a", 0.95)
        self.assertEqual(buffer.text, "A")  # no repeated character while held
        buffer.update("b", 0.6)
        buffer.update("b", 0.6)
        buffer.update("b", 0.6)
        self.assertEqual(buffer.text, "A")  # low confidence cannot commit
        buffer.update("c", 0.99, moving=True)
        self.assertEqual(buffer.text, "A")
        buffer.repeat()
        buffer.space()
        self.assertEqual(buffer.text, "AA ")
        buffer.backspace()
        self.assertEqual(buffer.text, "AA")
        buffer.clear()
        self.assertEqual(buffer.text, "")

    def test_invalid_labels_and_no_hand_do_not_commit(self):
        buffer = StableTextBuffer(commit_frames=1, confidence_threshold=0)
        self.assertFalse(buffer.update("word", 1.0))
        self.assertFalse(buffer.update(None, 1.0, hand_present=False))
        self.assertEqual(buffer.text, "")

    def test_repeated_letter_can_commit_after_release_rearms_it(self):
        buffer = StableTextBuffer(commit_frames=1, confidence_threshold=0)
        self.assertTrue(buffer.update("w", 0.9))
        self.assertFalse(buffer.update(None, 0.0, hand_present=False))
        self.assertTrue(buffer.update("w", 0.9))
        self.assertEqual(buffer.text, "WW")


if __name__ == "__main__":
    unittest.main()
