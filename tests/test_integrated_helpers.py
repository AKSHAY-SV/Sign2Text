import io
import ast
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import numpy as np

from sign2text.integrated import (
    TextAccumulator, apply_text_key, apply_worker_commit, encode_frame, enqueue_latest,
    read_frame, presentation_fields, switch_input_mode, valid_worker_result,
)
from sign2text.recognizers.alphabet import StableTextBuffer
from sign2text.recognizers.words import StableWordText, WordReleaseGate


class IntegratedProtocolTests(unittest.TestCase):
    def test_frame_round_trip_preserves_bgr_values_and_sequence(self):
        frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        header, payload = encode_frame(17, frame)
        received = read_frame(io.BytesIO(header + payload))
        self.assertEqual(received.sequence, 17)
        self.assertTrue(received.accept_input)
        np.testing.assert_array_equal(received.frame, frame)

    def test_clean_eof_and_truncated_message(self):
        self.assertIsNone(read_frame(io.BytesIO(b"")))
        header, payload = encode_frame(1, np.zeros((2, 3, 3), dtype=np.uint8))
        with self.assertRaises(EOFError):
            read_frame(io.BytesIO(header + payload[:-1]))

    def test_rejects_invalid_frame_dtype_and_shape(self):
        with self.assertRaises(ValueError):
            encode_frame(0, np.zeros((2, 2, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            encode_frame(0, np.zeros((2, 2), dtype=np.uint8))

    def test_rejects_mismatched_protocol_payload_size(self):
        header, payload = encode_frame(0, np.zeros((2, 3, 3), dtype=np.uint8))
        corrupted = bytearray(header)
        corrupted[-5] -= 1
        with self.assertRaises(ValueError):
            read_frame(io.BytesIO(bytes(corrupted) + payload))

    def test_mode_flag_round_trip(self):
        header, payload = encode_frame(3, np.zeros((1, 2, 3), dtype=np.uint8),
                                       accept_input=False)
        self.assertFalse(read_frame(io.BytesIO(header + payload)).accept_input)

    def test_latest_queue_discards_stale_item(self):
        import queue

        target = queue.Queue(maxsize=1)
        self.assertFalse(enqueue_latest(target, "old"))
        self.assertTrue(enqueue_latest(target, "new"))
        self.assertEqual(target.get_nowait(), "new")


class TextAccumulatorTests(unittest.TestCase):
    def test_word_letter_separator_clear_and_backspace(self):
        text = TextAccumulator()
        text.append_letter("a")
        text.append_letter("s")
        text.space()
        text.append_word("hello")
        self.assertEqual(text.text, "AS HELLO")
        text.backspace()
        self.assertEqual(text.text, "AS HELL")
        text.clear()
        self.assertEqual(text.text, "")

    def test_mode_gating_shared_buffer_and_hello_world_workflow(self):
        text = TextAccumulator()
        accepted_hello = {
            "type": "result", "mode": "word", "sequence": 1,
            "status": "accepted", "prediction": {"label": "hello", "confidence": .97},
            "top": [{"label": "hello", "confidence": .97}], "progress": 1.0,
            "committed": "hello", "input_enabled": True,
        }
        self.assertTrue(valid_worker_result(accepted_hello, "word", {"hello"}))
        self.assertTrue(apply_worker_commit(text, accepted_hello, selected_mode="word"))
        self.assertEqual(text.text, "HELLO")
        # Later display updates are not text edits and cannot clear the commit.
        for status in ("stabilizing", "release to repeat"):
            update = {**accepted_hello, "sequence": 2, "status": status, "committed": None}
            self.assertFalse(apply_worker_commit(text, update, selected_mode="word"))
            # OpenCV waitKey returns -1 while idle; this must never be mistaken
            # for Backspace after applying the 0xFF key mask.
            self.assertEqual((-1) & 0xFF, 255)
            self.assertFalse(apply_text_key(text, -1))
            self.assertEqual(text.text, "HELLO")
        # A background alphabet prediction cannot write while WORD is selected.
        self.assertFalse(apply_worker_commit(text, {
            "mode": "alphabet", "sequence": 2, "input_enabled": False, "committed": "w",
        }, selected_mode="word"))
        self.assertEqual(text.text, "HELLO")

        mode = switch_input_mode(text, "word", "alphabet")
        self.assertEqual(mode, "alphabet")
        for letter in "world":
            self.assertTrue(apply_worker_commit(text, {
                "mode": "alphabet", "sequence": 3, "input_enabled": True, "committed": letter,
            }, selected_mode="alphabet"))
        # A background word prediction cannot write while SPELL is selected.
        self.assertFalse(apply_worker_commit(text, {
            "mode": "word", "sequence": 4, "input_enabled": False, "committed": "hello",
        }, selected_mode="alphabet"))
        self.assertEqual(text.text, "HELLO WORLD")

        # An in-flight old-mode result cannot commit after mode switch.
        delayed = TextAccumulator()
        switch_input_mode(delayed, "word", "alphabet")
        self.assertFalse(apply_worker_commit(delayed, {
            "mode": "word", "sequence": 4, "input_enabled": True, "committed": "hello",
        }, selected_mode="alphabet"))
        self.assertTrue(apply_worker_commit(delayed, {
            "mode": "alphabet", "sequence": 5, "input_enabled": True, "committed": "w",
        }, selected_mode="alphabet"))
        self.assertEqual(delayed.text, "W")

    def test_old_frame_result_is_discarded_after_reselecting_mode(self):
        text = TextAccumulator()
        self.assertFalse(apply_worker_commit(text, {
            "mode": "word", "sequence": 9, "input_enabled": True, "committed": "hello",
        }, selected_mode="word", minimum_sequence=12))
        self.assertEqual(text.text, "")

    def test_word_and_letter_commits_require_stability_and_release(self):
        word = StableWordText({"hello"}, stable_updates=2, confidence_threshold=0.60)
        word_release = WordReleaseGate()
        text = TextAccumulator()
        accepted_events = []
        for _ in range(5):
            if word.update("hello", 0.97) and word_release.accept("hello"):
                accepted_events.append("hello")
        self.assertEqual(accepted_events, ["hello"])
        self.assertTrue(apply_worker_commit(text, {
            "mode": "word", "sequence": 1, "status": "accepted",
            "prediction": {"label": "hello", "confidence": .97},
            "input_enabled": True, "committed": "hello",
        }, selected_mode="word"))
        for _ in range(5):
            if word.update("hello", 0.97) and word_release.accept("hello"):
                accepted_events.append("hello")
        self.assertEqual(accepted_events, ["hello"])
        self.assertEqual(text.text, "HELLO")
        word.clear()
        word_release.reset()
        self.assertFalse(word.update("hello", 0.97))
        self.assertTrue(word.update("hello", 0.97))
        self.assertTrue(word_release.accept("hello"))
        self.assertTrue(apply_worker_commit(text, {
            "mode": "word", "sequence": 2, "status": "accepted",
            "prediction": {"label": "hello", "confidence": .97},
            "input_enabled": True, "committed": "hello",
        }, selected_mode="word"))
        self.assertEqual(text.text, "HELLO HELLO")
        switch_input_mode(text, "word", "alphabet")
        self.assertEqual(text.text, "HELLO HELLO ")

        letters = StableTextBuffer(commit_frames=2, confidence_threshold=0.75)
        self.assertFalse(letters.update("w", .89))
        self.assertTrue(letters.update("w", .89))
        self.assertFalse(letters.update("w", .89))
        self.assertFalse(letters.update(None, 0, hand_present=False))
        self.assertFalse(letters.update("w", .89))
        self.assertTrue(letters.update("w", .89))
        for letter in "orld":
            self.assertFalse(letters.update(None, 0, hand_present=False))
            self.assertFalse(letters.update(letter, .90))
            self.assertTrue(letters.update(letter, .90))
        for sequence, letter in enumerate("world", start=2):
            self.assertTrue(apply_worker_commit(text, {
                "mode": "alphabet", "sequence": sequence, "input_enabled": True,
                "committed": letter,
            }, selected_mode="alphabet"))
        self.assertEqual(text.text, "HELLO HELLO WORLD")

    def test_accepted_letter_and_later_prediction_updates_persist(self):
        text = TextAccumulator()
        accepted_w = {
            "mode": "alphabet", "sequence": 14, "status": "accepted",
            "prediction": {"label": "w", "confidence": .89},
            "committed": "w", "input_enabled": True,
        }
        self.assertTrue(apply_worker_commit(text, accepted_w, selected_mode="alphabet"))
        subsequent = {**accepted_w, "sequence": 15, "status": "stabilizing", "committed": None}
        self.assertFalse(apply_worker_commit(text, subsequent, selected_mode="alphabet"))
        self.assertEqual(text.text, "W")

    def test_duplicate_spaces_clear_backspace_and_mode_switch(self):
        text = TextAccumulator()
        text.append_word("hello")
        switch_input_mode(text, "word", "alphabet")
        text.space()
        self.assertEqual(text.text, "HELLO ")
        text.append_letter("w")
        self.assertEqual(text.text, "HELLO W")
        text.backspace()
        self.assertEqual(text.text, "HELLO ")
        text.clear()
        self.assertEqual(text.text, "")

    def test_text_keys_only_edit_when_a_real_key_was_pressed(self):
        text = TextAccumulator()
        text.append_word("hello")
        self.assertFalse(apply_text_key(text, -1))
        self.assertEqual(text.text, "HELLO")
        self.assertTrue(apply_text_key(text, 8))
        self.assertEqual(text.text, "HELL")
        self.assertTrue(apply_text_key(text, ord("c")))
        self.assertEqual(text.text, "")

    def test_presentation_reads_shared_buffer_and_hides_diagnostics(self):
        text = TextAccumulator()
        text.append_word("hello")
        latest = {
            "word": {"sequence": 1, "prediction": {"label": "hello", "confidence": .97},
                     "status": "stabilizing", "progress": .5,
                     "top": [{"label": "hello", "confidence": .97}],
                     "worker_id": "private"},
            "alphabet": {"sequence": 1, "prediction": {"label": "o", "confidence": .89},
                         "status": "ok", "progress": .3,
                         "top": [{"label": "o", "confidence": .89}]},
        }
        fields = presentation_fields("word", latest,
                                     {"word": "ready", "alphabet": "ready"}, text)
        self.assertEqual(fields, (
            ("Input mode", "WORD"), ("Word", "HELLO 97.0%"),
            ("Letter", "O 89.0%"), ("Status", "Stabilizing word"),
            ("Text", "HELLO"),
        ))
        self.assertEqual(fields[-1], ("Text", text.text))
        rendered = " | ".join(f"{name}: {value}" for name, value in fields)
        self.assertNotIn("Workers:", rendered)
        self.assertNotIn("Stable:", rendered)
        self.assertNotIn("Top:", rendered)
        self.assertNotIn("worker_id", rendered)

    def test_invalid_or_missing_worker_results_are_ignored(self):
        text = TextAccumulator()
        self.assertFalse(apply_worker_commit(text, None, selected_mode="word"))
        self.assertFalse(apply_worker_commit(text, {
            "mode": "word", "sequence": 1, "committed": "hello",
        }, selected_mode="word"))
        self.assertFalse(apply_worker_commit(text, {
            "mode": "alphabet", "sequence": 1, "input_enabled": True, "committed": "letters",
        }, selected_mode="alphabet"))
        self.assertEqual(text.text, "")

    def test_worker_result_validation_rejects_wrong_mode_and_invalid_scores(self):
        valid = {"type": "result", "mode": "word", "sequence": 2,
                 "status": "prediction", "prediction": {"label": "hello", "confidence": 0.9},
                 "top": [{"label": "hello", "confidence": 0.9}], "progress": 1.0,
                 "committed": None, "input_enabled": True}
        self.assertTrue(valid_worker_result(valid, "word", {"hello"}))
        self.assertFalse(valid_worker_result(valid, "alphabet"))
        invalid = {**valid, "prediction": {"label": "hello", "confidence": float("nan")}}
        self.assertFalse(valid_worker_result(invalid, "word", {"hello"}))
        self.assertFalse(valid_worker_result({**valid, "top": "not-a-list"}, "word", {"hello"}))
        unsupported = {**valid, "committed": "world"}
        self.assertFalse(valid_worker_result(unsupported, "word", {"hello"}))


class IntegratedGuardTests(unittest.TestCase):
    def test_integrated_worker_installs_guard_before_task_factory(self):
        from sign2text.recognizers.words.landmarker import create_integrated_word_landmarker

        order = []
        task = object()
        actual = create_integrated_word_landmarker(
            "source",
            install_guard=lambda: order.append("guard"),
            factory=lambda source: order.append(("factory", source)) or task,
            mediapipe_version="test-version",
        )
        self.assertIs(actual, task)
        self.assertEqual(order, ["guard", ("factory", "source")])

    def test_guard_failure_stops_task_initialization_with_diagnostic(self):
        from sign2text.recognizers.words.landmarker import create_integrated_word_landmarker

        called = []
        with self.assertRaisesRegex(RuntimeError, "empty-packet guard"):
            create_integrated_word_landmarker(
                object(),
                install_guard=lambda: (_ for _ in ()).throw(ValueError("unsupported API")),
                factory=lambda source: called.append(source),
                mediapipe_version="test-version",
            )
        self.assertEqual(called, [])

    def test_integrated_ui_opens_camera_once_workers_never_open_it(self):
        root = Path(__file__).resolve().parents[1]
        ui = ast.parse((root / "tools/run_sign2text_integrated.py").read_text(encoding="utf-8"))
        worker = ast.parse((root / "tools/sign2text_inference_worker.py").read_text(encoding="utf-8"))
        camera_calls = [node for node in ast.walk(ui)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "VideoCapture"]
        worker_camera_calls = [node for node in ast.walk(worker)
                               if isinstance(node, ast.Call)
                               and isinstance(node.func, ast.Attribute)
                               and node.func.attr == "VideoCapture"]
        self.assertEqual(len(camera_calls), 1)
        self.assertEqual(worker_camera_calls, [])
        worker_constructions = [node for node in ast.walk(ui)
                                if isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Name)
                                and node.func.id == "InferenceWorker"]
        self.assertEqual(len(worker_constructions), 2)
        worker_calls = [node for node in ast.walk(worker)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "create_integrated_word_landmarker"]
        self.assertEqual(len(worker_calls), 1)

    def test_worker_shutdown_joins_threads_and_waits_for_child(self):
        from tools.run_sign2text_integrated import InferenceWorker

        class FakePipe(io.BytesIO):
            pass

        class BlockingStdout:
            def __init__(self):
                self.closed = threading.Event()

            def __iter__(self):
                yield b'{"type":"ready","mode":"word","classes":["hello"]}\n'
                self.closed.wait(timeout=2)

            def close(self):
                self.closed.set()

        class FakeProcess:
            def __init__(self):
                self.stdin = FakePipe()
                self.stdout = BlockingStdout()
                self.returncode = None
                self.waited = False

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.waited = True
                self.returncode = 0
                self.stdout.close()
                return 0

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        process = FakeProcess()
        with patch("tools.run_sign2text_integrated.subprocess.Popen", return_value=process):
            worker = InferenceWorker("word", Path("/model"))
        worker.close()
        self.assertTrue(process.waited)
        self.assertFalse(worker._reader.is_alive())
        self.assertFalse(worker._writer.is_alive())

    def test_unexpected_worker_eof_is_reported(self):
        from tools.run_sign2text_integrated import InferenceWorker

        class FakeProcess:
            def __init__(self):
                self.stdin = io.BytesIO()
                self.stdout = io.BytesIO(b'{"type":"ready","mode":"alphabet","classes":["a"]}\n')
                self.returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = 1
                return 1

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        process = FakeProcess()
        with patch("tools.run_sign2text_integrated.subprocess.Popen", return_value=process):
            worker = InferenceWorker("alphabet", Path("/model"))
        worker._reader.join(timeout=1)
        self.assertIn("unexpectedly", worker.fatal)
        worker.close()

    def test_letter_requires_single_character(self):
        text = TextAccumulator()
        with self.assertRaises(ValueError):
            text.append_letter("hello")


if __name__ == "__main__":
    unittest.main()
