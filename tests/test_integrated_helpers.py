import io
import ast
from pathlib import Path
import threading
import queue
import unittest
from unittest.mock import patch

import numpy as np

from sign2text.integrated import (
    ConfirmationController, PendingCandidate, TextAccumulator, apply_candidate_action,
    apply_candidate_key, apply_text_key, apply_worker_commit, candidate_from_result, encode_frame,
    candidate_outside_cooldown, enqueue_latest, read_frame, presentation_fields, switch_input_mode,
    update_pending_candidate, valid_worker_result,
)
from sign2text.recognizers.alphabet import StableTextBuffer
from sign2text.recognizers.words import StableWordText, WordReleaseGate


class IntegratedProtocolTests(unittest.TestCase):
    def test_integrated_entry_point_accepts_tft_options(self):
        from tools.run_sign2text_integrated import main

        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as error:
                main(["--unsupported-serial-port", "unused"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("unrecognized arguments", stderr.getvalue())
        self.assertEqual(stderr.getvalue().count("--tft-port"), 1)
        self.assertEqual(stderr.getvalue().count("--no-tft"), 1)

    def test_integrated_cli_defaults_to_esp32_and_can_disable_tft(self):
        from tools.run_sign2text_integrated import main

        with patch("tools.run_sign2text_integrated.run", return_value=0) as run:
            self.assertEqual(main([]), 0)
            self.assertEqual(run.call_args.kwargs["arduino_port"], "/dev/ttyUSB0")
            self.assertEqual(run.call_args.kwargs["arduino_baud"], 115200)
            self.assertEqual(main(["--no-tft"]), 0)
            self.assertIsNone(run.call_args.kwargs["arduino_port"])

    def test_camera_frame_fits_window_with_letterboxing_and_aspect_ratio(self):
        from tools.run_sign2text_integrated import fit_camera_frame

        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:, :, 1] = 200
        fitted = fit_camera_frame(frame, 300, 300)
        self.assertEqual(fitted.shape, (300, 300, 3))
        np.testing.assert_array_equal(fitted[:75], 0)
        np.testing.assert_array_equal(fitted[225:], 0)
        self.assertTrue(np.all(fitted[75:225, :, 1] == 200))

    def test_camera_frame_fills_wide_window_without_distortion(self):
        from tools.run_sign2text_integrated import fit_camera_frame

        frame = np.ones((100, 200, 3), dtype=np.uint8)
        fitted = fit_camera_frame(frame, 400, 200)
        self.assertEqual(fitted.shape, (200, 400, 3))
        self.assertTrue(np.all(fitted == 1))

    def test_window_dimensions_use_camera_size_for_invalid_or_tiny_query(self):
        from tools.run_sign2text_integrated import _render_dimensions

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("cv2.getWindowImageRect", return_value=(0, 0, 1, 1)):
            self.assertEqual(_render_dimensions(frame, "test", (640, 480)), (640, 480))
        with patch("cv2.getWindowImageRect", side_effect=Exception("not mapped")):
            self.assertEqual(_render_dimensions(frame, "test", (0, 0)), (640, 480))

    def test_window_dimensions_accept_plausible_client_area(self):
        from tools.run_sign2text_integrated import _render_dimensions

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("cv2.getWindowImageRect", return_value=(10, 20, 1200, 800)):
            self.assertEqual(_render_dimensions(frame, "test", (640, 480)), (1200, 800))

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
    def test_predictions_stay_pending_until_confirmed(self):
        text = TextAccumulator()
        record = {"mode": "word", "sequence": 2, "input_enabled": True,
                  "prediction": {"label": "hello", "confidence": .97},
                  "signing_hand": "Right", "signing_hand_verified": True}
        candidate = candidate_from_result(record, selected_mode="word")
        self.assertEqual(candidate.label, "hello")
        self.assertEqual(text.text, "")
        self.assertTrue(apply_candidate_action(text, candidate, "confirm"))
        self.assertEqual(text.text, "HELLO")

    def test_reject_preserves_committed_text(self):
        text = TextAccumulator()
        text.append_word("hello")
        before = text.text
        self.assertTrue(apply_candidate_action(
            text, PendingCandidate("alphabet", "w", .9, 3), "reject"))
        self.assertEqual(text.text, before)

    def test_keyboard_candidate_confirmation_and_rejection_fallback(self):
        text = TextAccumulator()
        candidate = PendingCandidate("word", "hello", .9, 1)
        self.assertEqual(apply_candidate_key(text, candidate, ord("+")), "confirm")
        self.assertEqual(text.text, "HELLO")
        self.assertEqual(apply_candidate_key(
            text, PendingCandidate("word", "thanks", .8, 2), ord("r")), "reject")
        self.assertEqual(text.text, "HELLO")

    def test_open_palm_hold_confirms_once_and_requires_release(self):
        candidate = PendingCandidate("word", "hello", .9, 1)
        controller = ConfirmationController(.3)
        text = TextAccumulator()
        self.assertIsNone(controller.update("open_palm", 0, handedness_reliable=True,
                                            candidate=candidate))
        action = controller.update("open_palm", .31, handedness_reliable=True,
                                   candidate=candidate)
        self.assertEqual(action, "confirm")
        self.assertTrue(apply_candidate_action(text, candidate, action))
        self.assertEqual(text.text, "HELLO")
        second = PendingCandidate("word", "thanks", .9, 2)
        self.assertIsNone(controller.update("open_palm", 1.3, handedness_reliable=True,
                                            candidate=second))
        # One held palm cannot confirm the next candidate.
        controller.begin_cooldown(.61)
        self.assertAlmostEqual(controller.cooldown_remaining(.61), 3.0)
        self.assertEqual(controller.state, "cooldown")
        self.assertEqual(text.text, "HELLO")

    def test_cooldown_blocks_new_candidates_and_late_results(self):
        controller = ConfirmationController(.3)
        candidate = PendingCandidate("word", "hello", .9, 1)
        controller.update("open_palm", 0, handedness_reliable=True, candidate=candidate)
        self.assertEqual(controller.update("open_palm", .31, handedness_reliable=True,
                                           candidate=candidate), "confirm")
        controller.begin_cooldown(.31)
        during = PendingCandidate("word", "thanks", .9, 20)
        self.assertIsNone(candidate_outside_cooldown(
            during, controller=controller, now=2.0, blocked_through_sequence=-1))
        self.assertEqual(controller.cooldown_remaining(3.5), 0.0)
        self.assertIsNone(candidate_outside_cooldown(
            during, controller=controller, now=3.5, blocked_through_sequence=25))
        fresh = PendingCandidate("word", "thanks", .9, 26)
        self.assertIs(candidate_outside_cooldown(
            fresh, controller=controller, now=3.5, blocked_through_sequence=25), fresh)

    def test_neutral_release_rearms_open_palm_after_cooldown(self):
        candidate = PendingCandidate("word", "hello", .9, 1)
        next_candidate = PendingCandidate("word", "thanks", .9, 40)
        controller = ConfirmationController(.3)
        controller.update("open_palm", 0, handedness_reliable=True, candidate=candidate)
        self.assertEqual(controller.update("open_palm", .31, handedness_reliable=True,
                                           candidate=candidate), "confirm")
        # Open palm by itself cannot release the post-confirmation lock.
        controller.begin_cooldown(.31)
        self.assertIsNone(controller.update("open_palm", 3.5, handedness_reliable=True,
                                            candidate=next_candidate, observation_sequence=1))
        self.assertEqual(controller.state, "waiting for release")
        self.assertIsNone(controller.update("neutral", 3.6, handedness_reliable=True,
                                            candidate=next_candidate, observation_sequence=2))
        self.assertEqual(controller.state, "ready")
        self.assertIsNone(controller.update("open_palm", 3.7, handedness_reliable=True,
                                            candidate=next_candidate, observation_sequence=3))
        self.assertEqual(controller.state, "holding open palm")
        self.assertIsNone(controller.update("open_palm", 3.8, handedness_reliable=True,
                                            candidate=next_candidate, observation_sequence=4))
        self.assertEqual(controller.update("open_palm", 4.02, handedness_reliable=True,
                                           candidate=next_candidate, observation_sequence=5),
                         "confirm")

    def test_open_palm_never_edits_text_or_confirms(self):
        text = TextAccumulator()
        text.append_word("kept")
        before = text.text
        controller = ConfirmationController(.3)
        candidate = PendingCandidate("word", "new", .9, 1)
        self.assertIsNone(controller.update("open_palm", 0, handedness_reliable=True,
                                            candidate=candidate))
        self.assertEqual(controller.update("open_palm", .31, handedness_reliable=True,
                                           candidate=candidate), "confirm")
        self.assertFalse(apply_candidate_action(text, candidate, "open_palm"))
        self.assertEqual(text.text, before)
        no_candidate = ConfirmationController(.3)
        self.assertIsNone(no_candidate.update("open_palm", 0, handedness_reliable=True,
                                              candidate=None))
        self.assertIsNone(no_candidate.update("open_palm", .31, handedness_reliable=True,
                                              candidate=None))
        self.assertEqual(text.text, before)

    def test_unreliable_handedness_and_no_candidate_never_act(self):
        controller = ConfirmationController(.5)
        candidate = PendingCandidate("word", "hello", .9, 1)
        self.assertIsNone(controller.update("open_palm", 0, handedness_reliable=False,
                                            candidate=candidate))
        self.assertIsNone(controller.update("open_palm", 1, handedness_reliable=True,
                                            candidate=None))

    def test_candidate_change_restarts_hold_and_right_hand_is_not_controller(self):
        controller = ConfirmationController(.5)
        first = PendingCandidate("word", "hello", .9, 1)
        second = PendingCandidate("word", "thanks", .9, 2)
        self.assertIsNone(controller.update("open_palm", 0, handedness_reliable=True,
                                            candidate=first))
        self.assertIsNone(controller.update("open_palm", .4, handedness_reliable=True,
                                            candidate=second))
        self.assertEqual(controller.update("open_palm", .91, handedness_reliable=True,
                                           candidate=second), "confirm")
        controller.reset()
        self.assertIsNone(controller.update("open_palm", 2, handedness_reliable=False,
                                            candidate=second))

    def test_cached_worker_observation_cannot_satisfy_stable_hold(self):
        controller = ConfirmationController(.5)
        candidate = PendingCandidate("word", "hello", .9, 1)
        self.assertIsNone(controller.update("open_palm", 0, handedness_reliable=True,
                                            candidate=candidate, observation_sequence=10))
        self.assertIsNone(controller.update("open_palm", 1, handedness_reliable=True,
                                            candidate=candidate, observation_sequence=10))
        self.assertEqual(controller.state, "holding open palm")
        self.assertIsNone(controller.update("open_palm", 1.1, handedness_reliable=True,
                                            candidate=candidate, observation_sequence=11))
        self.assertEqual(controller.state, "holding open palm")

    def test_stale_or_wrong_mode_worker_prediction_is_not_pending(self):
        record = {"mode": "word", "sequence": 4, "input_enabled": True,
                  "prediction": {"label": "hello", "confidence": .9},
                  "signing_hand": "Right", "signing_hand_verified": True}
        self.assertIsNone(candidate_from_result(record, selected_mode="alphabet"))
        self.assertIsNone(candidate_from_result(record, selected_mode="word",
                                                minimum_sequence=5))
        self.assertIsNone(candidate_from_result({**record, "input_enabled": False},
                                                selected_mode="word"))

    def test_pending_survives_neutral_result_and_stale_candidate_cannot_replace_it(self):
        current = PendingCandidate("word", "hello", .9, 12)
        still_pending, changed = update_pending_candidate(current, None)
        self.assertIs(still_pending, current)
        self.assertFalse(changed)
        stale = PendingCandidate("word", "thanks", .9, 11)
        still_pending, changed = update_pending_candidate(current, stale)
        self.assertIs(still_pending, current)
        self.assertFalse(changed)
        fresh = PendingCandidate("word", "thanks", .92, 13)
        replaced, changed = update_pending_candidate(current, fresh)
        self.assertEqual(replaced, fresh)
        self.assertTrue(changed)

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
            "signing_hand": "Right", "signing_hand_verified": True,
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
            "signing_hand": "Right", "signing_hand_verified": True,
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
            "signing_hand": "Right", "signing_hand_verified": True,
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
            ("Letter", "O 89.0%"), ("Candidate", "(none)"),
            ("Status", "Stabilizing word"),
            ("Text", "HELLO"),
        ))
        self.assertEqual(fields[-1], ("Text", text.text))
        rendered = " | ".join(f"{name}: {value}" for name, value in fields)
        self.assertNotIn("Workers:", rendered)
        self.assertNotIn("Stable:", rendered)
        self.assertNotIn("Top:", rendered)
        self.assertNotIn("worker_id", rendered)

    def test_presentation_marks_pending_candidate_and_keeps_text_separate(self):
        text = TextAccumulator()
        text.append_word("hello")
        fields = presentation_fields(
            "word", {}, {"word": "ready", "alphabet": "ready"}, text,
            candidate=PendingCandidate("word", "thanks", .91, 3),
        )
        self.assertIn(("Candidate", "WORD THANKS 91.0% — Pending"), fields)
        self.assertEqual(fields[-1], ("Text", "HELLO"))

    def test_controller_identity_requires_confident_left_classification(self):
        from types import SimpleNamespace
        from tools.run_sign2text_integrated import classify_controller_hands

        def hand_at(x):
            point = SimpleNamespace(x=x, y=.5)
            return SimpleNamespace(landmark=[point] * 21)
        left = SimpleNamespace(label="Left", score=.95)
        right = SimpleNamespace(label="Right", score=.95)
        result = lambda hands, labels: SimpleNamespace(
            multi_hand_landmarks=hands, multi_handedness=labels)
        paired = classify_controller_hands(result(
            [hand_at(.2), hand_at(.7)], [[left], [right]]))
        self.assertTrue(paired["reliable"])
        self.assertEqual(paired["controller"], "Left")
        self.assertEqual(paired["signing_hand"], "Right")
        right_only = classify_controller_hands(result([hand_at(.7)], [[right]]))
        self.assertFalse(right_only["reliable"])
        ambiguous = classify_controller_hands(result(
            [hand_at(.2), hand_at(.7)], [[left], [SimpleNamespace(label="Right", score=.4)]]))
        self.assertFalse(ambiguous["reliable"])

    def test_controller_open_palm_is_left_only_and_right_palm_is_not_controller(self):
        from types import SimpleNamespace
        from tools.run_sign2text_integrated import classify_controller_hands

        def hand_at(x, *, palm=False):
            points = [SimpleNamespace(x=x, y=.6) for _ in range(21)]
            if palm:
                for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
                    points[tip].y = .2
                    points[pip].y = .4
            return SimpleNamespace(landmark=points)

        left = SimpleNamespace(label="Left", score=.97)
        right = SimpleNamespace(label="Right", score=.97)
        result = lambda hands, labels: SimpleNamespace(
            multi_hand_landmarks=hands, multi_handedness=labels)
        left_palm = classify_controller_hands(result([hand_at(.2, palm=True)], [[left]]))
        self.assertTrue(left_palm["reliable"])
        self.assertEqual(left_palm["gesture"], "open_palm")
        right_palm = classify_controller_hands(result([hand_at(.7, palm=True)], [[right]]))
        self.assertFalse(right_palm["reliable"])
        self.assertIsNone(right_palm["gesture"])

    def test_left_hand_only_never_creates_word_candidate_but_right_hand_does(self):
        left_result = {"mode": "word", "sequence": 2, "input_enabled": True,
                       "prediction": {"label": "hello", "confidence": .97},
                       "signing_hand": None, "signing_hand_verified": False}
        self.assertIsNone(candidate_from_result(left_result, selected_mode="word"))
        left_labeled = {**left_result, "signing_hand": "Left",
                        "signing_hand_verified": True}
        self.assertIsNone(candidate_from_result(left_labeled, selected_mode="word"))
        right_result = {**left_result, "signing_hand": "Right", "signing_hand_verified": True}
        candidate = candidate_from_result(right_result, selected_mode="word")
        self.assertEqual(candidate.label, "hello")
        text = TextAccumulator()
        self.assertTrue(apply_candidate_action(text, candidate, "confirm"))
        self.assertEqual(text.text, "HELLO")
        text.clear()
        self.assertFalse(apply_worker_commit(text, {
            **left_labeled, "committed": "hello",
        }, selected_mode="word"))
        self.assertEqual(text.text, "")

    def test_word_candidate_requires_right_identity_and_worker_verification(self):
        base = {"mode": "word", "sequence": 2, "input_enabled": True,
                "prediction": {"label": "hello", "confidence": .97}}
        for extra in (
            {"signing_hand": None, "signing_hand_verified": False},
            {"signing_hand": "Left", "signing_hand_verified": True},
            {"signing_hand": "Right", "signing_hand_verified": False},
            {"signing_hand": "Right"},
        ):
            with self.subTest(extra=extra):
                self.assertIsNone(candidate_from_result({**base, **extra},
                                                        selected_mode="word"))

    def test_left_controller_confirm_works_with_word_role_gate(self):
        controller = ConfirmationController(.5)
        candidate = PendingCandidate("word", "hello", .97, 2)
        self.assertIsNone(controller.update("open_palm", 0, handedness_reliable=True,
                                            candidate=candidate))
        action = controller.update("open_palm", .51, handedness_reliable=True,
                                   candidate=candidate)
        self.assertEqual(action, "confirm")
        text = TextAccumulator()
        self.assertTrue(apply_candidate_action(text, candidate, action))
        controller.begin_cooldown(.51)
        self.assertTrue(controller.in_cooldown(1.0))
        self.assertEqual(text.text, "HELLO")

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
