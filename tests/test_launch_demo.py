from types import SimpleNamespace
import queue
import threading
import unittest

import numpy as np

import launch_demo


def landmark_sequence(count, offset=0):
    return [SimpleNamespace(x=(i + offset) / count, y=(i + offset + 1) / count)
            for i in range(count)]


def holistic_results():
    return SimpleNamespace(
        pose_landmarks=SimpleNamespace(landmark=landmark_sequence(33)),
        left_hand_landmarks=SimpleNamespace(landmark=landmark_sequence(21, 40)),
        right_hand_landmarks=SimpleNamespace(landmark=landmark_sequence(21, 80)),
        face_landmarks=SimpleNamespace(landmark=landmark_sequence(478, 120)),
    )


def training_normalize(frame):
    """Independent transcription of preprocessing_split.py's training transforms."""
    frame = frame.copy()

    def distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def anchor(target, scale, reference):
        return (target - reference) / (scale + 0.01)

    neck = np.abs((frame[0] + frame[1]) / 2)
    frame[:] = anchor(frame, distance(frame[0], frame[1]), neck)
    frame[54:] = anchor(frame[54:], distance(frame[0], frame[1]), frame[79])
    frame[[2, 4, 6, 8, 10]] = anchor(
        frame[[2, 4, 6, 8, 10]], distance(frame[0], frame[2]), frame[0]
    )
    frame[[3, 5, 7, 9, 11]] = anchor(
        frame[[3, 5, 7, 9, 11]], distance(frame[1], frame[3]), frame[1]
    )
    for hand in (frame[12:33], frame[33:54]):
        for coordinate in (0, 1):
            low, high = hand[:, coordinate].min(), hand[:, coordinate].max()
            if high != low:
                hand[:, coordinate] = (hand[:, coordinate] - (low + high) / 2) / (high - low)
            else:
                hand[:, coordinate] = 0
    return frame


class LaunchDemoTests(unittest.TestCase):
    def test_feature_order_and_face_features_are_retained(self):
        features = launch_demo.extract_landmarks(holistic_results())

        self.assertEqual(features.shape, (86, 2))
        self.assertTrue(np.allclose(features[0], [11 / 33, 12 / 33]))
        self.assertTrue(np.allclose(features[12], [40 / 21, 41 / 21]))
        self.assertTrue(np.allclose(features[33], [80 / 21, 81 / 21]))
        face_index = launch_demo.FACE_LANDMARK_INDICES[0]
        self.assertTrue(np.allclose(features[54], [(face_index + 120) / 478,
                                                   (face_index + 121) / 478]))


    def test_frame_normalization_matches_training_preprocessing(self):
        source = np.arange(86 * 2, dtype=np.float32).reshape(86, 2) / 100
        expected = training_normalize(source)

        self.assertTrue(np.allclose(launch_demo.normalize_frame(source), expected))
        self.assertTrue(np.array_equal(source, np.arange(86 * 2, dtype=np.float32).reshape(86, 2) / 100))
    def test_model_input_shape_and_sequence_length(self):
        valid = np.zeros((150, 86, 2), dtype=np.float32)
        self.assertEqual(launch_demo.build_model_input(valid).shape, (1, 150, 172))
        with self.assertRaisesRegex(ValueError, "Expected sequence"):
            launch_demo.build_model_input(valid[:-1])


    def test_mapped_and_unmapped_output_handling_including_270(self):
        labels = {"0": "ABOUT1", "208": "YOUR"}
        probabilities = np.zeros(271)
        probabilities[270] = 0.8
        probabilities[208] = 0.15
        probabilities[0] = 0.05

        label, index, confidence, status, top = launch_demo.prediction_description(probabilities, labels)

        self.assertEqual((label, index, confidence, status), (None, 270, 0.8, "unmapped output 270"))
        self.assertEqual(top, [(208, "YOUR", 0.15), (0, "ABOUT1", 0.05)])

        probabilities[208] = 0.9
        mapped = launch_demo.prediction_description(probabilities, labels)
        self.assertEqual(mapped[:4], ("YOUR", 208, 0.9, "mapped"))


    def test_prediction_refresh_keeps_full_150_frame_window(self):
        self.assertFalse(launch_demo.is_prediction_frame(149, 5))
        self.assertTrue(launch_demo.is_prediction_frame(150, 5))
        self.assertFalse(launch_demo.is_prediction_frame(154, 5))
        self.assertTrue(launch_demo.is_prediction_frame(155, 5))


    def test_inference_worker_accepts_only_full_shape_inputs_and_returns_scores(self):
        class StubModel:
            def predict(self, model_input, verbose=0):
                self.input_shape = model_input.shape
                scores = np.zeros((1, 271), dtype=np.float32)
                scores[0, 208] = 1.0
                return scores

        jobs = queue.Queue(maxsize=1)
        results = queue.Queue(maxsize=1)
        stop = threading.Event()
        busy = threading.Event()
        busy.set()
        model = StubModel()
        jobs.put((150, np.zeros((1, 150, 172), dtype=np.float32)))
        stop.set()
        worker = threading.Thread(
            target=launch_demo.inference_worker,
            args=(model, jobs, results, stop, busy),
        )
        worker.start()
        worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(model.input_shape, (1, 150, 172))
        frame_number, scores, _, error = results.get_nowait()
        self.assertEqual(frame_number, 150)
        self.assertIsNone(error)
        self.assertEqual(int(np.argmax(scores)), 208)
        self.assertFalse(busy.is_set())


    def test_visualizer_draws_only_pose_and_hands_and_preserves_face_features(self):
        results = holistic_results()
        before = launch_demo.extract_landmarks(results)
        calls = []
        drawing = SimpleNamespace(draw_landmarks=lambda frame, landmarks, connections:
                                  calls.append((landmarks, connections)))
        holistic = SimpleNamespace(POSE_CONNECTIONS="pose", HAND_CONNECTIONS="hand")
        launch_demo.draw_visible_landmarks(object(), results, holistic, drawing)
        after = launch_demo.extract_landmarks(results)

        self.assertEqual(calls, [
            (results.pose_landmarks, "pose"),
            (results.left_hand_landmarks, "hand"),
            (results.right_hand_landmarks, "hand"),
        ])
        self.assertTrue(np.array_equal(before[54:], after[54:]))


if __name__ == "__main__":
    unittest.main()
