import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import cv2

from src.model.cnn_classifier import CnnClassifier, CLASS_NAMES, preprocess_image
from src.prediction.predictor import PredictionStabilizer, RealTimePredictor
from src.vision.display import to_display_uint8
from src.vision.hand_crop import crop_hand_square
from main import _compose, _insert
from src.reference_recognizer import ReferenceSignRecognizer, preprocess_hand
from src.text.buffer import TextBuffer


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "models/candidate_asl_letters_v2"


def test_display_canvas_is_uint8_and_preserves_normalized_pixels():
    cnn_input = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    display = to_display_uint8(cnn_input)
    assert display.dtype == np.uint8
    np.testing.assert_array_equal(display[0, 0], [0, 128, 255])
    # Display conversion must not mutate or replace the normalized CNN tensor.
    assert cnn_input.dtype == np.float32
    np.testing.assert_array_equal(cnn_input[0, 0], [0.0, 0.5, 1.0])
    canvas = cv2.resize(display, (64, 64))
    cv2.putText(canvas, "B: 99%", (1, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 0), 1, cv2.LINE_AA)


def test_candidate_checkpoint_and_metadata_contract():
    model = CnnClassifier(model_path=CANDIDATE / "best.keras")
    metadata = json.loads((CANDIDATE / "metadata.json").read_text())
    assert model.is_trained()
    assert model.image_size == metadata["image_size"] == 64
    assert model.num_classes == metadata["num_classes"] == 29
    assert model.class_names == CLASS_NAMES == metadata["class_names"]
    assert model.model_path == (CANDIDATE / "best.keras").resolve()
    assert model.metadata_path == (CANDIDATE / "metadata.json").resolve()


def test_selected_model_text_mapping_and_single_canvas_composition():
    buffer = TextBuffer()
    for char in "CAT":
        assert _insert(buffer, char)
    assert buffer.sentence == "CAT"
    assert _insert(buffer, "space")
    for char in "HELLO":
        _insert(buffer, char)
    assert buffer.sentence == "CAT HELLO"
    _insert(buffer, "del")
    assert buffer.sentence == "CAT HELL"
    before = buffer.sentence
    assert not _insert(buffer, "nothing")
    assert buffer.sentence == before

    frame = np.zeros((80, 120, 3), np.uint8)
    composed = _compose(frame, True, None, None, ["NO HAND DETECTED"])
    # Diagnostics are panels in the returned camera canvas, not a second window.
    assert composed.ndim == 3 and composed.shape[0] > frame.shape[0]
    assert composed.shape[1] > frame.shape[1]


def test_reference_checkpoint_mapping_and_preprocessing_are_a_matched_pair():
    recognizer = ReferenceSignRecognizer()
    assert list(recognizer.model.input_shape[1:]) == [128, 128, 3]
    assert recognizer.model.output_shape[-1] == 36
    assert recognizer.classes == [chr(code) for code in range(65, 91)] + list("0123456789")
    crop = np.zeros((80, 40, 3), dtype=np.uint8)
    crop[:] = (0, 0, 255)  # BGR red, same order as OpenCV webcam crops
    prepared = preprocess_hand(crop)
    assert prepared.shape == (1, 128, 128, 3)
    assert prepared.dtype == np.float32
    assert 0 <= prepared.min() <= prepared.max() <= 1
    # Aspect-preserving white side bars and BGR->RGB conversion are retained.
    np.testing.assert_allclose(prepared[0, 64, 10], [1, 1, 1], atol=0.02)
    np.testing.assert_allclose(prepared[0, 64, 64], [1, 0, 0], atol=0.02)
    model_input, scores = recognizer.predict_crop(crop)
    assert model_input.shape == (1, 128, 128, 3)
    assert scores.shape == (36,)
    assert np.isfinite(scores).all() and np.isclose(scores.sum(), 1.0, atol=1e-3)


def test_main_has_one_application_window_and_reference_default():
    import ast
    source = (ROOT / "main.py").read_text()
    tree = ast.parse(source)
    cv_calls = [node.func.attr for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "cv2"]
    assert cv_calls.count("namedWindow") == 1
    assert cv_calls.count("imshow") == 1
    assert "ReferenceSignRecognizer()" in source
    assert "--candidate-fallback" in source and "--model-dir" in source


def test_duplicate_aware_split_keeps_exact_duplicates_on_one_side(tmp_path):
    import cv2
    from train_cnn import split_files

    rng = np.random.default_rng(87)
    files = {"A": [], "B": []}
    for class_name in files:
        for index in range(12):
            image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            path = tmp_path / f"{class_name}_{index}.png"
            assert cv2.imwrite(str(path), image)
            files[class_name].append(path)
    duplicate = tmp_path / "A_duplicate.png"
    assert cv2.imwrite(str(duplicate), cv2.imread(str(files["A"][0])))
    files["A"].append(duplicate)

    train_paths, _, val_paths, _, classes, stats = split_files(files, .2, 19)
    train_set, val_set = set(train_paths), set(val_paths)
    assert classes == ["A", "B"]
    assert not (train_set & val_set)
    assert (files["A"][0] in train_set) == (duplicate in train_set)
    assert stats["near_duplicate_groups"] >= 1


def test_preprocess_contract_and_horizontal_mirror():
    image = np.zeros((17, 23, 3), dtype=np.uint8)
    image[:, :12] = (0, 0, 255)  # BGR red on left
    prepared = preprocess_image(image, (64, 64))
    flipped = preprocess_image(np.ascontiguousarray(image[:, ::-1]), (64, 64))
    assert prepared.shape == (64, 64, 3)
    assert prepared.dtype == np.float32
    assert 0 <= prepared.min() <= prepared.max() <= 1
    np.testing.assert_allclose(prepared[8, 8], [1.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(prepared[:, ::-1], flipped, atol=0.01)


def test_empty_invalid_and_mirrored_landmark_crop():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    assert RealTimePredictor._crop_hand_from_frame(frame, []) is None
    assert RealTimePredictor._crop_hand_from_frame(frame, [object()]) is None
    points = [SimpleNamespace(x=0.25 + (i % 3) * 0.01,
                               y=0.3 + (i % 5) * 0.01) for i in range(21)]
    crop = RealTimePredictor._crop_hand_from_frame(frame, points)
    mirrored = RealTimePredictor._crop_hand_from_frame(
        np.ascontiguousarray(frame[:, ::-1]),
        [SimpleNamespace(x=1 - p.x, y=p.y) for p in points],
    )
    assert crop is not None and mirrored is not None
    assert crop.shape == mirrored.shape
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert crop.shape[0] == crop.shape[1]


def test_crop_near_frame_edge_keeps_square_aspect_ratio():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    points = [SimpleNamespace(x=0.01 + (i % 3) * 0.015,
                              y=0.4 + (i % 5) * 0.02) for i in range(21)]
    crop = RealTimePredictor._crop_hand_from_frame(frame, points)
    assert crop is not None
    assert crop.shape[0] == crop.shape[1]


def test_crop_rejects_out_of_frame_and_zero_area_landmarks():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    outside = [SimpleNamespace(x=1.2, y=0.5) for _ in range(21)]
    same_point = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    assert RealTimePredictor._crop_hand_from_frame(frame, outside) is None
    assert RealTimePredictor._crop_hand_from_frame(frame, same_point) is None


def test_stale_detection_dimensions_clear_prediction_without_inference():
    class FakeClassifier:
        image_size = 64
        class_names = CLASS_NAMES
        def predict(self, image):
            raise AssertionError("stale landmarks must not reach the classifier")

    class FakeDetector:
        pass

    predictor = RealTimePredictor(
        detector=FakeDetector(), cnn_classifier=FakeClassifier(),
        stabilizer=PredictionStabilizer(stable_frames=1, min_consistent_predictions=1),
    )
    predictor._current_prediction = "B"
    predictor._current_confidence = 0.99
    stale = SimpleNamespace(hands=[SimpleNamespace(landmarks=[
        SimpleNamespace(x=0.5, y=0.5) for _ in range(21)
    ])], frame_width=99, frame_height=100, timestamp_ms=1)
    result = predictor.predict_from_frame_cnn(np.zeros((100, 120, 3), np.uint8), stale)
    assert result == (None, None)
    assert predictor.current_prediction is None
    assert predictor._last_cnn_crop is None


def test_repeated_detection_timestamp_is_rejected_as_stale():
    class FakeClassifier:
        image_size = 64
        class_names = CLASS_NAMES
        idx_to_label = {i: name for i, name in enumerate(CLASS_NAMES)}
        def predict_proba(self, image):
            scores = np.zeros(29, dtype=np.float32)
            scores[1] = 1.0
            return scores

    predictor = RealTimePredictor(
        detector=SimpleNamespace(), cnn_classifier=FakeClassifier(),
        stabilizer=PredictionStabilizer(stable_frames=1, min_consistent_predictions=1),
    )
    landmarks = [SimpleNamespace(x=0.35 + (i % 4) * 0.05,
                                  y=0.35 + (i % 5) * 0.04) for i in range(21)]
    detection = SimpleNamespace(hands=[SimpleNamespace(landmarks=landmarks)],
                                frame_width=120, frame_height=100, timestamp_ms=5)
    frame = np.zeros((100, 120, 3), np.uint8)
    assert predictor.predict_from_frame_cnn(frame, detection)[0] == "B"
    assert predictor.predict_from_frame_cnn(frame, detection) == (None, None)
    assert predictor._last_cnn_crop is None


def test_prediction_and_confidence_threshold():
    stabilizer = PredictionStabilizer(stable_frames=2, confidence_threshold=0.6)
    assert stabilizer.update("B", 0.59) is None
    assert stabilizer.current_stable_label is None
    assert stabilizer.update("B", 0.95) is None
    assert stabilizer.update("B", 0.96) == "B"


def test_probability_smoothing_rejects_single_noisy_frame_and_accepts_consensus():
    stabilizer = PredictionStabilizer(
        confidence_threshold=0.6,
        probability_history_size=3,
        min_consistent_predictions=2,
    )
    names = ["N", "V", "W", "H"]
    v = np.array([0.05, 0.85, 0.05, 0.05], dtype=np.float32)
    n = np.array([0.85, 0.05, 0.05, 0.05], dtype=np.float32)
    assert stabilizer.update_probabilities(v, names) is None
    assert stabilizer.update_probabilities(v, names) == "V"
    # A single competing observation becomes uncertain; it cannot switch to N.
    assert stabilizer.update_probabilities(n, names) is None
    assert stabilizer.current_stable_label is None
    assert stabilizer.update_probabilities(v, names) is None
    assert stabilizer.update_probabilities(v, names) is None
    assert stabilizer.update_probabilities(v, names) is None
    assert stabilizer.update_probabilities(v, names) == "V"


def test_low_confidence_and_invalid_probability_vectors_become_uncertain():
    stabilizer = PredictionStabilizer(
        confidence_threshold=0.6, probability_history_size=1,
        min_consistent_predictions=1,
    )
    names = ["N", "V", "W", "H"]
    confident = np.array([0.02, 0.92, 0.03, 0.03], dtype=np.float32)
    low = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    assert stabilizer.update_probabilities(confident, names) == "V"
    assert stabilizer.update_probabilities(low, names) is None
    assert stabilizer.current_stable_label is None
    with pytest.raises(ValueError, match="normalized probability"):
        stabilizer.update_probabilities(np.array([0.2, 0.2]), names)


def test_confirmation_requires_stable_observations_and_commits_once():
    stabilizer = PredictionStabilizer(
        confidence_threshold=0.6, probability_history_size=3,
        min_consistent_predictions=3, confirmation_hold_seconds=0,
    )
    names = ["A", "B"]
    a = np.array([0.95, 0.05], dtype=np.float32)
    assert stabilizer.update_probabilities(a, names) is None
    assert stabilizer.update_probabilities(a, names) is None
    assert stabilizer.update_probabilities(a, names) == "A"
    assert stabilizer.commit("A") == "A"
    assert stabilizer.commit("A") is None
    assert stabilizer.commit("A", explicit=True) == "A"
    # A changed confirmed sign is a new deliberate character.
    b = np.array([0.05, 0.95], dtype=np.float32)
    stabilizer.update_probabilities(b, names)
    stabilizer.update_probabilities(b, names)
    assert stabilizer.update_probabilities(b, names) is None
    assert stabilizer.update_probabilities(b, names) == "B"
    assert stabilizer.commit("B") == "B"


def test_no_hand_clears_stable_candidate_and_rearms_same_letter():
    stabilizer = PredictionStabilizer(
        confidence_threshold=0.6, probability_history_size=1,
        min_consistent_predictions=1, confirmation_hold_seconds=0,
    )
    a = np.array([0.99, 0.01], dtype=np.float32)
    assert stabilizer.update_probabilities(a, ["A", "B"]) == "A"
    assert stabilizer.commit("A") == "A"
    stabilizer.reset()  # same reset used by the camera loop when no hand/crop exists
    assert stabilizer.current_stable_label is None
    assert stabilizer.commit("A") is None
    assert stabilizer.update_probabilities(a, ["A", "B"]) == "A"
    assert stabilizer.commit("A") == "A"


def test_candidate_model_and_metadata_are_selected_as_matching_pair():
    model_path = CANDIDATE / "best.keras"
    metadata_path = CANDIDATE / "metadata.json"
    classifier = CnnClassifier(model_path, metadata_path)
    assert classifier.model_path == model_path.resolve()
    assert classifier.metadata_path == metadata_path.resolve()
    assert classifier.metadata["input_shape"] == [64, 64, 3]
    assert classifier.class_names == classifier.metadata["class_names"]


def test_new_crop_helper_validity_and_rgb_normalized_preprocessing():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    points = [SimpleNamespace(x=0.2 + (i % 5) * .02, y=.2 + (i // 5) * .02)
              for i in range(21)]
    crop = crop_hand_square(frame, points)
    assert crop is not None and crop.shape[0] == crop.shape[1]
    assert crop.dtype == np.uint8
    assert crop_hand_square(frame, []) is None
    prepared = preprocess_image(crop, (64, 64))
    assert prepared.shape == (64, 64, 3)
    assert prepared.dtype == np.float32
    assert np.isfinite(prepared).all() and 0 <= prepared.min() <= prepared.max() <= 1


def test_bad_class_mapping_rejected(tmp_path):
    model = CnnClassifier()
    model.model = SimpleNamespace(output_shape=(None, 29))
    model._model_input_size = 64
    meta = json.loads((CANDIDATE / "metadata.json").read_text())
    meta["class_to_idx"]["B"], meta["class_to_idx"]["N"] = (
        meta["class_to_idx"]["N"], meta["class_to_idx"]["B"]
    )
    broken = tmp_path / "metadata.json"
    broken.write_text(json.dumps(meta))
    with patch.object(Path, "is_file", return_value=True):
        with patch("builtins.open", return_value=broken.open()):
            with pytest.raises(ValueError, match="class_to_idx"):
                model.load_metadata(str(broken))
