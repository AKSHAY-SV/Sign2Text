from types import SimpleNamespace

import numpy as np

from src.text.buffer import TextBuffer
from src.vision.hand_crop import crop_hand_square, crop_hand_with_bounds


def hand(x0=0.25, y0=0.2, dx=0.01, dy=0.01):
    return [SimpleNamespace(x=x0 + (i % 5) * dx, y=y0 + (i // 5) * dy) for i in range(21)]


def test_crop_is_square_and_edge_safe():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    crop = crop_hand_square(frame, hand(x0=0.01, y0=0.01))
    assert crop is not None
    assert crop.shape[0] == crop.shape[1]
    assert crop.dtype == frame.dtype


def test_crop_and_classifier_input_come_from_landmark_roi_not_full_frame():
    from src.model.cnn_classifier import preprocess_image

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    # Simulate a brightly colored hand-sized object around the detected landmarks.
    frame[20:95, 45:110] = (20, 80, 240)
    points = [SimpleNamespace(x=x / 160, y=y / 120)
              for y in (30, 42, 54, 66, 78, 90)
              for x in (55, 63, 71, 79)]
    # MediaPipe supplies 21 landmarks; use 21 points spanning the same hand ROI.
    points = points[:21]
    geometry = crop_hand_with_bounds(frame, points)
    assert geometry is not None
    assert geometry.source_shape == (120, 160)
    assert geometry.landmark_bounds[0] >= 50 and geometry.landmark_bounds[2] <= 90
    assert geometry.image.shape[0] == geometry.image.shape[1]
    assert geometry.image.shape[:2] != frame.shape[:2]

    cnn_input = preprocess_image(geometry.image, (64, 64))
    full_frame_input = preprocess_image(frame, (64, 64))
    assert cnn_input.shape == (64, 64, 3)
    assert cnn_input.dtype == np.float32
    assert not np.array_equal(cnn_input, full_frame_input)
    assert np.mean(cnn_input[:, :, 2]) > np.mean(full_frame_input[:, :, 2])


def test_invalid_or_missing_hand_has_no_crop():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert crop_hand_square(frame, []) is None
    assert crop_hand_square(frame, hand(x0=1.1)) is None
    assert crop_hand_square(frame, [object()] * 21) is None


def test_text_buffer_assembles_words_and_special_signs():
    buffer = TextBuffer()
    for letter in "CAT":
        buffer.add_sign(letter)
    assert buffer.sentence == "CAT"
    buffer.add_space()
    for letter in "HELLO":
        buffer.add_sign(letter)
    assert buffer.sentence == "CAT HELLO"
    buffer.add_sign("nothing")
    assert buffer.sentence == "CAT HELLO"
    buffer.add_sign("del")
    assert buffer.sentence == "CAT HELL"
    buffer.backspace()
    assert buffer.sentence == "CAT HEL"
    buffer.clear()
    assert buffer.sentence == ""


def test_held_label_does_not_auto_insert_repeated_letters():
    buffer = TextBuffer()
    held_label = "A"
    for _ in range(30):
        # The demo only appends on an explicit Enter event.
        assert buffer.sentence == ""
    buffer.add_sign(held_label)
    assert buffer.sentence == "A"


def test_buffer_space_backspace_and_clear_are_consistent():
    buffer = TextBuffer()
    for letter in "CAT":
        buffer.add_sign(letter)
    buffer.add_space()
    assert buffer.sentence == "CAT"
    buffer.add_space()
    assert buffer.sentence == "CAT "
    buffer.add_space()
    assert buffer.sentence == "CAT "
    buffer.backspace()
    assert buffer.sentence == "CA"
    buffer.clear()
    assert buffer.sentence == ""
