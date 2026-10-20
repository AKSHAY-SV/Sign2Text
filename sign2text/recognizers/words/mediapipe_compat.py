"""Compatibility helpers for empty optional MediaPipe Holistic outputs.

MediaPipe 0.10.21's HolisticLandmarker result builder calls ``get_proto`` on
every landmark output packet after only checking the face packet. If another
output packet is empty, the native packet getter aborts the process. Converting
empty optional outputs to typed, empty landmark protos lets the existing source
preprocessor apply its established zero-fill behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def build_result_with_empty_landmarks(
    output_packets: Mapping[str, Any],
    original_builder: Callable[[Mapping[str, Any]], Any],
    create_proto_packet: Callable[[Any], Any],
    empty_proto_factories: Mapping[str, Callable[[], Any]],
) -> Any:
    """Call MediaPipe's builder after making empty landmark outputs safe.

    Each stream has its own protobuf type (normalized or world coordinates).
    The function copies the packet mapping and replaces only packets explicitly
    reported empty; nonempty packet values and stream ordering remain untouched.
    """
    safe_packets = dict(output_packets)
    for stream_name, make_empty_proto in empty_proto_factories.items():
        if stream_name not in safe_packets:
            raise KeyError(f"MediaPipe omitted expected Holistic output stream {stream_name!r}")
        packet = safe_packets[stream_name]
        if packet.is_empty():
            safe_packets[stream_name] = create_proto_packet(make_empty_proto())
    return original_builder(safe_packets)


def initialize_guarded_landmarker(
    source: Any,
    *,
    install_guard: Callable[[], None],
    factory: Callable[[Any], Any],
    mediapipe_version: str,
) -> Any:
    """Install the packet guard before constructing a word Holistic task.

    Kept injectable so the integrated worker's initialization order can be
    regression-tested without MediaPipe tasks or a webcam.
    """
    try:
        install_guard()
    except Exception as exc:
        raise RuntimeError(
            f"Could not install the MediaPipe {mediapipe_version} empty-packet guard: {exc}"
        ) from exc
    return factory(source)


def install_holistic_empty_packet_guard() -> None:
    """Guard MediaPipe's internal result conversion for this process.

    The installed MediaPipe Tasks implementation exposes the result conversion
    function as a module global used by ``HolisticLandmarker.detect``. Replacing
    that conversion function avoids calling the native proto getter with an
    empty optional packet. This does not change task options or detector output.
    """
    from mediapipe.framework.formats import landmark_pb2
    from mediapipe.python import packet_creator
    from mediapipe.tasks.python.vision import holistic_landmarker

    module = holistic_landmarker
    original_builder = getattr(module, "_build_landmarker_result", None)
    if not callable(original_builder):
        raise RuntimeError(
            "This MediaPipe version does not expose the Holistic result builder "
            "needed for the empty-packet compatibility guard."
        )
    if getattr(original_builder, "_sign2text_empty_packet_guard", False):
        return

    try:
        stream_specs = (
            (module._FACE_LANDMARKS_STREAM_NAME, landmark_pb2.NormalizedLandmarkList),
            (module._POSE_LANDMARKS_STREAM_NAME, landmark_pb2.NormalizedLandmarkList),
            (module._POSE_WORLD_LANDMARKS_STREAM_NAME, landmark_pb2.LandmarkList),
            (module._LEFT_HAND_LANDMARKS_STREAM_NAME, landmark_pb2.NormalizedLandmarkList),
            (module._LEFT_HAND_WORLD_LANDMARKS_STREAM_NAME, landmark_pb2.LandmarkList),
            (module._RIGHT_HAND_LANDMARKS_STREAM_NAME, landmark_pb2.NormalizedLandmarkList),
            (module._RIGHT_HAND_WORLD_LANDMARKS_STREAM_NAME, landmark_pb2.LandmarkList),
        )
    except AttributeError as exc:
        raise RuntimeError(
            "This MediaPipe version is missing expected Holistic output stream "
            "definitions; refusing to install an unverified packet guard."
        ) from exc
    empty_proto_factories = {name: proto_type for name, proto_type in stream_specs}

    def safe_builder(output_packets):
        return build_result_with_empty_landmarks(
            output_packets,
            original_builder,
            packet_creator.create_proto,
            empty_proto_factories,
        )

    safe_builder._sign2text_empty_packet_guard = True
    module._build_landmarker_result = safe_builder
