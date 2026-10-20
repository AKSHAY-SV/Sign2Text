"""Construction helper for the integrated word worker's guarded Holistic task."""

from __future__ import annotations

from typing import Any, Callable

from .mediapipe_compat import initialize_guarded_landmarker


def create_integrated_word_landmarker(
    source: Any,
    *,
    install_guard: Callable[[], None] | None = None,
    factory: Callable[[Any], Any] | None = None,
    mediapipe_version: str | None = None,
) -> Any:
    """Install the empty-packet guard before constructing the worker's task."""
    if install_guard is None:
        from .mediapipe_compat import install_holistic_empty_packet_guard
        install_guard = install_holistic_empty_packet_guard
    if factory is None:
        from tools.run_word_demo import create_local_landmarker

        factory = lambda value: create_local_landmarker(
            value, install_compatibility_guard=False
        )
    if mediapipe_version is None:
        import mediapipe as mp

        mediapipe_version = mp.__version__
    return initialize_guarded_landmarker(
        source,
        install_guard=install_guard,
        factory=factory,
        mediapipe_version=mediapipe_version,
    )
