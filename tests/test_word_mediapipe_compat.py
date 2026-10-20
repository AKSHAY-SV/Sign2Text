from dataclasses import dataclass
import unittest

from sign2text.recognizers.words.mediapipe_compat import build_result_with_empty_landmarks


@dataclass
class FakePacket:
    value: object = None

    def is_empty(self):
        return self.value is None


class EmptyPacketCompatibilityTests(unittest.TestCase):
    def test_empty_optional_packets_become_typed_empty_protos(self):
        packets = {
            "face": FakePacket("face-landmarks"),
            "pose": FakePacket(),
            "left_hand": FakePacket("left-hand-landmarks"),
            "right_hand": FakePacket(),
        }
        factories = {
            "face": lambda: ("normalized", "face"),
            "pose": lambda: ("normalized", "pose"),
            "left_hand": lambda: ("normalized", "left_hand"),
            "right_hand": lambda: ("normalized", "right_hand"),
        }
        created = []

        def create_proto(proto):
            packet = FakePacket(proto)
            created.append(packet)
            return packet

        def original_builder(safe_packets):
            self.assertIsNot(safe_packets, packets)
            return {name: packet.value for name, packet in safe_packets.items()}

        result = build_result_with_empty_landmarks(
            packets, original_builder, create_proto, factories
        )

        self.assertEqual(result["pose"], ("normalized", "pose"))
        self.assertEqual(result["right_hand"], ("normalized", "right_hand"))
        self.assertEqual(result["face"], "face-landmarks")
        self.assertEqual(result["left_hand"], "left-hand-landmarks")
        self.assertEqual(len(created), 2)
        self.assertTrue(packets["pose"].is_empty())  # Caller-owned input is unchanged.

    def test_missing_expected_stream_fails_clearly(self):
        with self.assertRaisesRegex(KeyError, "pose"):
            build_result_with_empty_landmarks(
                {"face": FakePacket("face")},
                lambda packets: packets,
                lambda proto: FakePacket(proto),
                {"pose": tuple},
            )


if __name__ == "__main__":
    unittest.main()
