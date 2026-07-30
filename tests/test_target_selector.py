import unittest

import numpy as np

from robot_video_client import (
    TargetSelector,
    assign_recognized_identities,
    is_exit_key,
    scale_yunet_face,
)


def face(name, x, y, area=1000, confidence=0.9):
    return {
        "name": name,
        "raw_name": name,
        "proba": 0.9,
        "center_x": x,
        "center_y": y,
        "area": area,
        "confidence": confidence,
    }


class TargetSelectorTests(unittest.TestCase):
    def test_video_exit_keys(self):
        self.assertTrue(is_exit_key(ord("q")))
        self.assertTrue(is_exit_key(ord("Q")))
        self.assertTrue(is_exit_key(27))
        self.assertFalse(is_exit_key(-1))

    def test_initial_selection_prefers_recognized_face(self):
        selector = TargetSelector(max_distance=100, hold_frames=2)

        selected = selector.select([
            face("unknown", 20, 20, area=5000),
            face("ana", 80, 20, area=1000),
        ])

        self.assertEqual(selected["name"], "ana")

    def test_selection_stays_with_nearest_previous_face(self):
        selector = TargetSelector(max_distance=100, hold_frames=2)
        selector.select([face("unknown", 20, 20)])

        selected = selector.select([
            face("unknown", 25, 20, area=1000),
            face("unknown", 100, 20, area=5000),
        ])

        self.assertEqual(selected["center_x"], 25)

    def test_known_identity_wins_among_nearby_faces(self):
        selector = TargetSelector(max_distance=100, hold_frames=2)
        selector.select([face("ana", 50, 50)])

        selected = selector.select([
            face("unknown", 52, 50),
            face("ana", 65, 50),
        ])

        self.assertEqual(selected["name"], "ana")

    def test_nearby_different_identity_does_not_replace_known_target(self):
        selector = TargetSelector(max_distance=100, hold_frames=2)
        selector.select([face("ana", 50, 50)])

        selected = selector.select([face("bob", 52, 50)])

        self.assertIsNone(selected)

    def test_known_identity_is_retained_between_recognition_frames(self):
        selector = TargetSelector(max_distance=100, hold_frames=2)
        selector.select([face("ana", 50, 50)])

        selected = selector.select([face("unknown", 54, 52)])

        self.assertEqual(selected["name"], "ana")
        self.assertEqual(selected["raw_name"], "ana")
        self.assertEqual(selected["proba"], 0.9)
        self.assertEqual(selected["center_x"], 54)

    def test_selector_waits_before_switching_target(self):
        selector = TargetSelector(max_distance=20, hold_frames=2)
        selector.select([face("ana", 10, 10)])
        other = face("bob", 200, 200)

        self.assertIsNone(selector.select([other]))
        self.assertIsNone(selector.select([other]))
        self.assertEqual(selector.select([other])["name"], "bob")


class SFaceTrackingTests(unittest.TestCase):
    def test_yunet_coordinates_are_scaled_to_output_frame(self):
        detected = np.array([
            10, 20, 30, 40,
            15, 25, 30, 25, 22, 35, 16, 50, 31, 50,
            0.95,
        ], dtype=np.float32)

        scaled = scale_yunet_face(
            detected, (240, 320, 3), (480, 640, 3)
        )

        np.testing.assert_allclose(scaled[:14], detected[:14] * 2)
        self.assertEqual(scaled[-1], detected[-1])

    def test_recognized_identity_is_assigned_to_nearest_detection(self):
        candidates = [
            face("unknown", 40, 40),
            face("unknown", 200, 100),
        ]
        identities = [{
            "name": "ana",
            "raw_name": "ana",
            "proba": 0.97,
            "center_x": 205,
            "center_y": 102,
            "area": 1000,
        }]

        assign_recognized_identities(candidates, identities)

        self.assertEqual(candidates[0]["name"], "unknown")
        self.assertEqual(candidates[1]["name"], "ana")
        self.assertEqual(candidates[1]["proba"], 0.97)


if __name__ == "__main__":
    unittest.main()
