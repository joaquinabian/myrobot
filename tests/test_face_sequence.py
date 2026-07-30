import json
import unittest

import numpy as np

from tools.benchmark_face_sequence import (
    SFaceSequenceBackend,
    phase_name_at,
    summarize,
)
from tools.capture_face_sequence import phase_at


class FaceSequenceTests(unittest.TestCase):
    def test_phase_boundaries(self):
        self.assertEqual(phase_at(0.0)[0], "center")
        self.assertEqual(phase_at(5.0)[0], "turn_left")
        self.assertEqual(phase_at(39.9)[0], "leave_and_return")
        self.assertIsNone(phase_at(40.0))

    def test_sface_landmarks_scale_to_original_frame(self):
        face = np.arange(15, dtype=np.float32)
        scaled = SFaceSequenceBackend.scale_face(face, 2.0, 3.0)

        for index in (0, 2, 4, 6, 8, 10, 12):
            self.assertEqual(scaled[index], face[index] * 2.0)
        for index in (1, 3, 5, 7, 9, 11, 13):
            self.assertEqual(scaled[index], face[index] * 3.0)
        self.assertEqual(scaled[14], face[14])

    def test_summary_accepts_numpy_values_and_is_json_serializable(self):
        summary = summarize([np.int64(10), np.int64(20)])
        json.dumps(summary)
        self.assertEqual(summary["median"], 10.0)

    def test_sequence_phase_lookup(self):
        phases = [("first", 0.0, 5.0), ("second", 5.0, 10.0)]
        self.assertEqual(phase_name_at(phases, 4.9), "first")
        self.assertEqual(phase_name_at(phases, 5.0), "second")
        self.assertEqual(phase_name_at(phases, 10.0), "unassigned")


if __name__ == "__main__":
    unittest.main()
