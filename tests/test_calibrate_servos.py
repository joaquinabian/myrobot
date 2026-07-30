import unittest

from tools.calibrate_servos import updated_angles


class UpdatedAnglesTests(unittest.TestCase):
    def test_wasd_moves_one_axis(self):
        self.assertEqual(updated_angles(ord("a"), 88, 90, 2, 1, 179),
                         (88, 88))
        self.assertEqual(updated_angles(ord("d"), 88, 90, 2, 1, 179),
                         (88, 92))
        self.assertEqual(updated_angles(ord("w"), 88, 90, 2, 1, 179),
                         (86, 90))
        self.assertEqual(updated_angles(ord("s"), 88, 90, 2, 1, 179),
                         (90, 90))

    def test_angles_are_clamped(self):
        self.assertEqual(updated_angles(ord("a"), 88, 1, 2, 1, 179),
                         (88, 1))
        self.assertEqual(updated_angles(ord("s"), 179, 90, 2, 1, 179),
                         (179, 90))


if __name__ == "__main__":
    unittest.main()
