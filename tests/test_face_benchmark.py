import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.benchmark_face_backends import image_number, scan_dataset


class FaceBenchmarkTests(unittest.TestCase):
    def test_image_number_sorts_numeric_suffixes(self):
        paths = [Path("face_10.jpg"), Path("face_2.jpg")]
        self.assertEqual(
            [path.name for path in sorted(paths, key=image_number)],
            ["face_2.jpg", "face_10.jpg"],
        )

    def test_scan_uses_last_readable_images_for_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for person in ("ana", "bob"):
                person_dir = root / person
                person_dir.mkdir()
                for index in range(1, 5):
                    image = np.full((20, 20, 3), index, dtype=np.uint8)
                    cv2.imwrite(str(person_dir / f"{person}_{index:04d}.jpg"),
                                image)

            identities, unreadable = scan_dataset(root, validation_count=2)

        self.assertEqual(unreadable, [])
        self.assertEqual(
            [path.name for path in identities["ana"]["validation"]],
            ["ana_0003.jpg", "ana_0004.jpg"],
        )


if __name__ == "__main__":
    unittest.main()
