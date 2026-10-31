import unittest
from aerofusion_ros2.frames import generate_rgb_frame

class FrameGenerationTests(unittest.TestCase):
    def test_dimensions_and_pixel_count(self):
        self.assertEqual(len(generate_rgb_frame(128, 96, 0)), 128 * 96 * 3)

    def test_deterministic_frames(self):
        self.assertEqual(generate_rgb_frame(128, 96, 4), generate_rgb_frame(128, 96, 4))

    def test_object_moves(self):
        self.assertNotEqual(generate_rgb_frame(128, 96, 0), generate_rgb_frame(128, 96, 1))

    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            generate_rgb_frame(20, 96, 0)

if __name__ == '__main__':
    unittest.main()
