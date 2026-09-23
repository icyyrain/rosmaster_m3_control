import pathlib
import sys
import unittest

import cv2
import numpy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'pc_client'))

import reach_candy


class TargetGeometryTests(unittest.TestCase):

    def test_principal_axis_of_diagonal_bar(self):
        image = numpy.full((240, 320, 3), 220, numpy.uint8)
        box = cv2.boxPoints(((160, 120), (160, 40), 35))
        cv2.fillConvexPoly(image, box.astype(numpy.int32), (255, 0, 0))
        colour, _ = reach_candy.parse_colour('blue')

        geometry = reach_candy.target_blob_geometry(
            image, colour, (160, 120), threshold=12)

        self.assertIsNotNone(geometry)
        self.assertAlmostEqual(geometry['angle_deg'], 35.0, delta=1.0)
        self.assertGreater(geometry['long_px'] / geometry['short_px'], 3.5)

    def test_clipped_centre_moves_by_half_the_missing_length(self):
        initial = dict(long_px=200.0, short_px=40.0, axis=(0.0, 1.0))
        current = dict(long_px=100.0, short_px=40.0, axis=(0.0, 1.0))

        result = reach_candy.clipped_centre_offset(
            initial, current, 'bottom', smoothing=1.0)

        self.assertEqual(result['missing_long_px'], 100.0)
        self.assertAlmostEqual(result['offset'][0], 0.0)
        self.assertAlmostEqual(result['offset'][1], 50.0)

    def test_clipped_centre_follows_a_diagonal_axis(self):
        root_half = 2.0 ** -0.5
        initial = dict(long_px=200.0, short_px=40.0,
                       axis=(root_half, root_half))
        current = dict(long_px=100.0, short_px=40.0,
                       axis=(root_half, root_half))

        result = reach_candy.clipped_centre_offset(
            initial, current, 'right', smoothing=1.0)

        self.assertAlmostEqual(result['magnitude_px'], 50.0)
        self.assertAlmostEqual(result['offset'][0], 50.0 * root_half)
        self.assertAlmostEqual(result['offset'][1], 50.0 * root_half)

    def test_line_angle_is_unsigned_and_wraps_at_180(self):
        self.assertAlmostEqual(reach_candy.line_angle_delta(10, 170), 20.0)
        self.assertAlmostEqual(reach_candy.line_angle_delta(170, 10), -20.0)
        self.assertAlmostEqual(reach_candy.line_angle_delta(45, 225), 0.0)


if __name__ == '__main__':
    unittest.main()
