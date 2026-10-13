import pathlib
import sys
import unittest
from unittest import mock

import numpy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'pc_client'))

import table_plane


class TablePlaneTests(unittest.TestCase):
    def test_negative_table_residual_does_not_flip_object_height(self):
        depth = numpy.full((60, 80), 300.08, numpy.float32)
        depth[24:37, 34:47] = 275.0
        plane = (numpy.array([0.0, 0.0, -1.0]), 300.0, 4000)

        with mock.patch.object(table_plane, 'fit_plane', return_value=plane):
            objects = table_plane.find_objects(depth, numpy.random.default_rng(31))

        self.assertEqual(len(objects), 1)
        self.assertAlmostEqual(objects[0]['top_mm'], 25.0, delta=0.1)
        self.assertAlmostEqual(objects[0]['dist_mm'], 275.0, delta=0.1)


if __name__ == '__main__':
    unittest.main()
