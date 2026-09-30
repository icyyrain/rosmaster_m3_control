import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'pc_client'))

import kinematics
import pick_and_place


class PlaceTrajectoryTests(unittest.TestCase):

    def assert_low_release(self, grasp_pose):
        release, clearance = pick_and_place.low_release_pose(grasp_pose)

        self.assertAlmostEqual(clearance, 10.0, delta=2.0)
        self.assertEqual(release[0], grasp_pose[0])
        self.assertEqual(release[3:], grasp_pose[3:])
        for index, target in ((1, 33), (2, 53)):
            self.assertGreaterEqual(
                release[index], min(grasp_pose[index], target))
            self.assertLessEqual(
                release[index], max(grasp_pose[index], target))
        self.assertGreater(
            kinematics.gripper_height_mm(release),
            kinematics.gripper_height_mm(grasp_pose))

    def test_blue_biscuit_releases_ten_mm_above_grasp(self):
        self.assert_low_release([96, 10, 54, 30, 57, 153])

    def test_red_sweet_releases_ten_mm_above_grasp(self):
        self.assert_low_release([90, 18, 62, 16, 90, 160])


if __name__ == '__main__':
    unittest.main()
