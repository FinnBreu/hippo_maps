import unittest
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
from hippo_maps.msg import OccupancyGrid3D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from occupancy_grid import (OCCUPIED, load_colors, unpack_message,
                            visible_voxels)
from occupancy_grid_bridge import to_voxel_grid


class OccupancyGridVisualizationTest(unittest.TestCase):
    def test_byte_grid_coordinates(self):
        message = OccupancyGrid3D()
        message.header.frame_id = 'map'
        message.position_max.x = 2.0
        message.position_max.y = 2.0
        message.position_max.z = 1.0
        message.resolution_per_meter = 1
        # x fastest: unknown, free, occupied, unknown.
        message.data = [0, 1, 2, 0]
        lower, upper, resolution, states = unpack_message(message)
        centres, codes = visible_voxels(lower, upper, resolution, states)
        np.testing.assert_array_equal(codes, [OCCUPIED])
        np.testing.assert_allclose(centres, [[0.5, 1.5, 0.5]])

    def test_rejects_bad_data_length(self):
        message = OccupancyGrid3D()
        message.position_max.x = 2
        message.position_max.y = 2
        message.position_max.z = 1
        message.resolution_per_meter = 1
        with self.assertRaisesRegex(ValueError, 'wrong length'):
            unpack_message(message)

    def test_rejects_reserved_state(self):
        message = OccupancyGrid3D()
        message.position_max.x = message.position_max.y = \
            message.position_max.z = 1
        message.resolution_per_meter = 1
        message.data = [3]
        with self.assertRaisesRegex(ValueError, 'reserved'):
            unpack_message(message)

    @unittest.skipUnless(importlib.util.find_spec('foxglove_msgs'),
                         'foxglove_msgs is not installed')
    def test_native_foxglove_voxel_grid(self):
        message = OccupancyGrid3D()
        message.header.frame_id = 'map'
        message.position_min.z = -1.0
        message.position_max.x = 2.0
        message.position_max.y = 2.0
        message.position_max.z = 0.0
        message.resolution_per_meter = 1
        message.data = [0, 1, 2, 0]
        grid = to_voxel_grid(message)
        self.assertEqual((grid.column_count, grid.row_count), (2, 2))
        self.assertEqual((grid.cell_stride, grid.row_stride,
                          grid.slice_stride), (5, 10, 20))
        self.assertEqual(grid.pose.position.z, -1.0)
        self.assertEqual(grid.pose.orientation.w, 1.0)
        self.assertEqual([field.name for field in grid.fields],
                         ['state', 'red', 'green', 'blue', 'alpha'])
        cells = np.frombuffer(grid.data, dtype=np.uint8).reshape(4, 5)
        np.testing.assert_array_equal(cells[:, 0], [0, 1, 2, 0])
        palette = load_colors()
        np.testing.assert_array_equal(cells[2, 1:4], palette[OCCUPIED][0])
        np.testing.assert_array_equal(
            cells[:, 4], [palette[state][1] for state in message.data])

    def test_custom_color_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'colors.yaml'
            path.write_text('''unknown: {rgb: [1, 2, 3], foxglove_alpha: 0, viewer_alpha: 0.5}
free: {rgb: [4, 5, 6], foxglove_alpha: 0, viewer_alpha: 0.0}
occupied: {rgb: [10, 20, 30], foxglove_alpha: 40, viewer_alpha: 0.7}
''')
            message = OccupancyGrid3D()
            message.position_max.x = message.position_max.y = \
                message.position_max.z = 1
            message.resolution_per_meter = 1
            message.data = [OCCUPIED]
            grid = to_voxel_grid(message, load_colors(path))
            self.assertEqual(list(grid.data), [2, 10, 20, 30, 40])


if __name__ == '__main__':
    unittest.main()
