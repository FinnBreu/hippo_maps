import unittest
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
from hippo_maps.msg import OccupancyGrid3D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from occupancy_grid import (BUFFER, OCCUPIED, bounds_edges, load_colors,
                            unpack_message, visible_voxels)
from occupancy_grid_bridge import to_bounds_scene, to_voxel_grid


class OccupancyGridVisualizationTest(unittest.TestCase):
    def test_byte_grid_coordinates(self):
        message = OccupancyGrid3D()
        message.header.frame_id = 'map'
        message.position_max.x = 2.0
        message.position_max.y = 2.0
        message.position_max.z = 1.0
        message.resolution_per_meter = 1
        # x fastest: unknown, free, occupied, buffer.
        message.data = [0, 1, 2, 3]
        lower, upper, resolution, states = unpack_message(message)
        centres, codes = visible_voxels(lower, upper, resolution, states)
        np.testing.assert_array_equal(codes, [OCCUPIED, BUFFER])
        np.testing.assert_allclose(centres, [[0.5, 1.5, 0.5],
                                             [1.5, 1.5, 0.5]])

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
        message.data = [4]
        with self.assertRaisesRegex(ValueError, 'reserved'):
            unpack_message(message)

    def test_position_bounds_have_twelve_unique_edges(self):
        lower = np.array([-1.0, 2.0, -3.0])
        upper = np.array([1.0, 5.0, 0.0])
        edges = bounds_edges(lower, upper)
        self.assertEqual(len(edges), 12)
        segments = set()
        corners = set()
        for start, end in edges:
            first, second = tuple(start), tuple(end)
            segments.add((first, second))
            corners.update((first, second))
            self.assertEqual(np.count_nonzero(start != end), 1)
        self.assertEqual(len(segments), 12)
        self.assertEqual(len(corners), 8)

    @unittest.skipUnless(importlib.util.find_spec('foxglove_msgs'),
                         'foxglove_msgs is not installed')
    def test_foxglove_position_bounds_are_lines(self):
        from foxglove_msgs.msg import LinePrimitive

        message = OccupancyGrid3D()
        message.header.frame_id = 'map'
        message.position_min.x = -1.0
        message.position_min.z = -2.0
        message.position_max.x = 1.0
        message.position_max.y = 3.0
        message.position_max.z = 0.0
        message.resolution_per_meter = 1
        message.data = [1] * 12
        scene = to_bounds_scene(message)
        self.assertEqual(len(scene.entities), 1)
        entity = scene.entities[0]
        self.assertEqual((entity.frame_id, entity.id),
                         ('map', 'position_bounds'))
        self.assertEqual(len(entity.lines), 1)
        line = entity.lines[0]
        self.assertEqual(line.type, LinePrimitive.LINE_LIST)
        self.assertTrue(line.scale_invariant)
        self.assertEqual(len(line.points), 24)
        self.assertEqual(len(entity.cubes), 0)

    def test_local_viewer_draws_position_edges(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from view_occupancy_grid import make_figure

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'map.npz'
            np.savez_compressed(path, position_min=[0, 0, -1],
                                position_max=[2, 3, 0],
                                resolution_per_meter=1,
                                frame_id='map', format_version=2,
                                data=np.ones(6, dtype=np.uint8))
            figure = make_figure(path)
            try:
                self.assertEqual(len(figure.axes[0].lines), 12)
            finally:
                plt.close(figure)

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
        message.data = [0, 1, 2, 3]
        grid = to_voxel_grid(message)
        self.assertEqual((grid.column_count, grid.row_count), (2, 2))
        self.assertEqual((grid.cell_stride, grid.row_stride,
                          grid.slice_stride), (5, 10, 20))
        self.assertEqual(grid.pose.position.z, -1.0)
        self.assertEqual(grid.pose.orientation.w, 1.0)
        self.assertEqual([field.name for field in grid.fields],
                         ['state', 'red', 'green', 'blue', 'alpha'])
        cells = np.frombuffer(grid.data, dtype=np.uint8).reshape(4, 5)
        np.testing.assert_array_equal(cells[:, 0], [0, 1, 2, 3])
        palette = load_colors()
        np.testing.assert_array_equal(cells[2, 1:4], palette[OCCUPIED][0])
        np.testing.assert_array_equal(cells[3, 1:4], palette[BUFFER][0])
        self.assertLess(cells[3, 4], cells[2, 4])
        np.testing.assert_array_equal(
            cells[:, 4], [palette[state][1] for state in message.data])

    @unittest.skipUnless(importlib.util.find_spec('foxglove_msgs'),
                         'foxglove_msgs is not installed')
    def test_occupied_and_buffer_display_layers_are_separate(self):
        message = OccupancyGrid3D()
        message.position_max.x = 3
        message.position_max.y = message.position_max.z = 1
        message.resolution_per_meter = 1
        message.data = [BUFFER, OCCUPIED, BUFFER]

        occupied_grid = to_voxel_grid(message, visible_states=(OCCUPIED,))
        buffer_grid = to_voxel_grid(message, visible_states=(BUFFER,))
        occupied_alpha = np.frombuffer(occupied_grid.data, dtype=np.uint8) \
            .reshape(3, 5)[:, 4]
        buffer_alpha = np.frombuffer(buffer_grid.data, dtype=np.uint8) \
            .reshape(3, 5)[:, 4]
        palette = load_colors()
        np.testing.assert_array_equal(occupied_alpha,
                                      [0, palette[OCCUPIED][1], 0])
        np.testing.assert_array_equal(buffer_alpha,
                                      [palette[BUFFER][1], 0,
                                       palette[BUFFER][1]])

    def test_custom_color_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'colors.yaml'
            path.write_text('''unknown: {rgb: [1, 2, 3], foxglove_alpha: 0, viewer_alpha: 0.5}
free: {rgb: [4, 5, 6], foxglove_alpha: 0, viewer_alpha: 0.0}
occupied: {rgb: [10, 20, 30], foxglove_alpha: 40, viewer_alpha: 0.7}
buffer: {rgb: [15, 25, 35], foxglove_alpha: 20, viewer_alpha: 0.3}
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
