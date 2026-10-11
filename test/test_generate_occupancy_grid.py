import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import generate_occupancy_grid as maps


class OccupancyGridGeneratorTest(unittest.TestCase):
    def test_byte_states_and_indexing(self):
        grid = maps.VoxelGrid([0, 0, 0], [4, 1, 1], 1)
        grid.states[0, 0] = [maps.UNKNOWN, maps.FREE,
                              maps.OCCUPIED, maps.BUFFER]
        self.assertEqual(grid.states.ravel().tolist(), [0, 1, 2, 3])
        self.assertEqual(grid.state_at([2.5, 0.5, 0.5]), maps.OCCUPIED)
        with self.assertRaises(ValueError):
            grid.index([4, 0, 0])

    def test_defined_wall_and_hollow_tube(self):
        grid = maps.VoxelGrid([0, 0, 0], [2, 2, 1], 20)
        grid.add_shape({'type': 'wall', 'position_min': [0.1, 0.1, 0],
                        'position_max': [0.2, 0.3, 1]})
        self.assertEqual(grid.state_at([0.15, 0.2, 0.5]), maps.OCCUPIED)
        grid.add_shape({'type': 'tube', 'center': [1, 1, 0.5],
                        'axis': [0, 0, 1], 'length': 0.5,
                        'inner_radius': 0.2, 'outer_radius': 0.35})
        self.assertEqual(grid.state_at([1, 1, 0.5]), maps.FREE)
        self.assertEqual(grid.state_at([1.3, 1, 0.5]), maps.OCCUPIED)

    def test_seeded_generation_and_save_load(self):
        config = {'position_min': [0, 0, -1], 'position_max': [2, 3, 0],
                  'resolution_per_meter': 10,
                  'random': {'seed': 7,
                             'maze': {'wall_count': 3, 'thickness': 0.1,
                                      'gap_width': 0.4},
                             'spheres': {'count': 5, 'radius_min': 0.1,
                                         'radius_max': 0.2}}}
        grid = maps.generate(config)
        self.assertGreater(np.count_nonzero(grid.states == maps.OCCUPIED), 0)
        np.testing.assert_array_equal(grid.states, maps.generate(config).states)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data' / 'maps' / 'map.npz'
            grid.save(path)
            self.assertTrue(path.is_file())
            restored = maps.VoxelGrid.load(path)
            np.testing.assert_array_equal(restored.states, grid.states)
            self.assertEqual(restored.frame_id, grid.frame_id)
            np.testing.assert_array_equal(restored.position_min, grid.position_min)
            with np.load(path) as saved:
                self.assertEqual(int(saved['format_version']), 2)
                self.assertEqual(saved['data'].size, grid.states.size)

    def test_buffer_expands_obstacles_by_configured_distance(self):
        config = {'position_min': [0, 0, 0],
                  'position_max': [0.5, 0.5, 0.5],
                  'resolution_per_meter': 10,
                  'buffer_distance': 0.11,
                  'obstacles': [{'type': 'sphere',
                                 'center': [0.25, 0.25, 0.25],
                                 'radius': 0.001}]}
        grid = maps.generate(config)
        self.assertEqual(grid.state_at([0.25, 0.25, 0.25]), maps.OCCUPIED)
        self.assertEqual(grid.state_at([0.35, 0.25, 0.25]), maps.BUFFER)
        self.assertEqual(grid.state_at([0.35, 0.35, 0.25]), maps.FREE)
        self.assertEqual(grid.state_at([0.45, 0.25, 0.25]), maps.FREE)
        config['buffer_distance'] = 0.0
        self.assertEqual(maps.generate(config).state_at([0.35, 0.25, 0.25]),
                         maps.FREE)
        config['buffer_distance'] = -0.1
        with self.assertRaisesRegex(ValueError, 'buffer_distance'):
            maps.generate(config)

    def test_reserved_codes_and_legacy_map(self):
        grid = maps.VoxelGrid([0, 0, 0], [4, 1, 1], 1,
                              default_state=maps.OCCUPIED)
        self.assertEqual(grid.state_at([0.5, 0.5, 0.5]), maps.OCCUPIED)
        grid.add_shape({'type': 'wall', 'position_min': [1, 0, 0],
                        'position_max': [2, 1, 1], 'state': maps.FREE})
        self.assertEqual(grid.state_at([1.5, 0.5, 0.5]), maps.FREE)
        with self.assertRaises(ValueError):
            maps.state_code(4)
        grid.states[0, 0, 0] = 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.npz'
            with self.assertRaisesRegex(ValueError, 'reserved'):
                grid.save(path)
            np.savez_compressed(path, position_min=[0, 0, 0],
                                position_max=[4, 1, 1],
                                resolution_per_meter=1, frame_id='map',
                                data=np.array([0xE4], dtype=np.uint8))
            loaded = maps.VoxelGrid.load(path)
            self.assertEqual(loaded.states.ravel().tolist(), [0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()
