import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
                             'maze': {'method': 'depth_first',
                                      'cell_size': 0.75,
                                      'thickness': 0.1},
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

    def test_depth_first_leaves_one_wall_in_two_by_two_cells(self):
        grid = maps.VoxelGrid([0, 0, 0], [2, 2, 1], 20)
        shapes = []
        grid.add_shape = shapes.append
        maps.generate_maze(grid, {'method': 'depth_first', 'cell_size': 1.0,
                                  'thickness': 0.05},
                           np.random.default_rng(21))
        self.assertEqual(len(shapes), 1)
        shape = shapes[0]
        self.assertEqual(shape['position_min'][2], 0)
        self.assertEqual(shape['position_max'][2], 1)
        self.assertAlmostEqual(shape['position_max'][0] -
                               shape['position_min'][0] +
                               shape['position_max'][1] -
                               shape['position_min'][1], 1.05)

    def test_depth_first_keeps_buffered_free_space_connected(self):
        from scipy.ndimage import label

        for seed in range(10):
            with self.subTest(seed=seed):
                grid = maps.VoxelGrid([0, 0, 0], [6, 8, 1], 20)
                shapes = []
                add_shape = grid.add_shape

                def record(shape):
                    shapes.append(shape)
                    add_shape(shape)

                grid.add_shape = record
                maps.generate_maze(grid, {'cell_size': 1.0,
                                          'thickness': 0.05},
                                   np.random.default_rng(seed))
                # 6x8 cells have 82 internal edges; DFS opens 47 of them.
                self.assertEqual(len(shapes), 35)
                grid.add_buffer(0.17)
                _, components = label(grid.states[10] == maps.FREE,
                                      structure=[[0, 1, 0],
                                                 [1, 1, 1],
                                                 [0, 1, 0]])
                self.assertEqual(components, 1)

    def test_depth_first_varies_with_seed(self):
        layouts = set()
        for seed in range(10):
            grid = maps.VoxelGrid([0, 0, 0], [6, 8, 1], 20)
            maps.generate_maze(grid, {'method': 'depth_first',
                                      'cell_size': 1.0,
                                      'thickness': 0.05},
                               np.random.default_rng(seed))
            layouts.add(grid.states.tobytes())
        self.assertEqual(len(layouts), 10)

    def test_depth_first_rejects_cells_thinner_than_walls(self):
        grid = maps.VoxelGrid([0, 0, 0], [2, 2, 1], 20)
        with self.assertRaisesRegex(ValueError, 'maze.thickness'):
            maps.generate_maze(grid, {'method': 'depth_first',
                                      'cell_size': 0.1, 'thickness': 0.1},
                               np.random.default_rng(0))
        with self.assertRaisesRegex(ValueError, 'maze.method'):
            maps.generate_maze(grid, {'method': 'recursive_division'},
                               np.random.default_rng(0))

    def test_baffles_leave_clear_connected_passages_and_vary_with_seed(self):
        from scipy.ndimage import label

        options = {'method': 'baffles', 'wall_count': 4,
                   'thickness': 0.03, 'passage_width': 0.9,
                   'min_spacing': 0.8, 'end_margin': 0.3,
                   'branch_count': 1}
        layouts = set()
        branch_seen = False
        for seed in range(10):
            grid = maps.VoxelGrid([0, 0, 0], [2, 8, 1], 20)
            shapes = []
            add_shape = grid.add_shape

            def record(shape):
                shapes.append(shape)
                add_shape(shape)

            grid.add_shape = record
            maps.generate_maze(grid, options, np.random.default_rng(seed))
            self.assertIn(len(shapes), (4, 5))
            branch_seen |= len(shapes) == 5
            cuts = [shape['position_min'][1] for shape in shapes[:4]]
            self.assertGreaterEqual(cuts[0], 1.2)
            self.assertLessEqual(shapes[3]['position_max'][1], 6.8)
            self.assertTrue(all(b - a >= 0.8 for a, b in
                                zip(cuts, cuts[1:])))
            for shape in shapes[:4]:
                self.assertAlmostEqual(shape['position_max'][0] -
                                       shape['position_min'][0], 1.1)
            layouts.add(tuple((round(float(cut), 2),
                               shape['position_min'][0] == 0)
                              for cut, shape in zip(cuts, shapes[:4])))
            grid.add_buffer(0.17)
            _, components = label(grid.states[10] == maps.FREE,
                                  structure=[[0, 1, 0],
                                             [1, 1, 1],
                                             [0, 1, 0]])
            self.assertEqual(components, 1)
        self.assertEqual(len(layouts), 10)
        self.assertTrue(branch_seen)

        grid = maps.VoxelGrid([0, 0, 0], [2, 8, 1], 20)
        shapes = []
        grid.add_shape = shapes.append
        options['wall_count'] = 100
        options['branch_count'] = 0
        maps.generate_maze(grid, options, np.random.default_rng(5))
        self.assertEqual(len(shapes), 7)

    def test_baffles_can_float_horizontally_at_full_height(self):
        from scipy.ndimage import label

        grid = maps.VoxelGrid([0, 0, 0], [2, 4, 1.5], 20)
        shapes = []
        add_shape = grid.add_shape

        def record(shape):
            shapes.append(shape)
            add_shape(shape)

        grid.add_shape = record
        maps.generate_maze(grid, {
            'method': 'baffles', 'wall_count': 3, 'thickness': 0.05,
            'passage_width': 0.6, 'min_spacing': 0.5,
            'end_margin': 0.1, 'floating_count': 2,
            'floating_clearance': 0.55,
        }, np.random.default_rng(8))
        self.assertEqual(len(shapes), 3)
        floating = [shape for shape in shapes
                    if shape['position_min'][0] > 0 and
                    shape['position_max'][0] < 2]
        self.assertEqual(len(floating), 2)
        for shape in floating:
            self.assertGreaterEqual(shape['position_min'][0], 0.55)
            self.assertLessEqual(shape['position_max'][0], 1.45)
            self.assertGreaterEqual(shape['position_min'][1], 0.75)
            self.assertLessEqual(shape['position_max'][1], 3.25)
            self.assertEqual(shape['position_min'][2], 0)
            self.assertEqual(shape['position_max'][2], 1.5)
        grid.add_buffer(0.17)
        for shape in floating:
            x = (shape['position_min'][0] + shape['position_max'][0]) / 2
            y = (shape['position_min'][1] + shape['position_max'][1]) / 2
            self.assertEqual(grid.state_at([x, y, 0.025]), maps.OCCUPIED)
            self.assertEqual(grid.state_at([x, y, 1.475]), maps.OCCUPIED)
        _, components = label(grid.states[15] == maps.FREE,
                              structure=[[0, 1, 0],
                                         [1, 1, 1],
                                         [0, 1, 0]])
        self.assertEqual(components, 1)

    def test_branches_use_end_gaps_without_reaching_boundary(self):
        from scipy.ndimage import label

        grid = maps.VoxelGrid([0, 0, 0], [2, 4, 1], 20)
        shapes = []
        add_shape = grid.add_shape

        def record(shape):
            shapes.append(shape)
            add_shape(shape)

        grid.add_shape = record
        options = {'method': 'baffles', 'wall_count': 3,
                   'thickness': 0.05, 'passage_width': 0.8,
                   'min_spacing': 0.6, 'end_margin': 0.3,
                   'branch_count': 3, 'floating_count': 1,
                   'floating_clearance': 0.6}
        maze_seed = np.random.SeedSequence(17).spawn(2)[0]
        maps.generate_maze(grid, options, np.random.default_rng(maze_seed))
        self.assertEqual(len(shapes), 6)
        main, branches = shapes[:3], shapes[3:]
        self.assertTrue(any(branch['position_max'][1] <=
                            main[0]['position_min'][1]
                            for branch in branches))
        self.assertTrue(any(branch['position_min'][1] >=
                            main[-1]['position_max'][1]
                            for branch in branches))
        for branch in branches:
            self.assertGreaterEqual(branch['position_min'][1], 0.6)
            self.assertLessEqual(branch['position_max'][1], 3.4)
        self.assertTrue(any(branch['position_min'][1] < 0.9
                            for branch in branches))
        grid.add_buffer(0.17)
        _, components = label(grid.states[10] == maps.FREE,
                              structure=[[0, 1, 0],
                                         [1, 1, 1],
                                         [0, 1, 0]])
        self.assertEqual(components, 1)

    def test_sphere_positions_do_not_depend_on_maze_wall_count(self):
        config = {'position_min': [0, 0, 0], 'position_max': [2, 4, 1],
                  'resolution_per_meter': 20,
                  'random': {'seed': 21,
                             'maze': {'method': 'baffles', 'wall_count': 3,
                                      'thickness': 0.03,
                                      'passage_width': 0.9,
                                      'min_spacing': 0.5},
                             'spheres': {'count': 4, 'radius_min': 0.05,
                                         'radius_max': 0.15}}}
        original = maps.VoxelGrid.add_shape
        placed = []

        def record(grid, shape):
            if shape['type'] == 'sphere':
                placed.append((tuple(shape['center']), shape['radius']))
            original(grid, shape)

        with patch.object(maps.VoxelGrid, 'add_shape', record):
            maps.generate(config)
            first = placed.copy()
            placed.clear()
            config['random']['maze']['wall_count'] = 4
            maps.generate(config)
            self.assertEqual(placed, first)

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
