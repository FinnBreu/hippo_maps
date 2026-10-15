#!/usr/bin/env python3
"""Generate, save, load, and optionally publish a 3-D occupancy grid."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


UNKNOWN, FREE, OCCUPIED, BUFFER = range(4)
STATE_NAMES = {'unknown': UNKNOWN, 'free': FREE, 'occupied': OCCUPIED,
               'buffer': BUFFER}
MAX_VOXELS = 50_000_000
FORMAT_VERSION = 2


def state_code(value):
    if isinstance(value, str) and value in STATE_NAMES:
        return STATE_NAMES[value]
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool) \
            and 0 <= value <= BUFFER:
        return int(value)
    raise ValueError('state must be unknown, free, occupied, buffer, or an integer from 0 to 3')


def vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be three finite numbers')
    return result


def positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'{name} must be finite and positive')
    return value


def integer(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or \
            value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'{name} must be a positive integer')
    return value


class VoxelGrid:
    def __init__(self, position_min, position_max, resolution_per_meter,
                 frame_id='map', default_state='free'):
        self.position_min = vector(position_min, 'position_min')
        self.position_max = vector(position_max, 'position_max')
        if np.any(self.position_max <= self.position_min):
            raise ValueError('position_max must exceed position_min on every axis')
        self.resolution_per_meter = integer(resolution_per_meter,
                                            'resolution_per_meter')
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError('frame_id must be a nonempty string')
        self.frame_id = frame_id
        default_state = state_code(default_state)
        # Array order is z, y, x; C-order flattening makes x vary fastest.
        nx, ny, nz = np.ceil((self.position_max - self.position_min) *
                             self.resolution_per_meter).astype(np.int64)
        if int(nx) * int(ny) * int(nz) > MAX_VOXELS:
            raise ValueError('grid exceeds 50 million voxels')
        self.states = np.full((nz, ny, nx), default_state, dtype=np.uint8)

    @property
    def dimensions(self):
        nz, ny, nx = self.states.shape
        return nx, ny, nz

    def index(self, position):
        p = vector(position, 'position')
        if np.any(p < self.position_min) or np.any(p >= self.position_max):
            raise ValueError('position is outside the grid bounds')
        return tuple(np.floor((p - self.position_min) *
                              self.resolution_per_meter).astype(int))

    def state_at(self, position):
        x, y, z = self.index(position)
        return int(self.states[z, y, x])

    def _region(self, lower, upper):
        """Candidate cells whose closed boxes can touch a shape AABB."""
        low = np.maximum(np.floor((lower - self.position_min) *
                                  self.resolution_per_meter).astype(int) - 1, 0)
        high = np.minimum(np.ceil((upper - self.position_min) *
                                   self.resolution_per_meter).astype(int) + 1,
                          np.array(self.dimensions) - 1)
        if np.any(high < low):
            return None
        x, y, z = np.meshgrid(np.arange(low[0], high[0] + 1),
                              np.arange(low[1], high[1] + 1),
                              np.arange(low[2], high[2] + 1), indexing='ij')
        center = self.position_min[:, None, None, None] + \
            (np.stack((x, y, z)) + 0.5) / self.resolution_per_meter
        return x, y, z, center

    def add_shape(self, shape):
        kind = shape.get('type')
        state_name = shape.get('state', 'occupied')
        state = state_code(state_name)
        half = 0.5 / self.resolution_per_meter
        if kind == 'wall':
            lower = vector(shape['position_min'], 'wall.position_min')
            upper = vector(shape['position_max'], 'wall.position_max')
            if np.any(upper <= lower):
                raise ValueError('wall bounds must be ordered')
            # AABB against AABB: touched cells form a rectangular slice.
            first = np.maximum(np.ceil((lower - self.position_min) *
                                       self.resolution_per_meter).astype(int) - 1, 0)
            last = np.minimum(np.floor((upper - self.position_min) *
                                       self.resolution_per_meter).astype(int),
                              np.array(self.dimensions) - 1)
            if np.all(last >= first):
                self.states[first[2]:last[2] + 1,
                            first[1]:last[1] + 1,
                            first[0]:last[0] + 1] = state
            return
        elif kind == 'sphere':
            center = vector(shape['center'], 'sphere.center')
            radius = positive(shape['radius'], 'sphere.radius')
            lower, upper = center - radius, center + radius
        elif kind == 'tube':
            center = vector(shape['center'], 'tube.center')
            axis = vector(shape['axis'], 'tube.axis')
            magnitude = np.linalg.norm(axis)
            if magnitude == 0:
                raise ValueError('tube.axis must be nonzero')
            axis /= magnitude
            length = positive(shape['length'], 'tube.length')
            inner = positive(shape['inner_radius'], 'tube.inner_radius', True)
            outer = positive(shape['outer_radius'], 'tube.outer_radius')
            if inner >= outer:
                raise ValueError('tube.inner_radius must be less than outer_radius')
            extent = np.abs(axis) * length / 2 + outer
            lower, upper = center - extent, center + extent
        else:
            raise ValueError(f'unsupported obstacle type: {kind}')
        region = self._region(lower, upper)
        if region is None:
            return
        x, y, z, voxel_center = region
        if kind == 'sphere':
            distance = np.maximum(np.abs(voxel_center - center[:, None, None, None])
                                  - half, 0)
            hit = np.sum(distance * distance, axis=0) <= radius * radius
        else:
            delta = voxel_center - center[:, None, None, None]
            axial = np.einsum('i,ijkl->jkl', axis, delta)
            radial = np.sqrt(np.maximum(np.sum(delta * delta, axis=0) -
                                        axial * axial, 0))
            half_diagonal = math.sqrt(3) * half
            hit = (np.abs(axial) <= length / 2 + half_diagonal) & \
                  (radial <= outer + half_diagonal) & \
                  (radial + half_diagonal >= inner)
        self.states[z[hit], y[hit], x[hit]] = state

    def add_buffer(self, distance):
        """Mark free voxel centres within distance metres of occupied centres."""
        distance = positive(distance, 'buffer_distance', allow_zero=True)
        if distance == 0 or not np.any(self.states == OCCUPIED):
            return
        from scipy.ndimage import distance_transform_edt
        distance_to_occupied = distance_transform_edt(
            self.states != OCCUPIED,
            sampling=1.0 / self.resolution_per_meter)
        self.states[(self.states == FREE) &
                    (distance_to_occupied <= distance)] = BUFFER

    def save(self, path):
        if np.any(self.states > BUFFER):
            raise ValueError('grid contains reserved voxel states')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as stream:
            np.savez_compressed(stream, position_min=self.position_min,
                                position_max=self.position_max,
                                resolution_per_meter=self.resolution_per_meter,
                                frame_id=np.array(self.frame_id),
                                format_version=FORMAT_VERSION,
                                data=self.states.ravel())

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            grid = cls(saved['position_min'], saved['position_max'],
                       int(saved['resolution_per_meter']), str(saved['frame_id']))
            data = np.asarray(saved['data'], dtype=np.uint8)
            version = int(saved['format_version']) if 'format_version' in saved \
                else 1
            flat = grid.states.ravel()
            if version == FORMAT_VERSION:
                if data.ndim != 1 or data.size != flat.size:
                    raise ValueError('saved map has the wrong data length')
                flat[:] = data
            elif version == 1:
                if data.ndim != 1 or data.size != (flat.size + 3) // 4:
                    raise ValueError('saved map has the wrong packed data length')
                for offset in range(4):
                    flat[offset::4] = (data[:flat[offset::4].size] >>
                                       (2 * offset)) & 3
                if flat.size % 4 and data[-1] >> (2 * (flat.size % 4)):
                    raise ValueError('saved map has nonzero unused bits')
            else:
                raise ValueError(f'unsupported map format version: {version}')
            if np.any(flat > BUFFER):
                raise ValueError('saved map contains reserved voxel states')
            return grid


def generate_depth_first(grid, options, rng):
    """Carve a randomized depth-first spanning tree through XY cells."""
    cell_size = positive(options.get('cell_size', 1.0), 'maze.cell_size')
    thickness = positive(options.get('thickness', 0.03), 'maze.thickness')
    span = grid.position_max[:2] - grid.position_min[:2]
    columns, rows = np.maximum(1, np.floor(span / cell_size).astype(int))
    pitch_x, pitch_y = span / [columns, rows]
    if thickness >= min(pitch_x, pitch_y):
        raise ValueError('maze.thickness must be smaller than each maze cell')

    visited = np.zeros((rows, columns), dtype=bool)
    open_x = np.zeros((rows, max(columns - 1, 0)), dtype=bool)
    open_y = np.zeros((max(rows - 1, 0), columns), dtype=bool)
    start = (int(rng.integers(columns)), int(rng.integers(rows)))
    visited[start[1], start[0]] = True
    stack = [start]
    while stack:
        x, y = stack[-1]
        neighbors = [(nx, ny) for nx, ny in
                     ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                     if 0 <= nx < columns and 0 <= ny < rows and
                     not visited[ny, nx]]
        if not neighbors:
            stack.pop()
            continue
        nx, ny = neighbors[int(rng.integers(len(neighbors)))]
        if nx != x:
            open_x[y, min(x, nx)] = True
        else:
            open_y[min(y, ny), x] = True
        visited[ny, nx] = True
        stack.append((nx, ny))

    x0, y0, z0 = grid.position_min
    z1 = grid.position_max[2]
    for y in range(rows):
        for x in range(columns - 1):
            if not open_x[y, x]:
                edge = x0 + (x + 1) * pitch_x
                lower = [edge - thickness / 2, y0 + y * pitch_y, z0]
                upper = [edge + thickness / 2, y0 + (y + 1) * pitch_y, z1]
                grid.add_shape({'type': 'wall', 'position_min': lower,
                                'position_max': upper})
    for y in range(rows - 1):
        for x in range(columns):
            if not open_y[y, x]:
                edge = y0 + (y + 1) * pitch_y
                lower = [x0 + x * pitch_x, edge - thickness / 2, z0]
                upper = [x0 + (x + 1) * pitch_x, edge + thickness / 2, z1]
                grid.add_shape({'type': 'wall', 'position_min': lower,
                                'position_max': upper})


def generate_baffles(grid, options, rng):
    """Place spaced crosswise walls with passages on alternating sides."""
    count = integer(options.get('wall_count', 0), 'maze.wall_count', True)
    thickness = positive(options.get('thickness', 0.03), 'maze.thickness')
    passage_width = positive(options.get('passage_width', 0.9),
                             'maze.passage_width')
    min_spacing = positive(options.get('min_spacing', 0.8),
                           'maze.min_spacing')
    if min_spacing <= thickness:
        raise ValueError('maze.min_spacing must exceed maze.thickness')
    end_margin = positive(options.get('end_margin', 0.3),
                          'maze.end_margin', True)
    branch_count = integer(options.get('branch_count', 0),
                           'maze.branch_count', True)
    floating_count = integer(options.get('floating_count', 0),
                             'maze.floating_count', True)
    floating_clearance = positive(options.get('floating_clearance', 0.55),
                                  'maze.floating_clearance')
    switch_probability = options.get('switch_probability', 0.7)
    if isinstance(switch_probability, bool) or not isinstance(
            switch_probability, (int, float)) or not math.isfinite(
            switch_probability) or not 0 <= switch_probability <= 1:
        raise ValueError('maze.switch_probability must be between 0 and 1')
    if count == 0:
        return

    size = grid.position_max - grid.position_min
    along = int(np.argmax(size[:2]))
    across = 1 - along
    span = size[along]
    width = size[across]
    boundary_clearance = max(end_margin, 1.5 * min_spacing)
    if width <= passage_width + thickness or \
            span < 2 * boundary_clearance + thickness:
        return
    capacity = int(math.floor((span - 2 * boundary_clearance - thickness) /
                              min_spacing)) + 1
    count = min(count, capacity)
    if width <= 2 * floating_clearance + thickness:
        floating_count = 0
    slack = span - 2 * boundary_clearance - thickness - \
        (count - 1) * min_spacing
    extra = rng.dirichlet(np.ones(count + 1)) * max(slack, 0)
    cuts = [grid.position_min[along] + boundary_clearance + extra[0]]
    for index in range(1, count):
        cuts.append(cuts[-1] + min_spacing + extra[index])
    floating_candidates = [index for index, cut in enumerate(cuts)
                           if cut >= grid.position_min[along] +
                           floating_clearance and cut + thickness <=
                           grid.position_max[along] - floating_clearance]
    floating_indices = set(rng.choice(
        floating_candidates,
        size=min(floating_count, len(floating_candidates)),
        replace=False).tolist())
    from_lower_side = bool(rng.integers(2))
    wall_bounds = []
    for index, cut in enumerate(cuts):
        lower = grid.position_min.copy()
        upper = grid.position_max.copy()
        lower[along] = cut
        upper[along] = cut + thickness
        if index in floating_indices:
            max_length = width - 2 * floating_clearance
            length = rng.uniform(0.65 * max_length, max_length)
            lower[across] = rng.uniform(
                grid.position_min[across] + floating_clearance,
                grid.position_max[across] - floating_clearance - length)
            upper[across] = lower[across] + length
        elif from_lower_side:
            upper[across] = grid.position_max[across] - passage_width
        else:
            lower[across] = grid.position_min[across] + passage_width
        grid.add_shape({'type': 'wall', 'position_min': lower,
                        'position_max': upper})
        wall_bounds.append((lower, upper))
        if index + 1 < count:
            if rng.random() < switch_probability:
                from_lower_side = not from_lower_side

    # Branches may enter the end gaps as well as spaces between walls. A long
    # wall can support two branches in one gap, separated across its length.
    gaps = [(grid.position_min[along], cuts[0], None, 0)]
    gaps.extend((wall_bounds[i][1][along], cuts[i + 1], i, i + 1)
                for i in range(count - 1))
    gaps.append((wall_bounds[-1][1][along], grid.position_max[along],
                 count - 1, None))
    min_branch_length = max(2 * thickness, 2.0 / grid.resolution_per_meter)
    gap_candidates = []
    for gap_start, gap_end, lower_index, upper_index in gaps:
        max_length = min(0.65, gap_end - gap_start - min_spacing)
        if max_length < min_branch_length:
            continue
        roots = []
        for index, direction in ((lower_index, 1), (upper_index, -1)):
            if index is not None and wall_bounds[index][1][across] - \
                    wall_bounds[index][0][across] >= 0.3:
                roots.append((index, direction))
        if roots:
            index, direction = roots[rng.integers(len(roots))]
            body_length = (wall_bounds[index][1][across] -
                           wall_bounds[index][0][across])
            fractions = (0.2, 0.8) if body_length >= 1.0 else (0.5,)
            gap_candidates.extend((max_length, index, direction, fraction)
                                  for fraction in fractions)
    for _ in range(min(branch_count, len(gap_candidates))):
        weights = np.array([candidate[0] for candidate in gap_candidates])
        choice = rng.choice(len(gap_candidates), p=weights / weights.sum())
        max_length, index, direction, fraction = gap_candidates.pop(choice)
        root_lower, root_upper = wall_bounds[index]
        branch_lower = root_lower.copy()
        branch_upper = root_upper.copy()
        body_start = root_lower[across]
        body_length = root_upper[across] - body_start
        branch_lower[across] = body_start + fraction * body_length
        branch_upper[across] = branch_lower[across] + thickness
        length = rng.uniform(max(min_branch_length, 0.65 * max_length),
                             max_length)
        if direction > 0:
            branch_lower[along] = root_upper[along]
            branch_upper[along] = root_upper[along] + length
        else:
            branch_upper[along] = root_lower[along]
            branch_lower[along] = root_lower[along] - length
        grid.add_shape({'type': 'wall', 'position_min': branch_lower,
                        'position_max': branch_upper})


def generate_maze(grid, options, rng):
    method = options.get('method', 'depth_first')
    if method == 'depth_first':
        generate_depth_first(grid, options, rng)
    elif method == 'baffles':
        generate_baffles(grid, options, rng)
    else:
        raise ValueError('maze.method must be depth_first or baffles')


def generate_spheres(grid, options, rng):
    count = integer(options.get('count', 0), 'spheres.count', True)
    radius_min = positive(options.get('radius_min', 0.1), 'spheres.radius_min')
    radius_max = positive(options.get('radius_max', 0.25), 'spheres.radius_max')
    if radius_max < radius_min:
        raise ValueError('spheres.radius_max must exceed radius_min')
    for _ in range(count):
        radius = rng.uniform(radius_min, radius_max)
        center = rng.uniform(grid.position_min, grid.position_max)
        grid.add_shape({'type': 'sphere', 'center': center,
                        'radius': radius})


def generate(config):
    grid = VoxelGrid(config['position_min'], config['position_max'],
                     config['resolution_per_meter'],
                     config.get('frame_id', 'map'),
                     config.get('default_state', 'free'))
    for shape in config.get('obstacles', []):
        grid.add_shape(shape)
    random_options = config.get('random', {})
    if random_options:
        maze_seed, sphere_seed = np.random.SeedSequence(
            random_options.get('seed')).spawn(2)
        if 'maze' in random_options:
            generate_maze(grid, random_options['maze'],
                          np.random.default_rng(maze_seed))
        if 'spheres' in random_options:
            generate_spheres(grid, random_options['spheres'],
                             np.random.default_rng(sphere_seed))
    grid.add_buffer(config.get('buffer_distance', 0.0))
    return grid


def publish(grid, topic):
    if np.any(grid.states > BUFFER):
        raise ValueError('grid contains reserved voxel states')
    import rclpy
    from hippo_maps.msg import OccupancyGrid3D
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    rclpy.init()
    node = rclpy.create_node('occupancy_grid_generator')
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
    publisher = node.create_publisher(OccupancyGrid3D, topic, qos)
    message = OccupancyGrid3D()
    message.header.frame_id = grid.frame_id
    message.header.stamp = node.get_clock().now().to_msg()
    message.position_min.x, message.position_min.y, message.position_min.z = \
        grid.position_min.tolist()
    message.position_max.x, message.position_max.y, message.position_max.z = \
        grid.position_max.tolist()
    message.resolution_per_meter = grid.resolution_per_meter
    message.data = grid.states.ravel().tolist()
    publisher.publish(message)
    node.get_logger().info(f'Published {grid.states.size} voxels on {topic}.')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', type=Path, help='JSON generation recipe')
    source.add_argument('--load', type=Path, help='saved .npz map')
    parser.add_argument('--output', type=Path, help='save generated map as .npz')
    parser.add_argument('--publish', action='store_true', help='publish until stopped')
    parser.add_argument('--topic', default='occupancy_grid_3d')
    args = parser.parse_args()
    if not args.output and not args.publish:
        parser.error('choose --output, --publish, or both')
    grid = VoxelGrid.load(args.load) if args.load else \
        generate(json.loads(args.config.read_text()))
    if args.output:
        grid.save(args.output)
        print(f'Saved {grid.states.size} voxels to {args.output}')
    if args.publish:
        publish(grid, args.topic)


if __name__ == '__main__':
    main()
