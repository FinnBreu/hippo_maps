#!/usr/bin/env python3
"""Generate, save, load, and optionally publish a 3-D occupancy grid."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


UNKNOWN, FREE, OCCUPIED = range(3)
STATE_NAMES = {'unknown': UNKNOWN, 'free': FREE, 'occupied': OCCUPIED}
MAX_VOXELS = 50_000_000
FORMAT_VERSION = 2


def state_code(value):
    if isinstance(value, str) and value in STATE_NAMES:
        return STATE_NAMES[value]
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool) \
            and 0 <= value <= OCCUPIED:
        return int(value)
    raise ValueError('state must be unknown, free, occupied, or an integer from 0 to 2')


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

    def save(self, path):
        if np.any(self.states > OCCUPIED):
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
            if np.any(flat > OCCUPIED):
                raise ValueError('saved map contains reserved voxel states')
            return grid


def generate_maze(grid, options, rng):
    count = integer(options.get('wall_count', 0), 'maze.wall_count', True)
    thickness = positive(options.get('thickness', 0.1), 'maze.thickness')
    gap_width = positive(options.get('gap_width', 0.4), 'maze.gap_width')
    regions = [(grid.position_min[0], grid.position_max[0],
                grid.position_min[1], grid.position_max[1])]
    for _ in range(count):
        def split_axes(region):
            width = region[1] - region[0]
            height = region[3] - region[2]
            x_ok = width > 2 * thickness + gap_width and \
                height > gap_width + 2 * thickness
            y_ok = height > 2 * thickness + gap_width and \
                width > gap_width + 2 * thickness
            return x_ok, y_ok

        eligible = [r for r in regions if any(split_axes(r))]
        if not eligible:
            raise ValueError('maze bounds cannot fit the requested wall_count')
        region = max(eligible, key=lambda r: (r[1]-r[0]) * (r[3]-r[2]))
        regions.remove(region)
        x0, x1, y0, y1 = region
        x_ok, y_ok = split_axes(region)
        split_x = x_ok and (not y_ok or x1 - x0 >= y1 - y0)
        a0, a1 = (x0, x1) if split_x else (y0, y1)
        b0, b1 = (y0, y1) if split_x else (x0, x1)
        cut = rng.uniform(a0 + thickness, a1 - 2 * thickness)
        gap = rng.uniform(b0 + thickness, b1 - gap_width - thickness)
        for lo, hi in ((b0, gap), (gap + gap_width, b1)):
            if split_x:
                lower, upper = [cut, lo, grid.position_min[2]], \
                               [cut + thickness, hi, grid.position_max[2]]
            else:
                lower, upper = [lo, cut, grid.position_min[2]], \
                               [hi, cut + thickness, grid.position_max[2]]
            if hi > lo:
                grid.add_shape({'type': 'wall', 'position_min': lower,
                                'position_max': upper})
        if split_x:
            regions.extend([(x0, cut, y0, y1),
                            (cut + thickness, x1, y0, y1)])
        else:
            regions.extend([(x0, x1, y0, cut),
                            (x0, x1, cut + thickness, y1)])


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
        rng = np.random.default_rng(random_options.get('seed'))
        if 'maze' in random_options:
            generate_maze(grid, random_options['maze'], rng)
        if 'spheres' in random_options:
            generate_spheres(grid, random_options['spheres'], rng)
    return grid


def publish(grid, topic):
    if np.any(grid.states > OCCUPIED):
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
