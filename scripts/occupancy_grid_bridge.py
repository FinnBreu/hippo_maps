#!/usr/bin/env python3
"""Publish an occupancy map as Foxglove voxels and a bounds wireframe."""

import argparse

import numpy as np
import rclpy
from hippo_maps.msg import OccupancyGrid3D
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from occupancy_grid import (BUFFER, OCCUPIED, bounds_edges, load_colors,
                            unpack_message)


def to_voxel_grid(message, colors=None, visible_states=None):
    """Expand the planning grid into Foxglove's native cell format."""
    from foxglove_msgs.msg import PackedElementField, VoxelGrid

    lower, _, resolution, states = unpack_message(message)
    nz, ny, nx = states.shape
    cells = np.zeros(states.size, dtype=[('state', 'u1'), ('red', 'u1'),
                                         ('green', 'u1'), ('blue', 'u1'),
                                         ('alpha', 'u1')])
    flat = states.ravel()
    cells['state'] = flat
    if colors is None:
        colors = load_colors()
    for state, (rgb, alpha, _) in colors.items():
        mask = flat == state
        cells['red'][mask] = rgb[0]
        cells['green'][mask] = rgb[1]
        cells['blue'][mask] = rgb[2]
        cells['alpha'][mask] = alpha
    if visible_states is not None:
        cells['alpha'][~np.isin(flat, visible_states)] = 0

    grid = VoxelGrid()
    grid.timestamp = message.header.stamp
    grid.frame_id = message.header.frame_id
    grid.pose.position.x, grid.pose.position.y, grid.pose.position.z = lower
    grid.pose.orientation.w = 1.0
    grid.row_count = ny
    grid.column_count = nx
    grid.cell_size.x = grid.cell_size.y = grid.cell_size.z = 1.0 / resolution
    grid.cell_stride = cells.dtype.itemsize
    grid.row_stride = nx * grid.cell_stride
    grid.slice_stride = ny * grid.row_stride
    grid.fields = [PackedElementField(name=name, offset=offset,
                                      type=PackedElementField.UINT8)
                   for offset, name in enumerate(
                       ('state', 'red', 'green', 'blue', 'alpha'))]
    grid.data = cells.tobytes()
    return grid


def to_bounds_scene(message):
    """Draw the exact position bounds as twelve independent line segments."""
    from foxglove_msgs.msg import Color, LinePrimitive, SceneEntity, SceneUpdate
    from geometry_msgs.msg import Point

    lower, upper, _, _ = unpack_message(message)
    points = [Point(x=float(value[0]), y=float(value[1]), z=float(value[2]))
              for edge in bounds_edges(lower, upper) for value in edge]
    line = LinePrimitive()
    line.type = LinePrimitive.LINE_LIST
    line.pose.orientation.w = 1.0
    line.thickness = 2.0
    line.scale_invariant = True
    line.color = Color(r=0.0, g=0.75, b=0.85, a=1.0)
    line.points = points
    entity = SceneEntity()
    entity.timestamp = message.header.stamp
    entity.frame_id = message.header.frame_id
    entity.id = 'position_bounds'
    entity.lines = [line]
    return SceneUpdate(entities=[entity])


class OccupancyGridBridge(Node):
    def __init__(self, input_topic, output_topic, buffer_topic, bounds_topic):
        super().__init__('occupancy_grid_bridge')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        from foxglove_msgs.msg import SceneUpdate, VoxelGrid
        self.publisher = self.create_publisher(VoxelGrid, output_topic, qos)
        self.buffer_publisher = self.create_publisher(VoxelGrid, buffer_topic, qos)
        self.bounds_publisher = self.create_publisher(SceneUpdate, bounds_topic,
                                                      qos)
        self.subscription = self.create_subscription(
            OccupancyGrid3D, input_topic, self.on_grid, qos)

    def on_grid(self, message):
        try:
            colors = load_colors()
            grid = to_voxel_grid(message, colors, (OCCUPIED,))
            buffer_grid = to_voxel_grid(message, colors, (BUFFER,))
            bounds_scene = to_bounds_scene(message)
        except (OSError, ValueError) as error:
            self.get_logger().error(f'Invalid occupancy grid: {error}')
            return
        # The source map may use wall time while this viewer uses /clock.
        # Use the bridge's clock so Foxglove can resolve the map frame now.
        timestamp = self.get_clock().now().to_msg()
        grid.timestamp = buffer_grid.timestamp = timestamp
        bounds_scene.entities[0].timestamp = timestamp
        self.publisher.publish(grid)
        self.buffer_publisher.publish(buffer_grid)
        self.bounds_publisher.publish(bounds_scene)
        self.get_logger().info(f'Published {len(grid.data) // grid.cell_stride} voxels in each display layer.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-topic', default='/occupancy_grid_3d')
    parser.add_argument('--output-topic', default='/pool/voxel_grid')
    parser.add_argument('--buffer-topic', default='/pool/buffer_voxel_grid')
    parser.add_argument('--bounds-topic', default='/pool/position_bounds')
    args, ros_args = parser.parse_known_args()
    try:
        import foxglove_msgs.msg  # noqa: F401
    except ImportError:
        parser.error('foxglove_msgs is required; install ros-jazzy-foxglove-msgs')
    rclpy.init(args=ros_args)
    node = OccupancyGridBridge(args.input_topic, args.output_topic,
                               args.buffer_topic, args.bounds_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
