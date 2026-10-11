# hippo_maps

`hippo_maps/msg/OccupancyGrid3D` carries an axis-aligned grid in the header
frame. `position_min` and `position_max` are ordered world-space bounds, and
`resolution_per_meter` sets a uniform voxel size. For the space from
`(0, 0, 0)` to `(2, 4, -1.5)`, use bounds `(0, 0, -1.5)` and `(2, 4, 0)`.
Each axis has `ceil((max - min) * resolution_per_meter)` cells. The byte index
is `i = x + nx * (y + ny * z)`, with x varying fastest. The current codes are
`UNKNOWN` (0), `FREE` (1), and `OCCUPIED` (2). Values 3 through 255 are
reserved; generators and viewers reject them until their meaning is defined.

Generate a map from a JSON recipe. The [example recipe](config/occupancy_grid_example.json)
defines walls and a hollow tube, plus seeded maze walls and random spheres.
Shapes and `default_state` accept the names `unknown`, `free`, and `occupied`,
or the integers 0, 1, and 2.

```sh
ros2 run hippo_maps generate_occupancy_grid \
  --config src/hippo_maps/config/occupancy_grid_example.json \
  --output data/maps/pool_01.npz
ros2 run hippo_maps generate_occupancy_grid \
  --load data/maps/pool_01.npz --publish
```

New `.npz` files contain one byte per voxel and a format version. Older
packed `.npz` maps remain readable and can be resaved in the new format.
`--publish` sends `hippo_maps/msg/OccupancyGrid3D` on `/occupancy_grid_3d`
with reliable, transient-local QoS and keeps the publisher alive for late
subscribers.

For a movable local 3D view, run:

```sh
ros2 run hippo_maps view_occupancy_grid data/maps/pool_01.npz
```

The Foxglove launcher in `tools` starts `hippo_maps`' occupancy grid bridge.
It publishes `foxglove_msgs/msg/VoxelGrid` on `/pool/voxel_grid`. Both the
bridge and local viewer read [voxel_colors.yaml](config/voxel_colors.yaml):
each state has an RGB color, a Foxglove alpha byte, and a local viewer alpha
fraction. The current palette makes free cells transparent in Foxglove;
`--show-unknown` displays unknown cells in the local viewer. Enable the topic
in Foxglove's 3D panel with **RGBA (separate fields)**. The bridge timestamps
the Foxglove message with its own clock, matching the visualization session's
wall or simulation time. It rereads the color file for each map update.
