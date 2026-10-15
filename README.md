# hippo_maps

`hippo_maps/msg/OccupancyGrid3D` carries an axis-aligned grid in the header
frame. `position_min` and `position_max` are ordered world-space bounds, and
`resolution_per_meter` sets a uniform voxel size. For the space from
`(0, 0, 0)` to `(2, 4, -1.5)`, use bounds `(0, 0, -1.5)` and `(2, 4, 0)`.
Each axis has `ceil((max - min) * resolution_per_meter)` cells. The byte index
is `i = x + nx * (y + ny * z)`, with x varying fastest. The current codes are
`UNKNOWN` (0), `FREE` (1), `OCCUPIED` (2), and `BUFFER` (3). Values 4 through
255 are reserved. Guidance can traverse only `FREE` voxels; buffer voxels block
planning like occupied voxels.

Generate a map from a JSON recipe. The [tank recipe](config/occupancy_grid_tank.json)
uses `maze.method: "baffles"` for its narrow XY area. It places spaced walls
across the short axis, leaving a passage at one end of each wall. The passage
side and wall spacing vary with the seed. `wall_count` is a target; the method
places as many walls as fit the requested `min_spacing`. A wall parallel to a
position boundary stays at least `1.5 * min_spacing` from that boundary;
`end_margin` can increase this clearance.
`passage_width` is the clear gap before voxelization and buffering, and
`thickness` sets wall thickness. `branch_count` adds short perpendicular
branches, including branches into the end gaps. Two branches may grow from a
long wall; each branch tip stays at least `min_spacing` from a parallel box
boundary. `floating_count` selects walls that stay
clear of the four horizontal position bounds while spanning the full map
height. `floating_clearance` controls clearance from the sides and ends; set
it beyond the buffer distance to keep the visible buffer clear of those sides.
Choose a
passage width larger than twice the buffer distance plus the desired free
clearance.

The [maze recipe](config/occupancy_grid_maze.json) uses
`maze.method: "depth_first"`. It carves a randomized depth-first path through
an XY cell lattice, leaving walls between cells that the path did not join.
`cell_size` sets the target cell pitch in metres; the map is divided into whole
cells of at least that size. `thickness` sets the full-height wall thickness.
The outer position bounds are not filled with walls. Both methods use a
separate random stream from spheres, so changing maze settings does not move
random spheres when the seed stays the same.
After changing the generator script, run
`colcon build --packages-select hippo_maps` from the workspace root before
using `ros2 run`; ROS executes the installed copy of the script.
Shapes and `default_state` accept the four state names or their integer codes.
Set `buffer_distance` in metres in the recipe (zero by default) to mark free
voxel centres within that Euclidean distance of an occupied voxel centre as
`BUFFER`. The buffer is applied after all defined and random obstacles are
placed. Unknown voxels remain unknown.

```sh
ros2 run hippo_maps generate_occupancy_grid \
  --config src/hippo_maps/config/occupancy_grid_tank.json \
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

The local viewer draws the twelve edges of the position bounds as cyan lines.

The Foxglove launcher in `tools` starts `hippo_maps`' occupancy grid bridge.
It publishes occupied cells as `foxglove_msgs/msg/VoxelGrid` on
`/pool/voxel_grid`, buffer cells on `/pool/buffer_voxel_grid`, and the position
bounds as a `foxglove_msgs/msg/SceneUpdate` wireframe on
`/pool/position_bounds`. Enable that topic in Foxglove's 3D panel to show the
twelve box edges. The bridge and local viewer read
[voxel_colors.yaml](config/voxel_colors.yaml):
each state has an RGB color, a Foxglove alpha byte, and a local viewer alpha
fraction. The buffer has a lighter color and lower alpha than occupied cells.
The current palette makes free cells transparent in Foxglove;
`--show-unknown` displays unknown cells in the local viewer. Enable both voxel
topics in Foxglove's 3D panel with **RGBA (separate fields)**, then set **Draw behind**
on `/pool/buffer_voxel_grid` so occupied cells remain visible through the buffer.
The bridge timestamps all three Foxglove messages with its own clock, matching the
visualization session's wall or simulation time. It rereads the color file for
each map update.
