# Axis-aligned voxel grid in header.frame_id, sampled at header.stamp.
std_msgs/Header header

# Componentwise ordered world-space bounds, in metres
geometry_msgs/Point position_min
geometry_msgs/Point position_max
# Positive, uniform number of voxels per metre on all three axes.
uint32 resolution_per_meter

# Byte state codes. Values 3..255 are reserved for future use.
uint8 UNKNOWN=0
uint8 FREE=1
uint8 OCCUPIED=2

# One state byte per voxel.
# Derive nx, ny, nz by taking ceil((position_max - position_min) *
# resolution_per_meter) on each axis. For a position p inside the bounds,
# (x, y, z) = floor((p - position_min) * resolution_per_meter).
# For zero-based (x, y, z): i = x + nx * (y + ny * z), state = data[i].
# Length must be nx * ny * nz bytes. Cells at the upper edges are clipped
# by position_max.
uint8[] data
