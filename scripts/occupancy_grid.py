"""Decode hippo_maps' 3-D occupancy grid for visualization."""

import numpy as np


UNKNOWN, FREE, OCCUPIED, BUFFER = range(4)
STATE_NAMES = ('unknown', 'free', 'occupied', 'buffer')
MAX_VOXELS = 50_000_000


def load_colors(path=None):
    """Read display colors shared by the Foxglove bridge and local viewer."""
    from pathlib import Path
    import yaml

    if path is None:
        from ament_index_python.packages import get_package_share_directory
        path = Path(get_package_share_directory('hippo_maps')) / \
            'config' / 'voxel_colors.yaml'
    try:
        with open(path, encoding='utf-8') as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError(f'invalid voxel color YAML: {error}') from error
    if not isinstance(config, dict) or set(config) != set(STATE_NAMES):
        raise ValueError('voxel colors must define unknown, free, occupied, and buffer')
    colors = {}
    for state, name in enumerate(STATE_NAMES):
        entry = config[name]
        if not isinstance(entry, dict) or \
                set(entry) != {'rgb', 'foxglove_alpha', 'viewer_alpha'}:
            raise ValueError(f'invalid color definition for {name}')
        rgb = entry['rgb']
        alpha = entry['foxglove_alpha']
        viewer_alpha = entry['viewer_alpha']
        if not isinstance(rgb, list) or len(rgb) != 3 or \
                any(isinstance(v, bool) or not isinstance(v, int) or
                    not 0 <= v <= 255 for v in rgb) or \
                isinstance(alpha, bool) or not isinstance(alpha, int) or \
                not 0 <= alpha <= 255 or \
                isinstance(viewer_alpha, bool) or \
                not isinstance(viewer_alpha, (int, float)) or \
                not np.isfinite(viewer_alpha) or \
                not 0 <= viewer_alpha <= 1:
            raise ValueError(f'invalid color values for {name}')
        colors[state] = (tuple(rgb), alpha, float(viewer_alpha))
    return colors


def unpack(position_min, position_max, resolution_per_meter, data,
           format_version=2):
    lower = np.asarray(position_min, dtype=np.float64)
    upper = np.asarray(position_max, dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,) or \
            not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or \
            np.any(upper <= lower):
        raise ValueError('invalid grid bounds')
    if not isinstance(resolution_per_meter, (int, np.integer)) or \
            resolution_per_meter <= 0:
        raise ValueError('resolution_per_meter must be positive')
    dimensions = np.ceil((upper - lower) * resolution_per_meter).astype(np.int64)
    count = int(np.prod(dimensions))
    if count > MAX_VOXELS:
        raise ValueError('grid exceeds 50 million voxels')
    values = np.asarray(data, dtype=np.uint8)
    if format_version == 2:
        if values.ndim != 1 or values.size != count:
            raise ValueError('voxel data has the wrong length')
        states = values
    elif format_version == 1:
        if values.ndim != 1 or values.size != (count + 3) // 4:
            raise ValueError('packed voxel data has the wrong length')
        states = np.empty(count, dtype=np.uint8)
        for offset in range(4):
            states[offset::4] = (values[:states[offset::4].size] >>
                                 (2 * offset)) & 3
        if count % 4 and values[-1] >> (2 * (count % 4)):
            raise ValueError('unused bits in the final byte must be zero')
    else:
        raise ValueError(f'unsupported map format version: {format_version}')
    if np.any(states > BUFFER):
        raise ValueError('grid contains reserved voxel states')
    nx, ny, nz = dimensions
    return lower, upper, int(resolution_per_meter), states.reshape((nz, ny, nx))


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        format_version = int(saved['format_version']) if 'format_version' in saved \
            else 1
        lower, upper, resolution, states = unpack(
            saved['position_min'], saved['position_max'],
            int(saved['resolution_per_meter']), saved['data'], format_version)
        frame_id = str(saved['frame_id'])
    if not frame_id:
        raise ValueError('saved grid has an empty frame_id')
    return frame_id, lower, upper, resolution, states


def unpack_message(message):
    lower = [message.position_min.x, message.position_min.y,
             message.position_min.z]
    upper = [message.position_max.x, message.position_max.y,
             message.position_max.z]
    return unpack(lower, upper, message.resolution_per_meter, message.data)


def visible_voxels(lower, upper, resolution, states, show_unknown=False,
                   max_points=None):
    """Return voxel centres and state codes; x varies fastest."""
    flat = states.ravel()
    visible = (flat == OCCUPIED) | (flat == BUFFER)
    if show_unknown:
        visible |= flat == UNKNOWN
    indices = np.flatnonzero(visible)
    if max_points is not None and len(indices) > max_points:
        # Regular deterministic sampling avoids changing the view on reload.
        indices = indices[np.linspace(0, len(indices) - 1,
                                      max_points, dtype=np.int64)]
    nx = states.shape[2]
    ny = states.shape[1]
    x = indices % nx
    y = (indices // nx) % ny
    z = indices // (nx * ny)
    cells = np.column_stack((x, y, z))
    cell_lower = lower + cells / resolution
    cell_upper = np.minimum(cell_lower + 1.0 / resolution, upper)
    return (cell_lower + cell_upper) / 2.0, flat[indices]
