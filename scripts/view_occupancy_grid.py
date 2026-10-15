#!/usr/bin/env python3
"""Interactive 3-D view of a saved occupancy grid (drag to rotate)."""

import argparse

from occupancy_grid import (BUFFER, OCCUPIED, UNKNOWN, bounds_edges,
                            load_colors, load_npz, visible_voxels)


def make_figure(path, show_unknown=False, max_points=40000):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    frame, lower, upper, resolution, states = load_npz(path)
    colors = load_colors()

    def display_color(state):
        rgb, _, alpha = colors[state]
        return (*[component / 255.0 for component in rgb], alpha)

    centres, codes = visible_voxels(lower, upper, resolution, states,
                                    show_unknown, max_points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if len(centres):
        ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2],
                   c=[display_color(code) for code in codes],
                   marker='s', s=12, depthshade=False, rasterized=True)
    for start, end in bounds_edges(lower, upper):
        ax.plot((start[0], end[0]), (start[1], end[1]),
                (start[2], end[2]), color='#00bcd4', linewidth=1.4)
    ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)',
           title=f'{path}  [{frame}]  {len(centres)} visible voxels')
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(upper - lower)
    legend = [Line2D([0], [0], color=display_color(OCCUPIED),
                     marker='s', linestyle='', label='Occupied'),
              Line2D([0], [0], color=display_color(BUFFER),
                     marker='s', linestyle='', label='Buffer')]
    if show_unknown:
        legend.append(Line2D([0], [0], color=display_color(UNKNOWN), marker='s',
                             linestyle='', label='Unknown'))
    ax.legend(handles=legend)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map', help='saved .npz map')
    parser.add_argument('--show-unknown', action='store_true')
    parser.add_argument('--max-points', type=int, default=40000)
    parser.add_argument('--save-preview', help='write a PNG instead of opening a window')
    args = parser.parse_args()
    if args.max_points <= 0:
        parser.error('--max-points must be positive')
    if args.save_preview:
        import matplotlib
        matplotlib.use('Agg')
    fig = make_figure(args.map, args.show_unknown, args.max_points)
    if args.save_preview:
        fig.savefig(args.save_preview, dpi=150)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()
