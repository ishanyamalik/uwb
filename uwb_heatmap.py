import argparse
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import anchor_layout
import utils

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate UWB TWR distance measurements."
    )
    parser.add_argument(
        "--room_x", type=float, default=10.0, help="Room size in x-direction (meters)"
    )
    parser.add_argument(
        "--room_y", type=float, default=10.0, help="Room size in y-direction (meters)"
    )
    parser.add_argument(
        "--room_z", type=float, default=10.0, help="Room size in z-direction (meters)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--anchors",
        type=str,
        default="",
        help="JSON string containing anchor coordinates (e.g., '[[0,0,0], [10,0,0], [0,10,0], [10,10,0]]')",
    )
    group.add_argument(
        "--anchor_layout",
        type=str,
        default="perimeter_corners_cuboid",
        choices=list(anchor_layout.PRESET_ANCHOR_LAYOUT_DESCRIPTIONS.keys()),
        help="Valid values="
        "perimeter_corners_cuboid|max_volume_4anchors|perimeter_staggered_heights|"
        "convex_polygon_center_elevated|equilateral_triangle_mesh|center_wall_ceiling|random",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="Number of TWR measurements to estimate median error",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set the room dimensions
    room_x = args.room_x
    room_y = args.room_y
    room_z = args.room_z

    anchor_coordinates = None
    if args.anchors.strip():
        anchor_coordinates = utils.parse_anchor_coordinates(args.anchors)
    elif args.anchor_layout:
        anchor_coordinates = anchor_layout.get_preset_anchor_layout(
            args.anchor_layout, room_x, room_y, room_z
        )
    anchors = anchor_coordinates

    increment = 0.5
    transmitter_z = 0.5
    x_range = np.arange(0, room_x + increment, increment)
    y_range = np.arange(0, room_y + increment, increment)
    x_count = len(x_range)
    y_count = len(y_range)
    total_points = x_count * y_count
    total_runs = total_points * args.num_trials

    print(f"Generating grid: {x_count}x{y_count} ({total_points} evaluation points)")
    print(f"Total solves: {total_runs} ({args.num_trials} trials/point)")
    error_grid = np.zeros((y_count, x_count))
    time_grid = np.zeros((y_count, x_count))
    initial_guess = np.mean(anchors, axis=0).astype(float)

    for idx_y, y in enumerate(y_range):
        row_pct = ((idx_y + 1) / y_count) * 100
        print(
            f"Simulating grid row {idx_y+1}/{y_count} (Y = {y:.1f}m)"
            f" [{row_pct:.1f}% complete]..."
        )
        for idx_x, x in enumerate(x_range):
            transmitter_position = np.array([x, y, transmitter_z])
            true_distances = utils.get_true_distance(anchors, transmitter_position)
            trial_errors = []

            start_time = time.perf_counter()
            for _ in range(args.num_trials):
                twr_measurements = true_distances + np.random.normal(
                    0, 0.1, len(anchors)
                )
                estimated_position = (
                    utils.estimate_transmitter_position_scipy_with_initial_guess(
                        anchors, twr_measurements, initial_guess
                    )
                )
                error = np.linalg.norm(estimated_position - transmitter_position)
                trial_errors.append(error)
                initial_guess = estimated_position
            time_grid[idx_y, idx_x] = time.perf_counter() - start_time
            # print(f"time[{idx_y}][{idx_x}] = {time_grid[idx_y, idx_x]}")
            error_grid[idx_y, idx_x] = float(np.median(trial_errors))

    print(f"Room Dimensions: X: {room_x} m Y: {room_y} m Z: {room_z} m")
    np.set_printoptions(precision=4, suppress=True)
    print("Anchor Coordinates:")
    for i, anchor in enumerate(anchors):
        print(f"Anchor {i + 1}: {anchor}")

    print("\nNumber of TWR Measurements:", args.num_trials)

    min_error = np.min(error_grid)
    max_error = np.max(error_grid)
    avg_error = np.mean(error_grid)
    median_error = np.median(error_grid)

    min_time = np.min(time_grid)
    max_time = np.max(time_grid)
    avg_time = np.mean(time_grid)
    median_time = np.median(time_grid)
    print(f"Total time: {np.sum(time_grid)}s")
    print(f"Minimum Error: {min_error:.2f}m Minimum Time: {min_time:.4f}s")
    print(f"Maximum Error: {max_error:.2f}m Maximum Time: {max_time:.4f}s")
    print(f"Average Error: {avg_error:.2f}m Average Time: {avg_time:.4f}s")
    print(f"Median Error: {median_error:.2f}m Median Time: {median_time:.4f}s")

    cap_value = 0.5
    visual_grid = np.clip(error_grid, 0, cap_value)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        visual_grid,
        cmap="RdYlGn_r",
        xticklabels=np.round(x_range, 1),
        yticklabels=np.round(y_range, 1),
        cbar_kws={"label": "Median Localization Error (meters)"},
    )
    x_ticks_step = max(1, x_count // 10)
    y_ticks_step = max(1, y_count // 10)
    ax.set_xticks(np.arange(0, x_count, x_ticks_step))
    ax.set_yticks(np.arange(0, y_count, y_ticks_step))
    ax.set_xticklabels(np.round(x_range[::x_ticks_step], 1))
    ax.set_yticklabels(np.round(y_range[::y_ticks_step], 1))

    plt.gca().invert_yaxis()

    plt.title(
        "UWB Localization Error Heatmap\nRoom:"
        f" {room_x}m x {room_y}m x {room_z}m | Transmitter Height : 0.5m | Trials:{args.num_trials}",
        fontsize=13,
    )
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.tight_layout()

    filename = "/tmp/uwb_heatmap.png"
    if args.anchor_layout:
        filename = f"/tmp/uwb_heatmap_{args.anchor_layout}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Heatmap saved to {filename}")


if __name__ == "__main__":
    main()
