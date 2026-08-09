import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import anchor_layout
import utils

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate UWB TWR distance measurements in parallel."
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (defaults to the available CPUs)",
    )
    return parser.parse_args()


def evaluate_grid_point(
    task: tuple[int, int, float, float, np.ndarray, int, float, int],
) -> tuple[int, int, float, float]:
    idx_y, idx_x, x, y, anchors, num_trials, transmitter_z, seed = task
    transmitter_position = np.array([x, y, transmitter_z])
    true_distances = utils.get_true_distance(anchors, transmitter_position)
    initial_guess = np.mean(anchors, axis=0).astype(float)
    rng = np.random.default_rng(seed)
    trial_errors = []

    start_time = time.perf_counter()
    for _ in range(num_trials):
        twr_measurements = true_distances + rng.normal(0, 0.1, len(anchors))
        estimated_position = utils.estimate_transmitter_position_scipy_2d(
            anchors, twr_measurements, initial_guess, transmitter_z
        )
        trial_errors.append(np.linalg.norm(estimated_position - transmitter_position))
        initial_guess = estimated_position

    elapsed = time.perf_counter() - start_time
    return idx_y, idx_x, float(np.median(trial_errors)), elapsed


def main() -> None:
    args = parse_args()

    anchor_coordinates = None
    if args.anchors.strip():
        anchor_coordinates = utils.parse_anchor_coordinates(args.anchors)
    elif args.anchor_layout:
        anchor_coordinates = anchor_layout.get_preset_anchor_layout(
            args.anchor_layout, args.room_x, args.room_y, args.room_z
        )
    anchors = anchor_coordinates

    increment = 0.5
    transmitter_z = 0.5
    x_range = np.arange(0, args.room_x + increment, increment)
    y_range = np.arange(0, args.room_y + increment, increment)
    x_count = len(x_range)
    y_count = len(y_range)
    total_points = x_count * y_count
    total_runs = total_points * args.num_trials

    print(f"Generating grid: {x_count}x{y_count} ({total_points} evaluation points)")
    print(f"Total solves: {total_runs} ({args.num_trials} trials/point)")
    error_grid = np.zeros((y_count, x_count))
    time_grid = np.zeros((y_count, x_count))
    seed_sequence = np.random.SeedSequence()
    seeds = seed_sequence.generate_state(total_points, dtype=np.uint64)
    start_time = time.perf_counter()
    tasks = [
        (
            idx_y,
            idx_x,
            x,
            y,
            anchors,
            args.num_trials,
            transmitter_z,
            int(seeds[idx_y * x_count + idx_x]),
        )
        for idx_y, y in enumerate(y_range)
        for idx_x, x in enumerate(x_range)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for completed, result in enumerate(executor.map(evaluate_grid_point, tasks), 1):
            idx_y, idx_x, median_error, elapsed = result
            error_grid[idx_y, idx_x] = median_error
            time_grid[idx_y, idx_x] = elapsed
            # print(f"Completed grid point {completed}/{total_points}")
    total_elapsed = time.perf_counter() - start_time

    print(f"Room Dimensions: X: {args.room_x} m Y: {args.room_y} m Z: {args.room_z} m")
    np.set_printoptions(precision=4, suppress=True)
    print("Anchor Coordinates:")
    for i, anchor in enumerate(anchors):
        print(f"Anchor {i + 1}: {anchor}")

    print("\nNumber of TWR Measurements:", args.num_trials)
    print(f"Total time: {np.sum(time_grid)}s")
    print(f"Total elapsed time for all grid points: {total_elapsed:.2f}s")

    print(
        f"Minimum Error: {np.min(error_grid):.2f}m Minimum Time: {np.min(time_grid):.4f}s"
    )
    print(
        f"Maximum Error: {np.max(error_grid):.2f}m Maximum Time: {np.max(time_grid):.4f}s"
    )
    print(
        f"Average Error: {np.mean(error_grid):.2f}m Average Time: {np.mean(time_grid):.4f}s"
    )
    print(
        f"Median Error: {np.median(error_grid):.2f}m Median Time: {np.median(time_grid):.4f}s"
    )

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
        "UWB Localization Error Heatmap (Parallel)\nRoom:"
        f" {args.room_x}m x {args.room_y}m x {args.room_z}m | Transmitter Height : 0.5m | Trials:{args.num_trials}",
        fontsize=13,
    )
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.tight_layout()

    filename = "/tmp/uwb_heatmap_parallel.png"
    if args.anchor_layout:
        filename = f"/tmp/uwb_heatmap_parallel_{args.anchor_layout}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Heatmap saved to {filename}")


if __name__ == "__main__":
    main()
