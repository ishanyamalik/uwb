import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import anchor_layout
import utils


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
        "--transmitter_z",
        type=float,
        default=0.5,
        help="Transmitter height (meters)",
    )
    parser.add_argument(
        "--grid_increment",
        type=float,
        default=0.5,
        help="Grid increment for x and y (meters)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (defaults to the available CPUs)",
    )
    return parser.parse_args()


def evaluate_grid_point(

    # Function to evaluate a single grid point for UWB localization error

    # Parameters:
    idx_y: int,
    idx_x: int,
    x: float,
    y: float,
    anchors: np.ndarray,
    num_trials: int,
    transmitter_z: float,
) -> tuple[int, int, float, float]:

    # Calculate the true distances from the anchors to the transmitter position
    transmitter_position = np.array([x, y, transmitter_z])
    true_distances = utils.get_true_distance(anchors, transmitter_position)

    # Set the initial guess for the transmitter position as the mean of the anchor coordinates
    initial_guess = np.mean(anchors, axis=0).astype(float)
    trial_errors = []

    # Perform multiple trials to estimate the median localization error at this grid point
    start_time = time.perf_counter()
    for _ in range(num_trials):
        twr_measurements = true_distances + np.random.normal(0, 0.1, len(anchors))
        estimated_position = utils.estimate_transmitter_position_scipy_2d(
            anchors, twr_measurements, initial_guess, transmitter_z
        )
        trial_errors.append(np.linalg.norm(estimated_position - transmitter_position))
        initial_guess = estimated_position

    # Calculate the elapsed time for all trials at this grid point
    elapsed = time.perf_counter() - start_time
    return idx_y, idx_x, float(np.median(trial_errors)), elapsed


def main() -> None:
    # Parse command-line arguments
    args = parse_args()

    # Initialize anchor coordinates based on user input or preset layout
    if args.anchors.strip():
        anchors = utils.parse_anchor_coordinates(args.anchors)
    else:
        anchors = anchor_layout.get_preset_anchor_layout(
            args.anchor_layout, args.room_x, args.room_y, args.room_z
        )

    # Initialize increment and transmitter height
    increment = args.grid_increment
    transmitter_z = args.transmitter_z

    # Initialize x and y ranges based on room dimensions and increment
    x_range = np.arange(0, args.room_x + increment, increment)
    y_range = np.arange(0, args.room_y + increment, increment)

    # Calculate the number of x and y intervals
    x_count = len(x_range)
    y_count = len(y_range)

    # Calculate total points and total runs for the grid
    total_points = x_count * y_count
    total_runs = total_points * args.num_trials
    print(f"Generating grid: {x_count}x{y_count} ({total_points} evaluation points)")
    print(f"Total solves: {total_runs} ({args.num_trials} trials/point)")

    # Initialize 2D arrays to store median errors and elapsed times for each grid point
    error_grid = np.zeros(shape=(y_count, x_count))
    time_grid = np.zeros(shape=(y_count, x_count))

    # Tasks is an array of tuples containing the grid point indices and coordinates for parallel processing
    tasks = []

    # Populate the tasks list with grid point indices and coordinates
    for idx_y, y in enumerate(y_range):
        for idx_x, x in enumerate(x_range):
            tasks.append((idx_y, idx_x, x, y))

    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:

        # Submit tasks to the executor for parallel processing
        futures = [
            executor.submit(
                evaluate_grid_point,
                idx_y,
                idx_x,
                x,
                y,
                anchors,
                args.num_trials,
                transmitter_z,
            )
            for idx_y, idx_x, x, y in tasks
        ]

        # Process the completed futures as they finish and update the error and time grids
        for completed, future in enumerate(as_completed(futures), 1):
            idx_y, idx_x, median_error, elapsed = future.result()
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
    print(f"Grid Increment: {args.grid_increment} meters")
    print(f"Transmitter Height: {transmitter_z} meters")

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

    visual_grid = np.clip(error_grid, 0, 0.5)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        visual_grid,
        cmap="RdYlGn_r",
        xticklabels=np.round(x_range, 1),
        yticklabels=np.round(y_range, 1),
        cbar_kws={"label": "Median Localization Error (meters)"},
        vmin=0.05,
        vmax=0.15,
    )
    x_ticks_step = max(1, x_count // 10)
    y_ticks_step = max(1, y_count // 10)
    ax.set_xticks(np.arange(0, x_count, x_ticks_step))
    ax.set_yticks(np.arange(0, y_count, y_ticks_step))
    ax.set_xticklabels(np.round(x_range[::x_ticks_step], 1))
    ax.set_yticklabels(np.round(y_range[::y_ticks_step], 1))
    ax.invert_yaxis()
    plt.title(
        "UWB Localization Error Heatmap (submit)\nRoom:"
        f" {args.room_x}m x {args.room_y}m x {args.room_z}m | "
        f"Transmitter Height: {transmitter_z}m | Trials: {args.num_trials}",
        fontsize=13,
    )
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.tight_layout()

    current_file = Path(__file__).stem
    filename = f"/tmp/{current_file}.png"
    if args.anchor_layout:
        filename = f"/tmp/{current_file}_{args.anchor_layout}_{args.num_trials}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Heatmap saved to {filename}")


if __name__ == "__main__":
    main()
