import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utils

matplotlib.use("Agg")


def plot_anchors_and_transmitter(
    anchors: np.ndarray,
    true_pos: np.ndarray,
    min_estimate: np.ndarray,
    max_estimate: np.ndarray,
    median_estimate: np.ndarray,
    avg_estimate: np.ndarray,
    output_file: str,
) -> None:
    """
    Plot the anchors, true transmitter position, and estimated positions in 3D.

    Args:
        anchors (np.ndarray): 3D coordinates of the anchors.
        true_pos (np.ndarray): True 3D position of the transmitter.
        min_estimate (np.ndarray): Minimum estimate position.
        max_estimate (np.ndarray): Maximum estimate position.
        median_estimate (np.ndarray): Median estimate position.
        avg_estimate (np.ndarray): Average estimate position.
        output_file (str): Path to save the output plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot anchors
    ax.scatter(
        anchors[:, 0],
        anchors[:, 1],
        anchors[:, 2],
        c="blue",
        marker="^",
        s=100,
        label="Anchors",
    )

    ax.scatter(
        min_estimate[0],
        min_estimate[1],
        min_estimate[2],
        c="green",
        marker="x",
        label="Minimum Estimate",
        s=120,
    )

    ax.scatter(
        max_estimate[0],
        max_estimate[1],
        max_estimate[2],
        c="orange",
        marker="d",
        label="Maximum Estimate",
        s=120,
    )
    ax.scatter(
        median_estimate[0],
        median_estimate[1],
        median_estimate[2],
        c="purple",
        marker="+",
        label="Median Estimate",
        s=120,
    )
    ax.scatter(
        avg_estimate[0],
        avg_estimate[1],
        avg_estimate[2],
        c="brown",
        marker="*",
        label="Average Estimate",
        s=120,
    )
    # Plot true transmitter position
    ax.scatter(
        true_pos[0],
        true_pos[1],
        true_pos[2],
        c="red",
        marker="o",
        s=20,
        label="True Position",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("UWB TWR Localization")
    ax.legend()
    plt.savefig(output_file)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate UWB TWR distance measurements."
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="/tmp/uwb.png",
        help="Output file name for the 3D image",
    )
    parser.add_argument(
        "--room_x", type=float, default=50.0, help="Room size in x-direction (meters)"
    )
    parser.add_argument(
        "--room_y", type=float, default=50.0, help="Room size in y-direction (meters)"
    )
    parser.add_argument(
        "--room_z", type=float, default=10.0, help="Room size in z-direction (meters)"
    )
    parser.add_argument(
        "--anchors",
        type=str,
        default="",
        help="JSON string containing anchor coordinates (e.g., '[[0,0,0], [10,0,0], [0,10,0], [10,10,0]]')",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="Number of TWR measurements to estimate median error",
    )
    args = parser.parse_args()

    # Set the room dimensions
    room_x = args.room_x
    room_y = args.room_y
    room_z = args.room_z

    anchor_coordinates = None
    if args.anchors.strip():
        anchor_coordinates = utils.parse_anchor_coordinates(args.anchors)

    anchors, transmitter_position, true_distances = utils.generate_data(
        room_x, room_y, room_z, anchor_coordinates
    )

    trial_errors = []
    estimated_positions = []
    for _ in range(args.num_trials):
        twr_measurements = utils.simulate_twr_measurements(
            anchors, transmitter_position
        )
        estimated_position = utils.estimate_transmitter_position_scipy(
            anchors, twr_measurements
        )
        error = np.linalg.norm(estimated_position - transmitter_position)
        trial_errors.append(error)
        estimated_positions.append(estimated_position)

    print(f"Room Dimensions: X: {room_x} m Y: {room_y} m Z: {room_z} m")
    if anchor_coordinates is not None:
        print(f"Using provided anchor coordinates: {anchor_coordinates}")
    else:
        print(f"\nGenerated {len(anchors)} anchors")
    np.set_printoptions(precision=4, suppress=True)
    print("Anchor Coordinates:")
    for i, anchor in enumerate(anchors):
        print(f"Anchor {i + 1}: {anchor}" f" | True Distance: {true_distances[i]:.4f}m")

    print("\nNumber of TWR Measurements:", args.num_trials)
    print("Transmitter Position:", transmitter_position)

    min_error = np.min(trial_errors)
    min_estimate = estimated_positions[np.argmin(trial_errors)]
    print(f"Minimum Error: {min_error:.4f}m" f" Position: {min_estimate}")

    max_error = np.max(trial_errors)
    max_estimate = estimated_positions[np.argmax(trial_errors)]
    print(f"Maximum Error: {max_error:.4f}m" f" Position: {max_estimate}")

    median_error = np.median(trial_errors)
    median_estimate = np.median(estimated_positions, axis=0)
    print(f"Median Error: {median_error:.4f}m" f" Position: {median_estimate}")

    avg_error = np.mean(trial_errors)
    avg_estimate = np.mean(estimated_positions, axis=0)
    print(f"Average Error: {avg_error:.4f}m" f" Position: {avg_estimate}")

    if args.output_plot:
        plot_anchors_and_transmitter(
            anchors,
            transmitter_position,
            min_estimate,
            max_estimate,
            median_estimate,
            avg_estimate,
            args.output_plot,
        )
        print(f"3D plot saved to {args.output_plot}")

    plt.hist(trial_errors, bins=20, color="blue", alpha=0.7)
    plt.title("Distribution of TWR Estimation Errors")
    plt.xlabel("Error (meters)")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.savefig("/tmp/uwb_error_histogram.png")
    print("Error histogram saved to /tmp/uwb_error_histogram.png")


if __name__ == "__main__":
    main()
