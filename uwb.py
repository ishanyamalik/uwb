import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utils

matplotlib.use("Agg")


def plot_anchors_and_transmitter(
    anchors: np.ndarray,
    true_pos: np.ndarray,
    estimated_pos_gn: np.ndarray,
    estimated_pos_lm: np.ndarray,
    estimated_pos_scipy: np.ndarray,
    output_file: str,
) -> None:
    """
    Plot the anchors, true transmitter position, and estimated positions in 3D.

    Args:
        anchors (np.ndarray): 3D coordinates of the anchors.
        true_pos (np.ndarray): True 3D position of the transmitter.
        estimated_pos_gn (np.ndarray): Estimated position using Gauss-Newton.
        estimated_pos_lm (np.ndarray): Estimated position using Levenberg-Marquardt.
        estimated_pos_scipy (np.ndarray): Estimated position using Scipy LM.
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

    # Plot true transmitter position
    ax.scatter(
        true_pos[0],
        true_pos[1],
        true_pos[2],
        c="green",
        marker="o",
        s=120,
        label="True Position",
    )

    # Plot estimated positions
    ax.scatter(
        estimated_pos_gn[0],
        estimated_pos_gn[1],
        estimated_pos_gn[2],
        c="red",
        marker="x",
        label="Gauss-Newton Estimate",
        s=120,
    )
    ax.scatter(
        estimated_pos_lm[0],
        estimated_pos_lm[1],
        estimated_pos_lm[2],
        c="orange",
        marker="+",
        label="Levenberg-Marquardt Estimate",
        s=120,
    )
    ax.scatter(
        estimated_pos_scipy[0],
        estimated_pos_scipy[1],
        estimated_pos_scipy[2],
        c="purple",
        marker="d",
        label="Scipy LM Estimate",
        s=120,
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
    args = parser.parse_args()

    # Set the room dimensions
    room_x = args.room_x
    room_y = args.room_y
    room_z = args.room_z

    anchor_coordinates = None
    if args.anchors.strip():
        anchor_coordinates = utils.parse_anchor_coordinates(args.anchors)
    anchors, transmitter_position, true_distances, twr_measurements = (
        utils.generate_data(room_x, room_y, room_z, anchor_coordinates)
    )

    print(f"Room Dimensions: X: {room_x} m Y: {room_y} m Z: {room_z} m")
    if anchor_coordinates is not None:
        print(f"Using provided anchor coordinates: {anchor_coordinates}")
    else:
        print(f"\nGenerated {len(anchors)} anchors")
    np.set_printoptions(precision=4, suppress=True)
    print("Anchor Coordinates:")
    for i, anchor in enumerate(anchors):
        print(
            f"Anchor {i + 1}: {anchor}"
            f" | True Distance: {true_distances[i]:.4f}m"
            f" | TWR Distance: {twr_measurements[i]:.4f}m"
        )

    print("\nTransmitter Position:", transmitter_position)

    gn_position = utils.estimate_transmitter_position_gn(anchors, twr_measurements)
    print("Gauss-Newton Estimate:", gn_position)

    lm_position = utils.estimate_transmitter_position_lm(anchors, twr_measurements)
    print("Levenberg-Marquardt Estimate:", lm_position)

    scipy_position = utils.estimate_transmitter_position_scipy(
        anchors, twr_measurements
    )
    print("SciPy LM Estimate:", scipy_position)

    if args.output_plot:
        plot_anchors_and_transmitter(
            anchors,
            transmitter_position,
            gn_position,
            lm_position,
            scipy_position,
            args.output_plot,
        )
        print(f"3D plot saved to {args.output_plot}")


if __name__ == "__main__":
    main()
