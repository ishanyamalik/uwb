import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

matplotlib.use("Agg")


def generate_data(
    room_x: float, room_y: float, room_z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random 3D anchor coordinates, transmitter position, true distances
    and TWR distances.

    Args:
        room_x (float): Size of the room in the x-direction (meters).
        room_y (float): Size of the room in the y-direction (meters).
        room_z (float): Size of the room in the z-direction (meters).
    Returns:
        tuple: A tuple of (anchors, transmitter_position, true_distances,
        twr_measurements).
    """

    # Randomly choose between 4 to 10 anchors
    num_anchors = np.random.randint(4, 11)

    # Generate random 3D coordinates for anchors within the specified room dimensions
    anchors = np.zeros((num_anchors, 3))
    anchors[:, 0] = np.random.uniform(0, room_x, num_anchors)  # x-coordinates
    anchors[:, 1] = np.random.uniform(0, room_y, num_anchors)  # y-coordinates
    anchors[:, 2] = np.random.uniform(0, room_z, num_anchors)  # z-coordinates

    # Generate a random true position for the transmitter within the same space
    transmitter_position = np.array(
        [
            np.random.uniform(0, room_x - 1),
            np.random.uniform(0, room_y - 1),
            np.random.uniform(0, room_z - 1),
        ]
    )

    # Calculate true TWR distances with small Gaussian noise (std dev of 10cm)
    # to simulate measurement errors
    noise_std = 0.1
    true_distances = np.linalg.norm(anchors - transmitter_position, axis=1)
    twr_measurements = true_distances + np.random.normal(0, noise_std, num_anchors)

    return anchors, transmitter_position, true_distances, twr_measurements


def estimate_transmitter_position_gn(
    anchors: np.ndarray, twr_measurements: np.ndarray
) -> np.ndarray:
    """
    Estimate the transmitter position using Gauss-Newton least squares
    optimization.

    Args:
        anchors (np.ndarray): 3D coordinates of the anchors.
        twr_measurements (np.ndarray): Measured distances from anchors to
        transmitter.

    Returns:
        np.ndarray: Estimated 3D position of the transmitter.
    """

    # Initial guess: centroid of anchors
    pos = np.mean(anchors, axis=0).astype(float)

    for _ in range(40):  # Iterate to refine the estimate
        diff = pos - anchors
        distances = np.linalg.norm(diff, axis=1)
        # Avoid division by zero
        distances = np.where(distances == 0, 1e-6, distances)
        residuals = distances - twr_measurements
        if np.max(np.abs(residuals)) < 1e-6:  # Convergence check
            break
        jacobian = diff / distances[:, np.newaxis]
        delta, _, _, _ = np.linalg.lstsq(jacobian, -residuals, rcond=None)
        pos += delta  # Update position estimate

    return pos


def estimate_transmitter_position_lm(
    anchors: np.ndarray, twr_measurements: np.ndarray
) -> np.ndarray:
    """
    Estimate the transmitter position using Levenberg-Marquardt algorithm.

    Args:
        anchors (np.ndarray): 3D coordinates of the anchors.
        twr_measurements (np.ndarray): Measured distances from anchors to transmitter.

    Returns:
        np.ndarray: Estimated 3D position of the transmitter.
    """
    pos = np.mean(anchors, axis=0).astype(float)  # Initial guess: centroid of anchors
    damping_lambda = 0.01

    for _ in range(100):  # Iterate to refine the estimate
        diff = pos - anchors
        distances = np.linalg.norm(diff, axis=1)
        distances = np.where(distances == 0, 1e-6, distances)  # Avoid division by zero
        residuals = distances - twr_measurements
        cost = 0.5 * np.sum(residuals**2)

        if np.max(np.abs(residuals)) < 1e-9:  # Convergence check
            break

        jacobian = diff / distances[:, np.newaxis]
        normal_matrix = jacobian.T @ jacobian
        gradient = jacobian.T @ residuals

        if np.linalg.norm(gradient) < 1e-9:  # Gradient convergence check
            break

        diag_scaling = np.eye(3) * np.where(
            np.diag(normal_matrix) > 0, np.diag(normal_matrix), 1.0
        )
        damped_matrix = normal_matrix + damping_lambda * diag_scaling
        try:
            delta = np.linalg.solve(damped_matrix, -gradient)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(damped_matrix) @ (-gradient)

        if np.linalg.norm(delta) < 1e-9:  # Step size convergence check
            break

        candidate_pos = pos + delta
        candidate_diff = candidate_pos - anchors
        candidate_distances = np.linalg.norm(candidate_diff, axis=1)
        candidate_residuals = candidate_distances - twr_measurements
        candidate_cost = 0.5 * np.sum(candidate_residuals**2)

        if candidate_cost < cost:  # Accept the candidate position
            pos = candidate_pos
            damping_lambda = max(damping_lambda / 10.0, 1e-12)
        else:  # Reject the candidate position and increase damping
            damping_lambda = 10.0
    return pos


def localization_residuals(
    transmitter_guess: np.ndarray, anchors: np.ndarray, twr_measurements: np.ndarray
) -> np.ndarray:
    calculated_distance = np.linalg.norm(anchors - transmitter_guess, axis=1)
    return calculated_distance - twr_measurements


def estimate_transmitter_position_scipy(
    anchors: np.ndarray, twr_measurements: np.ndarray
) -> np.ndarray:
    initial_guess = np.mean(anchors, axis=0).astype(float)
    result = scipy.optimize.least_squares(
        localization_residuals,
        initial_guess,
        args=(anchors, twr_measurements),
        method="lm",
    )
    return result.x


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
    args = parser.parse_args()

    # Set the room dimensions
    room_x = args.room_x
    room_y = args.room_y
    room_z = args.room_z

    anchors, transmitter_position, true_distances, twr_measurements = generate_data(
        room_x, room_y, room_z
    )

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

    estimated_position = estimate_transmitter_position_gn(anchors, twr_measurements)
    print("Gauss-Newton Estimate:", estimated_position)

    estimated_position = estimate_transmitter_position_lm(anchors, twr_measurements)
    print("Levenberg-Marquardt Estimate:", estimated_position)

    estimated_position = estimate_transmitter_position_scipy(anchors, twr_measurements)
    print("Scipy LM Estimate:", estimated_position)

    if args.output_plot:
        plot_anchors_and_transmitter(
            anchors,
            transmitter_position,
            estimate_transmitter_position_gn(anchors, twr_measurements),
            estimate_transmitter_position_lm(anchors, twr_measurements),
            estimate_transmitter_position_scipy(anchors, twr_measurements),
            args.output_plot,
        )
        print(f"3D plot saved to {args.output_plot}")


if __name__ == "__main__":
    main()
