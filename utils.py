import numpy as np
import scipy.optimize
import json


def parse_anchor_coordinates(anchor_str: str) -> np.ndarray:
    """
    Parse anchor coordinates from a JSON string.
    Example: '[[0,0,0], [10,0,0], [0,10,0], [10,10,0]]'

    Args:
        anchor_str (str): JSON string containing anchor coordinates.

    Returns:
        np.ndarray: 3D array of anchor coordinates.

    Raises:
        ValueError: If the JSON string is invalid or contains fewer than 3 anchors.
    """
    anchor_str = anchor_str.strip()
    try:
        anchor_data = json.loads(anchor_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string for anchor coordinates: {e}")

    if not isinstance(anchor_data, list):
        raise ValueError("Anchor coordinates must be a list of 3D points.")

    anchors = []
    for coord in anchor_data:
        if len(coord) != 3:
            raise ValueError("Each anchor coordinate must be a 3D point.")
        anchors.append(coord)

    if len(anchors) < 3:
        raise ValueError("At least 3 anchors are required.")

    return np.array(anchors, dtype=float)


def generate_data(
    room_x: float,
    room_y: float,
    room_z: float,
    anchor_coordinates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random 3D anchor coordinates, transmitter position, true distances
    and TWR distances.

    Args:
        room_x (float): Size of the room in the x-direction (meters).
        room_y (float): Size of the room in the y-direction (meters).
        room_z (float): Size of the room in the z-direction (meters).
        anchor_coordinates (np.ndarray | None): Optional array of anchor coordinates.
            If provided, these coordinates will be used instead of generating random ones.
    Returns:
        tuple: A tuple of (anchors, transmitter_position, true_distances).
    """

    if anchor_coordinates is not None:
        anchors = anchor_coordinates
        num_anchors = len(anchors)
    else:
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

    true_distances = np.linalg.norm(anchors - transmitter_position, axis=1)

    return anchors, transmitter_position, true_distances


def simulate_twr_measurements(
    anchors: np.ndarray, transmitter_position: np.ndarray, noise_std: float = 0.1
) -> np.ndarray:
    """
    Simulate TWR measurements with Gaussian noise.

    Args:
        anchors (np.ndarray): 3D coordinates of the anchors.
        transmitter_position (np.ndarray): 3D position of the transmitter.
        noise_std (float): Standard deviation of the Gaussian noise (default: 0.1 meters).

    Returns:
        np.ndarray: Simulated TWR measurements with noise.
    """
    true_distances = np.linalg.norm(anchors - transmitter_position, axis=1)
    twr_measurements = true_distances + np.random.normal(0, noise_std, len(anchors))
    return twr_measurements


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
