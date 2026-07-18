import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def generate_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random 3D anchor coordinates, transmitter position and TWR distances.

    Returns:
        tuple: A tuple of (anchors, transmitter_position, true_distances, twr_measurements).
    """

    # Randomly choose between 4 to 10 anchors
    num_anchors = np.random.randint(4, 11)  
    
    # Generate random 3D coordinates for anchors within 50x50x10 meter space
    anchors = np.zeros((num_anchors, 3))
    anchors[:, 0] = np.random.uniform(0, 50, num_anchors)  # x-coordinates
    anchors[:, 1] = np.random.uniform(0, 50, num_anchors)  # y-coordinates
    anchors[:, 2] = np.random.uniform(0, 10, num_anchors)  # z-coordinates

    # Generate a random true position for the transmitter within the same space
    transmitter_position = np.array([
        np.random.uniform(0, 49),
        np.random.uniform(0, 49),
        np.random.uniform(0, 2)])

    # Calculate true TWR distances with small Gaussian noise (std dev of 10cm) 
    # to simulate measurement errors
    noise_std = 0.1
    true_distances = np.linalg.norm(anchors - transmitter_position, axis=1)
    twr_measurements = true_distances + np.random.normal(0, noise_std, num_anchors)

    return anchors, transmitter_position, true_distances, twr_measurements

def estimate_transmitter_position_gn(anchors: np.ndarray, twr_measurements: np.ndarray) -> np.ndarray:
    """
    Estimate the transmitter position using Gauss-Newton least squares optimization.

    Args:
        anchors (np.ndarray): 3D coordinates of the anchors.
        twr_measurements (np.ndarray): Measured distances from anchors to transmitter.

    Returns:
        np.ndarray: Estimated 3D position of the transmitter.
    """
    
    # Initial guess: centroid of anchors
    pos = np.mean(anchors, axis=0).astype(float) 

    for _ in range(40):  # Iterate to refine the estimate
        diff = pos - anchors
        distances = np.linalg.norm(diff, axis=1)
        distances = np.where(distances == 0, 1e-6, distances)  # Avoid division by zero
        residuals = distances - twr_measurements
        if np.max(np.abs(residuals)) < 1e-6:  # Convergence check
            break
        jacobian = diff / distances[:, np.newaxis]
        delta, _, _, _ = np.linalg.lstsq(jacobian, -residuals, rcond=None)
        pos += delta  # Update position estimate

    return pos    

def estimate_transmitter_position_lm(anchors: np.ndarray, twr_measurements: np.ndarray) -> np.ndarray:
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

        if (np.linalg.norm(gradient) < 1e-9):  # Gradient convergence check
            break

        diag_scaling = np.eye(3) * np.where(np.diag(normal_matrix) > 0, np.diag(normal_matrix), 1.0)
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate UWB TWR distance measurements.")
    parser.add_argument("--output_plot", type=str, default="/tmp/uwb.png", help="Output file name for the 3D image")
    args = parser.parse_args()

    anchors, transmitter_position, true_distances, twr_measurements = generate_data()

    print("Anchor Coordinates:\n", anchors)
    print("Transmitter Position:\n", transmitter_position)
    print("True TWR Distances:\n", true_distances)
    print("TWR Measurements:\n", twr_measurements)

    estimated_position = estimate_transmitter_position_gn(anchors, twr_measurements)
    print("Estimated Transmitter Position (Gauss-Newton):\n", estimated_position)

    estimated_position = estimate_transmitter_position_lm(anchors, twr_measurements)
    print("Estimated Transmitter Position (Levenberg-Marquardt):\n", estimated_position)

if __name__ == "__main__":
    main()