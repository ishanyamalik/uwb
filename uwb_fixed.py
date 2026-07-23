import utils
import numpy as np


def generate_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random 3D anchor coordinates, transmitter position and TWR distances.

    Returns:
        tuple: A tuple of (anchors, transmitter_position, true_distances, twr_measurements).
    """

    num_anchors = 8

    # Generate random 3D coordinates for anchors within 50x50x10 meter space
    anchors = np.array(
        [
            [0, 0, 0],
            [50, 0, 0],
            [50, 50, 0],
            [0, 50, 0],
            [0, 0, 10],
            [50, 0, 10],
            [50, 50, 10],
            [0, 50, 10],
        ],
        dtype=float,
    )

    # Generate a random true position for the transmitter within the same space
    transmitter_position = np.array(
        [np.random.uniform(0, 49), np.random.uniform(0, 49), np.random.uniform(0, 2)]
    )

    # Calculate true TWR distances with small Gaussian noise (std dev of 10cm)
    # to simulate measurement errors
    noise_std = 0.1
    true_distances = np.linalg.norm(anchors - transmitter_position, axis=1)
    twr_measurements = true_distances + np.random.normal(0, noise_std, num_anchors)

    return anchors, transmitter_position, true_distances, twr_measurements


def main() -> None:
    anchors, transmitter_position, true_distances, twr_measurements = generate_data()

    print("Anchor Coordinates:\n", anchors)
    print("Transmitter Position:\n", transmitter_position)
    print("True TWR Distances:\n", true_distances)
    print("TWR Measurements:\n", twr_measurements)

    estimated_position = utils.estimate_transmitter_position_gn(
        anchors, twr_measurements
    )
    print("Estimated Transmitter Position (Gauss-Newton):\n", estimated_position)

    estimated_position = utils.estimate_transmitter_position_lm(
        anchors, twr_measurements
    )
    print("Estimated Transmitter Position (Levenberg-Marquardt):\n", estimated_position)


if __name__ == "__main__":
    main()
