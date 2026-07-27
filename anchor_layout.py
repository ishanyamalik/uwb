import matplotlib.pyplot as plt
import numpy as np

PRESET_ANCHOR_LAYOUT_DESCRIPTIONS: dict[str, str] = {
    "perimeter_corners_cuboid": (
        "8 corner vertices cuboid. "
        "Places non-coplanar alternating high/low anchors at room corners."
        "https://www.pozyx.io/pozyx-academy/where-to-place-the-anchors"
    ),
    "max_volume_4anchors": (
        "4 Anchors Maximum Volume Strategy."
        "https://www.sciencedirect.com/science/article/pii/S0957417424016750#s0035"
    ),
    "perimeter_staggered_heights": (
        "8 Perimeter Wall Anchors with staggered heights."
        "https://follow-me.com/knowledge-base/anchor-tag-placement/"
        "https://www.pozyx.io/pozyx-academy/where-to-place-the-anchors"
    ),
    "convex_polygon_center_elevated": (
        "6 anchors: 4 low outer corners + 2 elevated center ceiling anchors."
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC12389748/"
    ),
    "equilateral_triangle_mesh": (
        "10 Anchors layout arranged in an equilateral triangle mesh pattern."
    ),
    "center_wall_ceiling": (
        "5 Anchors layout with 4 anchors on the center of each wall and 1 anchor on the ceiling."
    ),
    "random": ("Random anchor layout with 4 to 10 anchors."),
}


def get_preset_anchor_layout(
    layout_name: str, room_length: float, room_width: float, room_height: float
) -> np.ndarray:
    print(f"Layout: {layout_name}")
    z_low = max(0.5, room_height * 0.1)
    z_mid = room_height * 0.5
    z_high = max(z_low + 0.5, room_height - 0.5)

    if layout_name == "perimeter_corners_cuboid":
        return np.array(
            [
                [0.5, 0.5, z_low],  # Bottom-left Front
                [room_length - 0.5, 0.5, z_high],  # Bottom-right Front
                [room_length - 0.5, room_width - 0.5, z_low],  # Bottom-right Back
                [0.5, room_width - 0.5, z_high],  # Bottom-left Back
                [0.5, 0.5, z_high],  # Top-left Front (Ceiling)
                [room_length - 0.5, 0.5, z_low],  # Top-right Front
                [room_length - 0.5, room_width - 0.5, z_high],  # Top-right Back
                [0.5, room_width - 0.5, z_low],  # Top-left Back
            ],
            dtype=float,
        )
    elif layout_name == "max_volume_4anchors":
        z_mid_low = z_low + 0.25 * (z_high - z_low)
        return np.array(
            [
                [0.5, 0.5, z_low],  # Bottom-left Front
                [room_length - 0.5, 0.5, z_mid_low],  # Bottom-right Front
                [0.5, room_width - 0.5, z_high],  # Top-left Back
                [room_length - 0.5, room_width - 0.5, z_high],  # Top-right Back
            ],
            dtype=float,
        )
    elif layout_name == "perimeter_staggered_heights":
        return np.array(
            [
                [0.5, 0.5, z_low],  # SW corner (Low)
                [room_length * 0.5, 0.5, z_high],  # S Mid (High)
                [room_length - 0.5, 0.5, z_low],  # SE corner (Low)
                [room_length - 0.5, room_width * 0.5, z_high],  # EM corner (High)
                [room_length - 0.5, room_width - 0.5, z_low],  # NE corner (Low)
                [room_length * 0.5, room_width - 0.5, z_high],  # N Mid (High)
                [0.5, room_width - 0.5, z_low],  # NW corner (Low)
                [0.5, room_width * 0.5, z_high],  # W Mid (High)
            ],
            dtype=float,
        )
    elif layout_name == "convex_polygon_center_elevated":
        return np.array(
            [
                [0.5, 0.5, z_low],  # Bottom-left Front
                [room_length - 0.5, 0.5, z_low],  # Bottom-right Front
                [room_length - 0.5, room_width - 0.5, z_low],  # Bottom-right Back
                [0.5, room_width - 0.5, z_low],  # Bottom-left Back
                [room_length * 0.35, room_width * 0.5, z_high],  # High Overhead Left
                [room_length * 0.65, room_width * 0.5, z_high],  # High Overhead Right
            ],
            dtype=float,
        )
    elif layout_name == "equilateral_triangle_mesh":
        # 10 anchors arranged in equilateral triangle rows with height diversity
        s = max(2.0, (room_length - 2.0) / 3.0)
        h_tri = s * (np.sqrt(3) / 2.0)
        y1 = max(1.0, (room_width - 2 * h_tri) / 2.0)
        y2 = y1 + h_tri
        y3 = y1 + 2 * h_tri
        return np.array(
            [
                # Row 1 (low elevation)
                [1.0 + s * 0.5, min(y1, room_width - 1.0), z_low],
                [1.0 + s * 1.5, min(y1, room_width - 1.0), z_low],
                [1.0 + s * 2.5, min(y1, room_width - 1.0), z_low],
                # Row 2 (high ceiling elevation, offset by s/2)
                [1.0 + s * 0.0, min(y2, room_width - 1.0), z_high],
                [1.0 + s * 1.0, min(y2, room_width - 1.0), z_high],
                [1.0 + s * 2.0, min(y2, room_width - 1.0), z_high],
                [
                    min(room_length - 1.0, 1.0 + s * 3.0),
                    min(y2, room_width - 1.0),
                    z_high,
                ],
                # Row 3 (mid elevation)
                [1.0 + s * 0.5, min(y3, room_width - 1.0), z_mid],
                [1.0 + s * 1.5, min(y3, room_width - 1.0), z_mid],
                [1.0 + s * 2.5, min(y3, room_width - 1.0), z_mid],
            ],
            dtype=float,
        )
    elif layout_name == "center_wall_ceiling":
        return np.array(
            [
                [room_length * 0.5, 0.5, z_mid],  # Front wall center
                [room_length * 0.5, room_width - 0.5, z_mid],  # Back wall center
                [0.5, room_width * 0.5, z_mid],  # Left wall center
                [room_length - 0.5, room_width * 0.5, z_mid],  # Right wall center
                [room_length * 0.5, room_width * 0.5, z_high],  # Ceiling center
            ],
            dtype=float,
        )
    elif layout_name == "random":
        num_anchors = np.random.randint(4, 11)
        return np.random.rand(num_anchors, 3) * np.array(
            [room_length, room_width, room_height]
        )
    else:
        raise ValueError(f"Unknown layout name: {layout_name}")


def plot_anchor_layout(
    layout_name: str,
    anchors: np.ndarray,
    room_length: float,
    room_width: float,
    room_height: float,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot anchors
    ax.scatter(
        anchors[:, 0],
        anchors[:, 1],
        anchors[:, 2],
        c="r",
        marker="o",
        label="Anchors",
        s=100,
    )
    for i, a in enumerate(anchors):
        ax.text(
            a[0] + 0.5,
            a[1] + 0.5,
            a[2] + 0.5,
            f"({a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f})",
            color="black",
            fontsize=9,
        )
    # Set room dimensions
    ax.set_xlim([0, room_length])
    ax.set_ylim([0, room_width])
    ax.set_zlim([0, room_height])

    ax.set_xlabel("X (Length)")
    ax.set_ylabel("Y (Width)")
    ax.set_zlabel("Z (Height)")
    ax.set_title(f"Anchor Layout in Room: {layout_name}")
    ax.legend()
    filename = f"images/anchor_layout_{layout_name}.png"
    print(f"Saving plot to {filename}")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Preset Anchor Layout Descriptions:")
    room_length = 10.0
    room_width = 10.0
    room_height = 10.0

    for layout_name in PRESET_ANCHOR_LAYOUT_DESCRIPTIONS:
        anchors = get_preset_anchor_layout(
            layout_name, room_length, room_width, room_height
        )
        print(f"Layout: {layout_name}")
        print(anchors)
        plot_anchor_layout(layout_name, anchors, room_length, room_width, room_height)
        print()


if __name__ == "__main__":
    main()
