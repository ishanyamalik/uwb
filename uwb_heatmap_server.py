"""Web server for generating and displaying UWB localization heatmaps."""

import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from flask import Flask, Response, request

import anchor_layout
from uwb_heatmap_2d_parallel import evaluate_grid_point

# if (image.src.startsWith("blob:")) URL.revokeObjectURL(image.src);

MAX_GRID_POINTS = 2_500
MAX_TRIALS = 1_000
HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UWB Heatmap</title>
  <style>
    body { font-family: sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; }
    form { display: grid; gap: .75rem; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
    label { display: grid; gap: .25rem; }
    input, select, button { font: inherit; padding: .45rem; }
    button { cursor: pointer; grid-column: 1 / -1; }
    #status { min-height: 1.5rem; }
    img { display: block; max-width: 100%; }
  </style>
</head>
<body>
  <h1>UWB Localization Error Heatmap</h1>
  <form id="heatmap-form">
    <label>Room X (m)<input name="room_x" type="number" min="0.5" step="0.5" value="10"></label>
    <label>Room Y (m)<input name="room_y" type="number" min="0.5" step="0.5" value="10"></label>
    <label>Room Z (m)<input name="room_z" type="number" min="0.5" step="0.5" value="10"></label>
    <label>Trials<input name="num_trials" type="number" min="1" max="1000" step="1" value="20"></label>
    <label>Transmitter Z (m)<input name="transmitter_z" type="number" min="0" step="0.1" value="0.5"></label>
    <label>Grid increment (m)<input name="grid_increment" type="number" min="0.1" step="0.1" value="1"></label>
    <label>Anchor layout
    <select name="anchor_layout">__ANCHOR_OPTIONS__</select>
    </label>
    <button type="submit">Generate heatmap</button>
  </form>
  <p id="status" role="status"></p>
  <img id="heatmap">
  <script>
    const form = document.getElementById("heatmap-form");
    const status = document.getElementById("status");
    const image = document.getElementById("heatmap");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      status.textContent = "Generating...";
      image.removeAttribute("src");
      const values = Object.fromEntries(new FormData(form));
      for (const key of ["room_x", "room_y", "room_z", "num_trials", "transmitter_z", "grid_increment"]) {
        values[key] = Number(values[key]);
      }
      try {
        const response = await fetch("/heatmap", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(values)
        });
        if (!response.ok) throw new Error(await response.text());
        const blob = await response.blob();
        image.src = URL.createObjectURL(blob);
        status.textContent = "Heatmap generated.";
      } catch (error) {
        status.textContent = `Error: ${error.message}`;
      }
    });
  </script>
</body>
</html>
"""
app = Flask(__name__)


def render_page() -> bytes:
    options = "".join(
        f'<option value="{name}">{name}</option>'
        for name in anchor_layout.PRESET_ANCHOR_LAYOUT_DESCRIPTIONS
    )
    return HTML_PAGE.replace("__ANCHOR_OPTIONS__", options).encode("utf-8")


def validate_request(data: dict) -> dict:
    required = (
        "room_x",
        "room_y",
        "room_z",
        "num_trials",
        "transmitter_z",
        "grid_increment",
    )
    if not isinstance(data, dict):
        raise TypeError("Request body must be a JSON object")
    if any(key not in data for key in required):
        raise ValueError("Missing heatmap parameters")

    values = {key: float(data[key]) for key in required}
    values["num_trials"] = int(values["num_trials"])
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Heatmap parameters must be finite numbers")
    if any(
        values[key] <= 0 for key in ("room_x", "room_y", "room_z", "grid_increment")
    ):
        raise ValueError("Room dimensions and grid increment must be positive")
    if values["num_trials"] < 1 or values["num_trials"] > MAX_TRIALS:
        raise ValueError(f"num_trials must be between 1 and {MAX_TRIALS}")
    if values["transmitter_z"] < 0:
        raise ValueError("transmitter_z must not be negative")
    values["anchor_layout"] = data.get("anchor_layout", "perimeter_corners_cuboid")
    if values["anchor_layout"] not in anchor_layout.PRESET_ANCHOR_LAYOUT_DESCRIPTIONS:
        raise ValueError("Unknown anchor layout")
    return values


def generate_heatmap(data: dict) -> bytes:
    args = validate_request(data)
    anchors = anchor_layout.get_preset_anchor_layout(
        args["anchor_layout"], args["room_x"], args["room_y"], args["room_z"]
    )
    increment = args["grid_increment"]
    x_range = np.arange(0, args["room_x"] + increment, increment)
    y_range = np.arange(0, args["room_y"] + increment, increment)
    if len(x_range) * len(y_range) > MAX_GRID_POINTS:
        raise ValueError(f"Grid is too large; use at most {MAX_GRID_POINTS} points")

    error_grid = np.zeros((len(y_range), len(x_range)))
    tasks = [
        (row, col, x, y)
        for row, y in enumerate(y_range)
        for col, x in enumerate(x_range)
    ]
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                evaluate_grid_point,
                *task,
                anchors,
                args["num_trials"],
                args["transmitter_z"],
            )
            for task in tasks
        ]
        for future in as_completed(futures):
            row, col, error, _ = future.result()
            error_grid[row, col] = error

    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        np.clip(error_grid, 0, 0.5),
        cmap="RdYlGn_r",
        xticklabels=np.round(x_range, 1),
        yticklabels=np.round(y_range, 1),
        cbar_kws={"label": "Median Localization Error (meters)"},
        ax=axis,
    )
    axis.invert_yaxis()
    axis.set_title(
        f"UWB Localization Error Heatmap\nRoom: {args['room_x']}m x {args['room_y']}m x {args['room_z']}m | "
        f"Transmitter Height: {args['transmitter_z']}m | Trials: {args['num_trials']}"
    )
    axis.set_xlabel("X (meters)")
    axis.set_ylabel("Y (meters)")
    figure.tight_layout()
    output = BytesIO()
    figure.savefig(output, format="png")
    plt.close(figure)
    return output.getvalue()


@app.get("/")
def index() -> Response:
    return Response(render_page(), mimetype="text/html")


@app.post("/heatmap")
def heatmap() -> Response:
    try:
        body = generate_heatmap(request.get_json(silent=False))
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return Response(str(error), status=400, mimetype="text/plain")
    return Response(body, mimetype="image/png")


def main() -> None:
    app.run(host="127.0.0.1", port=8000, threaded=True)


if __name__ == "__main__":
    main()
