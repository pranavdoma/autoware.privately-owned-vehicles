"""Generate preprocess homography C: dataset ground H (YAML) + fixed VisionPilot model V."""

import argparse
from pathlib import Path

import cv2
import numpy as np

CANONICAL_WORLD_PTS = np.array(
    [[15, 5, 1], [150, 5, 1], [15, -5, 1], [150, -5, 1]], dtype=np.float32
)

# DO NOT MODIFY! VisionPilot model-view homography (1024x512 pixel -> world).
VISIONPILOT_MODEL_H = np.array(
    [
        [0.00209514907, -0.000941721466, -9.24906396],
        [0.00662758637, -0.000352940531, -3.33396502],
        [0.000120077371, -0.00411343505, 1.0],
    ],
    dtype=np.float32,
)


def load_homography_yaml(path: Path) -> np.ndarray:
    """Load 3x3 H from OpenCV or list-style YAML (same formats as fusion loader)."""
    if not path.is_file():
        raise FileNotFoundError(f"Cannot open homography YAML: {path}")

    text = path.read_text()
    data: list[float] = []
    in_data = False
    bracket_buf = ""

    def flush_bracket(buf: str) -> None:
        for part in buf.replace(",", " ").split():
            if part.strip():
                data.append(float(part))

    for line in text.splitlines():
        if "data:" in line and "[" in line:
            bracket_buf = line[line.find("[") + 1 :]
            if "]" in bracket_buf:
                flush_bracket(bracket_buf[: bracket_buf.find("]")])
                break
            in_data = True
            continue
        if bracket_buf:
            if "]" in line:
                flush_bracket(bracket_buf + line[: line.find("]")])
                break
            bracket_buf += " " + line
            continue
        if in_data:
            dash = line.find("-")
            if dash == -1:
                continue
            try:
                data.append(float(line[dash + 1 :].strip()))
            except ValueError:
                continue
        elif line.strip().startswith("data:"):
            in_data = True

    if len(data) != 9:
        raise ValueError(f"Expected 9 values under H/data in {path}, got {len(data)}")
    return np.array(data, dtype=np.float32).reshape(3, 3)


def find_homography_C_matrix(H: np.ndarray) -> np.ndarray:
    V = VISIONPILOT_MODEL_H
    H_inv = np.linalg.inv(H)
    V_inv = np.linalg.inv(V)

    uv = H_inv @ CANONICAL_WORLD_PTS.T
    uv_pts = (uv[:2] / uv[2]).T
    pq = V_inv @ CANONICAL_WORLD_PTS.T
    pq_pts = (pq[:2] / pq[2]).T

    C, _ = cv2.findHomography(uv_pts, pq_pts, method=0)
    return C


def main() -> None:
    parser = argparse.ArgumentParser(description="Build C from --ground-h and fixed VP model V")
    parser.add_argument("--ground-h", type=Path, required=True, help="Your full-frame ground H YAML")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ground_path = args.ground_h.resolve()
    H = load_homography_yaml(ground_path)
    C = find_homography_C_matrix(H)

    out = args.output or (Path(__file__).resolve().parents[1] / "build/config/homography_C_matrix.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(out.resolve()), cv2.FILE_STORAGE_WRITE)
    fs.write("C", C.astype(np.float32))
    fs.release()
    print(f"Wrote C from ground={ground_path} -> {out.resolve()}")


if __name__ == "__main__":
    main()
