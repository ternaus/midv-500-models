import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def paths2ids(paths: List[Path]) -> Dict[str, Path]:
    return {x.stem: x for x in paths}


def get_mask(size: Tuple[int, int], label: dict) -> np.ndarray:
    mask = np.zeros(size)

    poly = np.array(label["quad"])

    return cv2.fillPoly(mask, [poly], (255,))


def get_args():
    parser = argparse.ArgumentParser("Preprocessing the dataset.")
    arg = parser.add_argument
    arg(
        "-i",
        "--input_path",
        type=Path,
        help="Path to the unpacked dataset.",
        required=True,
    )
    arg("-o", "--output_path", type=Path, help="Path where to store resulting files.")
    return parser.parse_args()


def main():
    args = get_args()

    output_path = args.output_path
    image_output_path = output_path / "images"
    mask_output_path = output_path / "masks"
    image_output_path.mkdir(exist_ok=True, parents=True)
    mask_output_path.mkdir(exist_ok=True, parents=True)

    images = sorted(args.input_path.rglob("*.tif"))
    labels = sorted(args.input_path.rglob("*.json"))

    images_id2path = paths2ids(images)
    labels_id2path = paths2ids(labels)

    for image_id in tqdm(images_id2path.keys()):
        if image_id not in labels_id2path:
            continue
        image_path = images_id2path[image_id]
        label_path = labels_id2path[image_id]

        with open(label_path) as f:
            label = json.load(f)

        if "quad" not in label:
            continue

        image = cv2.imread(str(image_path))

        height, width = image.shape[:2]

        mask = get_mask((height, width), label)

        cv2.imwrite(str(image_output_path / f"{image_id}.jpg"), image)
        cv2.imwrite(str(mask_output_path / f"{image_id}.png"), mask)


if __name__ == "__main__":
    main()
