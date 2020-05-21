import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import yaml
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.utils.image_utils import unpad
from iglovikov_helper_functions.utils.mask_utils import remove_small_connected_binary
from pytorch_toolbelt.inference import tta
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from midv500models.dataloaders import SegmentationDatasetTest
from midv500models.train import SegmentDocs
from midv500models.utils import load_checkpoint, mask_overlay


def fill_small_holes(mask: np.ndarray, min_area: int) -> np.ndarray:
    inverted_mask = 1 - mask
    inverted_mask = remove_small_connected_binary(inverted_mask, min_area=min_area)
    return 1 - inverted_mask


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg("-i", "--image_path", type=Path, help="Path to images.", required=True)
    arg("-o", "--output_path", type=Path, help="Path to images.", required=True)
    arg("-w", "--checkpoint_path", type=Path, help="Path to weights.", required=True)
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    model = SegmentDocs(hparams)

    test_file_names = sorted(Path(args.image_path).glob("*.jpg"))

    test_mask_path = args.output_path / "masks"
    test_vis_path = args.output_path / "vis"

    test_mask_path.mkdir(exist_ok=True, parents=True)
    test_vis_path.mkdir(exist_ok=True, parents=True)

    test_aug = from_dict(hparams["test_aug"])

    dataloader = DataLoader(
        SegmentationDatasetTest(test_file_names, test_aug),
        batch_size=hparams["test_parameters"]["batch_size"],
        num_workers=hparams["num_workers"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    corrections: Dict[str, str] = {}

    checkpoint = load_checkpoint(file_path=args.checkpoint_path, rename_in_layers=corrections)  # type: ignore

    model.load_state_dict(checkpoint["state_dict"])
    model = nn.Sequential(model, nn.Sigmoid())

    model = tta.MultiscaleTTAWrapper(model, [0.5, 2])

    model.eval()
    model = model.half()
    model.cuda()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            features = batch["features"]
            image_ids = batch["image_id"]

            preds = tta.fliplr_image2mask(model, features.half().cuda())

            for batch_id in range(features.shape[0]):
                image_id = image_ids[batch_id]
                mask = (preds[batch_id][0] > 0.5).cpu().numpy().astype(np.uint8)

                height = batch["height"][batch_id].item()
                width = batch["width"][batch_id].item()
                pads = batch["pads"][batch_id].cpu().numpy()

                mask = unpad(mask, pads)

                mask = remove_small_connected_binary(mask, min_area=100)
                mask = fill_small_holes(mask, min_area=100)

                mask = cv2.resize(
                    mask, (width, height), interpolation=cv2.INTER_NEAREST
                )

                cv2.imwrite(str(test_mask_path / f"{image_id}.png"), mask * 255)

                image = cv2.imread(str(args.image_path / f"{image_id}.jpg"))

                mask_image = mask_overlay(image, mask)

                cv2.imwrite(
                    str(test_vis_path / f"{image_id}.jpg"),
                    np.hstack(
                        [mask_image, cv2.cvtColor((mask * 255), cv2.COLOR_GRAY2BGR)]
                    ),
                )


if __name__ == "__main__":
    main()
