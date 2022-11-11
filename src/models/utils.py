import numpy as np
import wandb

import math
import torch as T
import torch.nn as nn
from torch.nn.functional import avg_pool2d

from PIL.Image import Image, fromarray
from torchvision.utils import make_grid

from mattstools.torch_utils import to_np

def tens_to_img(out) -> Image:
    """Convert a NORMALIZED pytorch tensor to a PIL Image"""
    out = make_grid(out, value_range=(-1, 1), normalize=True)
    out = out.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return fromarray(out)

def log_wandb_upscale_images(
    model: nn.Module,
    sample: tuple,
    upscale_factor: int,
) -> None:
    """Log the upscaled images and targets in using a wandb data table"""

    # Only runs if wandb is currently active
    if wandb.run is None:
        return

    # Split the input batch
    images, ctxt = sample

    # Get the low quality input images for the network
    in_images = avg_pool2d(images, upscale_factor)

    # Get the network outputs
    outputs = model(in_images, ctxt)

    # Create the wandb table
    test_table = wandb.Table(columns=["idx", "input", "output", "truth"])
    for idx, (i, o, t) in enumerate(zip(in_images, outputs, images)):
        img_id = str(idx)
        test_table.add_data(
            img_id,
            wandb.Image(tens_to_img(i)),
            wandb.Image(tens_to_img(o)),
            wandb.Image(tens_to_img(t)),
        )

    # Log the table
    wandb.run.log({"test_predictions": test_table}, commit=False)
