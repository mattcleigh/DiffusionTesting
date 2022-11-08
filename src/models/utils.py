import numpy as np
import wandb

import torch.nn as nn
from torch.nn.functional import avg_pool2d

from mattstools.torch_utils import to_np


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

    # Create the wandb table
    columns = ["idx", "input", "output", "truth"]
    test_table = wandb.Table(columns=columns)

    # Get the low quality input images for the network
    in_images = avg_pool2d(images, upscale_factor)

    # Get the network outputs
    outputs = model(in_images, ctxt)

    # Convert to numpy
    inputs = to_np(in_images)
    outputs = to_np(outputs)
    truth = to_np(images)

    # Add all data to the table
    for idx, (i, o, t) in enumerate(zip(inputs, outputs, truth)):
        img_id = str(idx)
        test_table.add_data(
            img_id,
            wandb.Image(np.transpose(i, (1, 2, 0))),
            wandb.Image(np.transpose(o, (1, 2, 0))),
            wandb.Image(np.transpose(t, (1, 2, 0))),
        )

    # Log the table
    wandb.run.log({"test_predictions": test_table}, commit=False)
