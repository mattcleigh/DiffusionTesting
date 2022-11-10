import pytorch_lightning as pl
import torch.nn as nn
from src.models.unetsr import UNetSuperResolution

model = UNetSuperResolution.load_from_checkpoint(
    "~/scratch/Saved_Networks/ImageProcessing/super_resolution/2022-11-09_01-17-18/checkpoints/last.ckpt",
    "cpu",
    loss_fn=nn.MSELoss(),
)
