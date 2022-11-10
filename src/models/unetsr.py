import logging
from typing import Optional
import pytorch_lightning as pl
from dotmap import DotMap
from functools import partial

import torch as T
import torch.nn as nn
from torch.nn.functional import avg_pool2d, interpolate
from torchvision.transforms import Compose

from mattstools.cnns import UNet
from mattstools.torch_utils import get_sched

from src.datamodules.images import ImageDataModule
from src.models.utils import log_wandb_upscale_images


log = logging.getLogger(__name__)


class UNetSuperResolution(pl.LightningModule):
    """A image to image model for doubling the resolution of an image"""

    def __init__(
        self,
        *,
        inpt_dim: list,
        ctxt_dim: int,
        upscale_factor: int,
        unet_kwargs: DotMap,
        loss_fn: nn.Module,
        optimizer: partial,
        sched_conf: dict,
    ) -> None:
        """Init for UNetSuperResolution

        Args:
            inpt_dim: The input dimension of the images
            ctxt_dim: The size of the context vector for each image
            upscale_factor: The super resolution upscaling amount
            unet_kwargs: The UNet module to use as the base of this network
            loss-fn: Loss function to use for the reconstruction error
            optimizer: Partially initialised optimizer
            sched_conf: The config for how to apply the scheduler
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Save specific class attributes
        self.loss_fn = loss_fn

        # Initialise the main model making up this network
        self.unet = UNet(
            inpt_size=inpt_dim[1:],
            inpt_channels=inpt_dim[0],
            outp_channels=inpt_dim[0],
            ctxt_dim=ctxt_dim,
            **unet_kwargs,
        )

    def _step(self, sample: tuple) -> T.Tensor:
        """Unpacks sample, downsamples image, passes through net, returns recon loss"""
        images, ctxt = sample
        in_images = avg_pool2d(images, self.hparams.upscale_factor)
        return self.loss_fn(self.forward(in_images, ctxt), images)

    def forward(self, images: T.Tensor, ctxt: Optional[T.Tensor] = None) -> T.Tensor:
        """Takes an image, interpolates to upscale dims, and passes through net"""
        images = interpolate(images, scale_factor=self.hparams.upscale_factor)
        return self.unet(images, ctxt)

    def training_step(self, batch: tuple, _batch_idx: int) -> T.Tensor:
        """Single training step"""
        loss = self._step(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> T.Tensor:
        """Single validation step, also visualises the outputs of the first batch"""
        if batch_idx == 0:
            log_wandb_upscale_images(self, batch, self.hparams.upscale_factor)
        loss = self._step(batch)
        self.log("valid/loss", loss)
        return loss

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate sheduler for this model"""

        # Finish initialising the partialy created methods
        opt = self.hparams.optimizer(params=self.parameters())

        # Use mattstools to initialise the scheduler (cyclic-epoch sync)
        sched = get_sched(
            self.hparams.sched_conf.mattstools,
            opt,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            max_epochs=self.trainer.max_epochs,
        )

        # Return the dict for the lightning trainer
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, **self.hparams.sched_conf.lightning},
        }
