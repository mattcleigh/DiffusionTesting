from functools import partial
import math
import copy
from typing import Mapping, Optional, Tuple
import PIL.Image

from tqdm import tqdm

import torch as T
from torchvision.utils import make_grid

import pytorch_lightning as pl

from mattstools.cnns import UNet
from mattstools.torch_utils import get_loss_fn, get_sched
from mattstools.modules import IterativeNormLayer
from mattstools.diffusion import CosineEncoding, DiffusionSchedule

import wandb


class DiffusionGenerator(pl.LightningModule):
    """A generative model which uses the diffusion process

    Uses an exponential moving average version of the network for stability during
    evaluation
    """

    def __init__(
        self,
        *,
        inpt_dim: list,
        ctxt_dim: int,
        normaliser_config: Mapping,
        cosine_config: Mapping,
        diff_shedule_config: Mapping,
        unet_config: Mapping,
        optimizer: partial,
        sched_config: Mapping,
        ema_sync: float = 0.999,
        loss_name: str = "mse",
    ) -> None:
        """
        Args:
            inpt_dim: The input dimension of the images
            ctxt_dim: The size of the context vector for each image (ignored)
            normaliser_config: For the normalisation layer in the model
            cosine_config: For defining the cosine embedding arguments
            diff_shedule_config: For defining the diffusion scheduler
            time_embed_config: Keyword arguments for the Dense time embedder
            unet_config: Keyword arguments for the UNet network
            optimizer: Partially initialised optimizer
            sched_config: The config for how to apply the scheduler
            loss_name: Name of the loss function to use for noise estimation
            ema_sync: How fast the ema network syncs with the given one
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Class attributes
        self.loss_fn = get_loss_fn(loss_name)
        self.inpt_dim = inpt_dim
        self.cosine_config = cosine_config
        self.diff_shedule_config = diff_shedule_config
        self.ema_sync = ema_sync
        self.register_buffer("fixed_noise", T.randn((32, *self.inpt_dim)))

        # The learnable normalisation layer
        self.normaliser = IterativeNormLayer(self.inpt_dim, **normaliser_config)

        # The encoder and scheduler needed for diffusion
        self.diff_sched = DiffusionSchedule(**diff_shedule_config)
        self.time_encoder = CosineEncoding(**cosine_config)

        # The denoising UNet
        self.unet = UNet(
            inpt_size=inpt_dim[1:],
            inpt_channels=inpt_dim[0],
            outp_channels=inpt_dim[0],
            ctxt_dim=self.time_encoder.outp_dim,
            **unet_config,
        )

        # A copy of the unet which will sync with an exponential moving average
        self.ema_unet = copy.deepcopy(self.unet)

        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("train/noise_loss", summary="min")
            wandb.define_metric("train/image_loss", summary="min")
            wandb.define_metric("valid/noise_loss", summary="min")
            wandb.define_metric("valid/image_loss", summary="min")

    def denoise(
        self,
        noisy_images: T.Tensor,
        diffusion_times: T.Tensor,
        signal_rates: T.Tensor,
        noise_rates: T.Tensor,
    ) -> Tuple[T.Tensor, T.Tensor]:
        """Predict the gaussian noise that has been added to the image and predict the
        clean image using the
        """

        # Use the appropriate network for training or validation
        if self.training:
            network = self.unet
        else:
            network = self.ema_unet

        # Encode the times
        encoded_times = self.time_encoder(diffusion_times)

        # Use the selected network to esitmate the noise present in the image
        pred_noises = network(noisy_images, ctxt=encoded_times)

        # Apply the DDIM method to estimate the cleaned up image
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def _shared_step(self, sample: tuple) -> Tuple[T.Tensor, T.Tensor]:
        """Shared step used in both training and validaiton"""

        # Unpack the sample tuple (images come with labels/ctxt -> ignore)
        images, _ = sample

        # Pass through the normaliser
        images = self.normaliser(images)

        # Sample from the gaussian latent space to perturb the images
        noises = T.randn_like(images)

        # Sample uniform random diffusion times and get the rates
        diffusion_times = T.rand(size=(len(images), 1), device=self.device)
        signal_rates, noise_rates = self.diff_sched(diffusion_times.view(-1, 1, 1, 1))

        # Mix the signal and noise according to the diffusion equation
        noisy_images = signal_rates * images + noise_rates * noises

        # Predict the noise using the UNet
        pred_noises, pred_images = self.denoise(
            noisy_images, diffusion_times, signal_rates, noise_rates
        )

        # Calculate the loss terms, log, and return
        noise_loss = self.loss_fn(noises, pred_noises).mean()  # used for training
        image_loss = self.loss_fn(images, pred_images).mean()  # only used as metric
        return noise_loss, image_loss

    def training_step(self, sample: tuple, _batch_idx: int) -> T.Tensor:
        noise_loss, image_loss = self._shared_step(sample)
        self._sync_ema_network()
        self.log("train/noise_loss", noise_loss)
        self.log("train/image_loss", image_loss)
        self.log("means", self.normaliser.means.mean())
        self.log("vars", self.normaliser.vars.mean())
        return noise_loss

    def validation_step(self, sample: tuple, _batch_idx: int) -> T.Tensor:
        noise_loss, image_loss = self._shared_step(sample)
        self.log("valid/noise_loss", noise_loss)
        self.log("valid/image_loss", image_loss)
        return noise_loss

    def _sync_ema_network(self) -> None:
        """Updates the Exponential Moving Average Network"""
        with T.no_grad():
            for params, ema_params in zip(
                self.unet.parameters(), self.ema_unet.parameters()
            ):
                ema_params.data.copy_(
                    self.ema_sync * ema_params.data
                    + (1.0 - self.ema_sync) * params.data
                )

    def generate(
        self,
        initial_noise: Optional[T.Tensor] = None,
        n_steps: int = 50,
        keep_all: bool = False,
        num_images: int = 1,
        clip_predictions: Optional[tuple] = None,
    ) -> Tuple[T.Tensor, list]:
        """Apply the full reverse process to noise to generate a batch of images

        Args:
            initial_noise: The initial noise to pass through the process
                If none it will be generated here
            n_steps: The number of iterations to generate the images
            keep_all: Return all stages of diffusion process
                Can be memory heavy for large batches
            num_images: How many images to generate
                Ignored if initial_noise is provided
            clip_predictions: Can stabalise generation by clipping the outputs
        """

        # Get the initial noise for generation and the number of images
        if initial_noise is None:
            initial_noise = T.randn((num_images, *self.inpt_dim), device=self.device)
        num_images = initial_noise.shape[0]

        # Check the input argument for the n_steps, must be less than what was trained
        all_image_stages = []
        step_size = 1 / n_steps

        # Do the very first step of the iteration using pure noise
        next_noisy_images = initial_noise
        next_diff_times = T.ones((num_images, 1), device=self.device)
        next_signal_rates, next_noise_rates = self.diff_sched(
            next_diff_times.view(-1, 1, 1, 1)
        )

        # Cycle through the remainder of all the steps
        for step in tqdm(range(n_steps), "generating", leave=False):

            # Update with the previous 'next' step
            noisy_images = next_noisy_images
            diff_times = next_diff_times
            signal_rates = next_signal_rates
            noise_rates = next_noise_rates

            # Apply the denoise step to get X_0 and expected noise
            pred_noises, pred_images = self.denoise(
                noisy_images, diff_times, signal_rates, noise_rates
            )

            # Get the next predicted components using the next signal and noise rates
            next_diff_times = diff_times - step_size
            next_signal_rates, next_noise_rates = self.diff_sched(
                next_diff_times.view(-1, 1, 1, 1)
            )

            # Clamp the predicted X_0 for stability
            if clip_predictions is not None:
                pred_images.clamp_(*clip_predictions)

            # Remix the predicted components to go from estimated X_0 -> X_{t-1}
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

            # Keep track of the diffusion evolution
            if keep_all:
                all_image_stages.append(self.normaliser.reverse(next_noisy_images))

        pred_images = self.normaliser.reverse(pred_images)

        return pred_images, all_image_stages

    def validation_epoch_end(self, *_args) -> None:
        """Use this callback to log generated images using wandb"""

        # Only runs if wandb is currently active
        if wandb.run is None:
            return

        # Get the generated samples
        outputs, _ = self.generate(initial_noise=self.fixed_noise, n_steps=50)

        # Create the wandb table and add the data
        gen_table = wandb.Table(columns=["idx", "output"])
        for idx, out in enumerate(outputs):

            # Manually scale the image (expecting values normalised between -1, 1)
            out = make_grid(out, value_range=(-1, 1), normalize=True)
            out = out.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            out = PIL.Image.fromarray(out)

            img_id = str(idx)
            gen_table.add_data(img_id, wandb.Image(out))

        # Log the table
        wandb.run.log({"generated": gen_table}, commit=False)

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate sheduler for this model"""

        # Finish initialising the partialy created methods
        opt = self.hparams.optimizer(params=self.parameters())

        # Use mattstools to initialise the scheduler (cyclic-epoch sync)
        sched = get_sched(
            self.hparams.sched_config.mattstools,
            opt,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            max_epochs=self.trainer.max_epochs,
        )

        # Return the dict for the lightning trainer
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, **self.hparams.sched_config.lightning},
        }
