from functools import partial
import math
from typing import Mapping, Optional, Tuple

from tqdm import tqdm
import numpy as np

import torch as T
import pytorch_lightning as pl

from mattstools.cnns import UNet
from mattstools.modules import DenseNetwork
from mattstools.torch_utils import get_loss_fn, to_np, get_sched

import wandb


def cosine_encoding(
    x: T.Tensor,
    out_dim: int,
    min_value: float = 0.0,
    max_value: float = 1.0,
    frequency_scaling: str = "exponential",
) -> T.Tensor:
    """Computes a positional cosine encodings with an increasing series of frequencies

    The frequencies either increase linearly or exponentially (default).
    The latter is good for when max_value is large and extremely high sensitivity to the
    input is required.
    If inputs greater than the max value are provided, the outputs become degenerate.
    If inputs smaller than the min value are provided, the inputs the the cosine will
    be both positive and negative, which may lead degenerate outputs.

    Always make sure that the min and max bounds are not exceeded!

    Args:
        x: The input, the final dimension is encoded. If 1D then it will be unqueezed
        out_dim: The dimension of the output encoding
        min_value: Added to x (and max) as cosine embedding works with positive inputs
        max_value: The maximum expected value, sets the scale of the lowest frequency
        frequency_scaling: Either 'linear' or 'exponential'

    Returns:
        The cosine embeddings of the input using (out_dim) many frequencies
    """

    # Unsqueeze if final dimension is flat
    if x.shape[-1] != 1:
        x = x.unsqueeze(-1)

    # Check the the bounds are obeyed
    if T.any(x > max_value):
        print("Warning! Passing values to cosine_encoding encoding that exceed max!")
    if T.any(x < min_value):
        print("Warning! Passing values to cosine_encoding encoding below min!")

    # Calculate the various frequencies
    if frequency_scaling == "exponential":
        freqs = T.arange(out_dim, device=x.device).exp()
    elif frequency_scaling == "linear":
        freqs = T.arange(1, out_dim + 1, device=x.device)
    else:
        raise RuntimeError(f"Unrecognised frequency scaling: {frequency_scaling}")

    return T.cos((x + min_value) * freqs * math.pi / (max_value + min_value))


def diffusion_shedule(
    diff_time: T.Tensor, max_sr: float = 1, min_sr: float = 1e-8
) -> Tuple[T.Tensor, T.Tensor]:
    """Calculates the signal and noise rate for any point in the diffusion processes

    Using continuous diffusion times between 0 and 1 which make switching between
    different numbers of diffusion steps between training and testing much easier.
    Returns only the values needed for the jump forward diffusion step and the reverse
    DDIM step.
    These are sqrt(alpha_bar) and sqrt(1-alphabar) which are called the signal_rate
    and noise_rate respectively.

    The jump forward diffusion process is simply a weighted sum of:
        input * signal_rate + eps * noise_rate

    Uses a cosine annealing schedule as proposed in
    Proposed in https://arxiv.org/abs/2102.09672

    Args:
        diff_time: The time used to sample the diffusion scheduler
            Output will match the shape
            Must be between 0 and 1
        max_sr: The initial rate at the first step
        min_sr: How much signal is preserved at end of diffusion
            (can't be zero due to log)
    """

    ## Use cosine annealing, which requires switching from times -> angles
    start_angle = math.acos(max_sr)
    end_angle = math.acos(min_sr)
    diffusion_angles = start_angle + diff_time * (end_angle - start_angle)
    signal_rates = T.cos(diffusion_angles)
    noise_rates = T.sin(diffusion_angles)
    return signal_rates, noise_rates


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
        time_embedding_dim: int,
        cosine_config: Mapping,
        diff_shedule_config: Mapping,
        time_embed_config: Mapping,
        unet_config: Mapping,
        optimizer: partial,
        sched_conf: dict,
        loss_name: str = "mse",
    ) -> None:
        """
        Args:
            inpt_dim: The input dimension of the images
            ctxt_dim: The size of the context vector for each image (ignored)
            time_embedding_dim: Embedding size of the diffusion time encoding
            cosine_config: For defining the cosine embedding arguments
            diff_shedule_config: For defining the diffusion scheduler
            time_embed_config: Keyword arguments for the Dense time embedder
            unet_config: Keyword arguments for the UNet network
            optimizer: Partially initialised optimizer
            sched_conf: The config for how to apply the scheduler
            loss_name: Name of the loss function to use for noise estimation
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Load the appropriate loss function
        self.time_embedding_dim = time_embedding_dim
        self.loss_fn = get_loss_fn(loss_name)
        self.inpt_dim = inpt_dim
        self.cosine_config = cosine_config
        self.diff_shedule_config = diff_shedule_config
        self.register_buffer("fixed_noise", T.randn((32, *self.inpt_dim)))

        # The dense network used to embed the time information
        self.time_embed = DenseNetwork(
            inpt_dim=time_embedding_dim,
            outp_dim=time_embedding_dim,
            **time_embed_config,
        )

        # The denoising UNet
        self.unet = UNet(
            inpt_size=inpt_dim[1:],
            inpt_channels=inpt_dim[0],
            outp_channels=inpt_dim[0],
            ctxt_dim=self.time_embed.outp_dim,
            **unet_config,
        )

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

        # Use the unet to esitmate the noise present in the image
        pred_noises = self.unet(
            noisy_images,
            ctxt=self.time_embed(
                cosine_encoding(
                    diffusion_times,
                    out_dim=self.time_embedding_dim,
                    **self.cosine_config
                )
            ),
        )

        # Apply the DDIM method to estimate the cleaned up image
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def _shared_step(self, sample: tuple) -> Tuple[T.Tensor, T.Tensor]:
        """Shared step used in both training and validaiton"""

        # Unpack the sample tuple (images come with labels/ctxt -> ignore)
        images, _ = sample

        # Sample noise to perturb the images
        noises = T.randn_like(images)

        # Sample uniform random diffusion times and get the rates
        diffusion_times = T.rand(size=(len(images), 1), device=self.device)
        signal_rates, noise_rates = diffusion_shedule(diffusion_times.view(-1, 1, 1, 1), **self.diff_shedule_config)

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
        self.log("train/noise_loss", noise_loss)
        self.log("train/image_loss", image_loss)
        return noise_loss

    def validation_step(self, sample: tuple, _batch_idx: int) -> T.Tensor:
        noise_loss, image_loss = self._shared_step(sample)
        self.log("valid/noise_loss", noise_loss)
        self.log("valid/image_loss", image_loss)
        return noise_loss

    def generate(
        self,
        initial_noise: Optional[T.Tensor] = None,
        n_steps: int = 50,
        keep_all: bool = False,
        num_images: int = 1,
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
        """

        # Get the initial noise for generation and the number of images
        if initial_noise is None:
            initial_noise = T.randn((num_images, *self.inpt_dim), device=self.device)
        num_images = initial_noise.shape[0]

        # Check the input argument for the n_steps, must be less than what was trained
        all_image_stages = []
        step_size = 1 / n_steps

        # Do the very first step of the iteration using pure noise
        noisy_images = initial_noise
        diff_times = T.ones((num_images, 1), device=self.device)
        signal_rates, noise_rates = diffusion_shedule(
            diff_times.view(-1, 1, 1, 1), **self.diff_shedule_config
        )
        pred_noises, pred_images = self.denoise(
            noisy_images, diff_times, signal_rates, noise_rates
        )

        # Cycle through the remainder of all the steps
        for step in tqdm(range(1, n_steps), "generating"):

            # Make a single backward step using the predicted clean image
            diff_times = T.ones((num_images, 1), device=self.device) - step * step_size
            signal_rates, noise_rates = diffusion_shedule(diff_times.view(-1, 1, 1, 1), **self.diff_shedule_config)
            noisy_images = signal_rates * pred_images + noise_rates * pred_noises

            # Seperate the image out into the noise and final prediction
            pred_noises, pred_images = self.denoise(
                noisy_images, diff_times, signal_rates, noise_rates
            )

            # Keep track of the diffusion evolution
            if keep_all:
                all_image_stages.append(noisy_images)

        return pred_images, all_image_stages

    def validation_epoch_end(self, *_args) -> None:
        """Use this callback to log generated images using wandb"""

        # Only runs if wandb is currently active
        if wandb.run is None:
            return

        # Get the generated samples, convert to numpy
        outputs, _ = self.generate(initial_noise=self.fixed_noise, n_steps=50)
        outputs = to_np(outputs)

        # Create the wandb table amd add the data
        gen_table = wandb.Table(columns=["idx", "output"])
        for idx, out in enumerate(outputs):
            img_id = str(idx)
            gen_table.add_data(img_id, wandb.Image(np.transpose(out, (1, 2, 0))))

        # Log the table
        wandb.run.log({"generated": gen_table}, commit=False)

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