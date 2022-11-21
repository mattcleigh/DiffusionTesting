import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import hydra
import logging
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.utils import (
    instantiate_collection,
    log_hyperparameters,
    print_and_save_config,
    reload_original_config
)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=root / "configs", config_name="train_diff_gen.yaml")  # type: ignore
def main(cfg: DictConfig) -> None:

    # Check if the original config should be reloaded and print
    if cfg.full_resume:
        cfg = reload_original_config(cfg)
    print_and_save_config(cfg)

    if cfg.seed:
        log.info(f"Setting seed to: {cfg.seed}")
        pl.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    log.info("Instantiating the model")
    model = hydra.utils.instantiate(
        cfg.model,
        inpt_dim=datamodule.get_image_shape(),
        ctxt_dim=datamodule.get_ctxt_shape(),
    )

    log.info("Instantiating all callbacks")
    callbacks = instantiate_collection(cfg.callbacks)

    log.info("Instantiating the loggers")
    loggers = instantiate_collection(cfg.loggers)
    log.info(model)

    log.info("Instantiating the trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if loggers:
        log.info("Logging all hyperparameters")
        log_hyperparameters(cfg, model, trainer)

    if cfg.train:
        log.info("Starting training!")
        trainer.fit(model, datamodule, ckpt_path=cfg.ckpt_path)

    if cfg.test:
        log.info("Starting testing!")
        trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
