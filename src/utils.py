"""
A collection of misculaneous functions usefull for the lighting/hydra template
"""

import os
from pathlib import Path
import rich
import rich.tree
import rich.syntax
import hydra
import logging
from typing import List, Any, Sequence
from omegaconf import OmegaConf, DictConfig

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning import Trainer, LightningModule

log = logging.getLogger(__name__)

@rank_zero_only
def reload_original_config(cfg: OmegaConf) -> OmegaConf:
    """Replaces the cfg with the one stored at the checkpoint location

    Will also set the chkpt_dir to the latest version of the 'last' checkpoint

    """
    orig_cfg = OmegaConf.load(Path("full_config.yaml"))
    orig_cfg.ckpt_path = sorted(
        Path.cwd().glob("checkpoints/last*.ckpt"), key=os.path.getmtime
    )[-1]
    return orig_cfg


@rank_zero_only
def print_and_save_config(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "loggers",
        "trainer",
        "paths",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Also saves the config to the output directory.
    This is necc ontop of hydra's default conf.yaml as it will resolve the entries
    allowing one to resume jobs identically with elements such as ${now:%H-%M-%S}.

    Furthermore, hydra does not allow resuming a previous job from the same dir.
    The work around is reload_original_config but that will fail as hydra overwites
    the default config.yaml file on startup, so this backup is needed for resuming.

    Args:
        cfg: Configuration composed by Hydra.
        print_order: Determines in what order config components are printed.
        resolve: Whether to resolve reference fields of DictConfig.
        save_to_file: Whether to export config to the hydra output folder.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.insert(0, field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    OmegaConf.save(cfg, Path(cfg.paths.full_path, "full_config.yaml"), resolve=True)


@rank_zero_only
def log_hyperparameters(
    cfg: DictConfig, model: LightningModule, trainer: Trainer
) -> None:
    """Passes the config dict to the trainer's logger, also calculates # params"""

    # Convert the config object to a hyperparameter dict
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # calculate the number of trainable parameters in the model and add it
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    trainer.logger.log_hyperparams(hparams)


def instantiate_collection(cfg_coll: DictConfig) -> List[Any]:
    """Uses hydra to instantiate a collection of classes and return a list"""
    objs = []

    if not cfg_coll:
        log.warning("List of configs is empty")
        return objs

    if not isinstance(cfg_coll, DictConfig):
        raise TypeError("List of configs must be a DictConfig!")

    for _, cb_conf in cfg_coll.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating <{cb_conf._target_}>")
            objs.append(hydra.utils.instantiate(cb_conf))

    return objs
