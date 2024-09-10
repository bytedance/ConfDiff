"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

import os
from typing import List, Optional, Tuple, Dict

import hydra
import torch
from omegaconf import DictConfig
from lightning.pytorch.loggers import Logger
from lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything

from src.utils import hydra_utils
from src.utils.misc.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

import pyrootutils

log = hydra_utils.get_pylogger(__name__)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    # load environment variables from `.env` file if it exists
    # recursively searches for `.env` in all folders starting from work dir
    dotenv=True,
)


def load_state_dict_from_checkpoint(ckpt_path):
    if os.path.isdir(ckpt_path):
        # process deepspeed zero checkpoint
        state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
        state_dict = {key.replace("_forward_module.", ""): val for key, val in state_dict.items()}
    else:
        # regular checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
    return state_dict


@hydra_utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict, Dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[Dict, Dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = hydra_utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = hydra_utils.instantiate_loggers(cfg.get("logger"))
    
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        hydra_utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("ckpt_path") is not None:
        if cfg.get("load_state_dict_only", False):
            # load state_dict from checkpoint
            log.info(f"Loading model state_dict from {cfg.ckpt_path}")
            state_dict = load_state_dict_from_checkpoint(cfg.ckpt_path)
            model.load_state_dict(state_dict=state_dict, strict=True)
            log.info("Start training!")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)
        else:
            # resume training
            log.info(f"Resume training from checkpoint {cfg.ckpt_path}")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    else:
        # train
        log.info("Start training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    hydra_utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = hydra_utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
