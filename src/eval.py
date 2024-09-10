"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
import shutil
from omegaconf import DictConfig, OmegaConf, open_dict
from lightning.pytorch.loggers import Logger
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.utilities import rank_zero_only

from src.utils import hydra_utils
from src.train import load_state_dict_from_checkpoint
from src.utils.misc.misc import replace_dot_key_val, get_git_commit

log = hydra_utils.get_pylogger(__name__)


@hydra_utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    cfg.logger = None  # turn off logger

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = hydra_utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        hydra_utils.log_hyperparameters(object_dict)

    # load state_dict from checkpoint
    if cfg.ckpt_path is not None:
        ckpt_path = list(Path(cfg.paths.log_dir).joinpath("checkpoints").glob("best_model*.ckpt"))[0] \
                    if cfg.ckpt_path == "infer" else cfg.ckpt_path
        assert os.path.exists(ckpt_path), f"{ckpt_path} does not exist."
        log.info(f"Loading model state_dict from {ckpt_path}")
        state_dict = load_state_dict_from_checkpoint(ckpt_path)
        model.load_state_dict(state_dict=state_dict, strict=False)
    else:
        log.warning("No checkpoint provided for evaluation")
    #     raise FileNotFoundError("No checkpoint provided for evaluation")
    
    # run test
    # log.info("Starting testing!")
    # ckpt_name = Path(ckpt_path).stem
    # output_dir = Path(cfg.paths.output_dir).joinpath(ckpt_name)
    # output_dir.mkdir(exist_ok=True, parents=True)
    # # copy checkpoint to the output dir if not already existed
    # if not output_dir.joinpath(Path(ckpt_path).name).exists():
    #     shutil.copy2(ckpt_path, output_dir)

    # model.output_dir = output_dir

    trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


def clean_cfg(cfg):
    """Process cfg for evaluation"""
    assert cfg.task_name is not None, 'Please set task_name'

    # modify config
    with open_dict(cfg):
        # Log code git commit
        try:
            cfg.commit = commit = get_git_commit()
            log.info(f"Commit: {commit}")
        except:
            pass
        # remove training related cfg to avoid dataset initialization
        train_keys = ['callbacks', 'data.train_dataset', 'data.val_dataset', 'data.val_gen_dataset']
        for key in train_keys:
            replace_dot_key_val(cfg, dot_key=key, replace_to=None, inplace=True, ignore_error=True)
        # turn off loggers, notes, tags
        cfg.logger = None
        cfg.notes = None
        cfg.tags = None 
        # print cfg manually later
        cfg.extras.print_config = False
    
    return cfg


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    cfg = clean_cfg(cfg)

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    hydra_utils.extras(cfg)

    # Manually save and print cfg
    hydra_utils.print_config_tree(cfg, resolve=True, save_to_file=False)
    save_cfg = rank_zero_only(OmegaConf.save)
    runtime_output_dir = str(HydraConfig.get().runtime.output_dir)
    save_cfg(OmegaConf.to_container(cfg, resolve=True), f"{runtime_output_dir}/eval.yaml")

    evaluate(cfg)


if __name__ == "__main__":
    main()
