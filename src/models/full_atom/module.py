from typing import Dict, Any

import torch
from lightning import LightningModule, Fabric, seed_everything
from torchmetrics import MeanMetric, MinMetric
from openfold.np.protein import Protein
from openfold.np import protein
from src.utils import hydra_utils
from src.analysis.eval import eval_gen_conf
import os
import numpy as np
import torch.distributed as dist

logger = hydra_utils.get_pylogger(__name__)


class FullAtomLitModule(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        score_network,
        diffuser,
        optimizer,
        scheduler,
        lr_warmup_steps: int,
        val_gen_every_n_epochs: int,
        output_dir: str,
        log_loss_name=[
            "total",
            "rot",
            "trans",
            "bb_coords",
            "bb_dist_map",
            "torsion",
            "fape",
        ],
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["score_network"])
        self.output_dir = output_dir
        self.val_output_dir = f'{output_dir}/val_gen/'
        self.test_output_dir = f'{output_dir}/test_gen/'
        self.score_network = score_network
        self.score_network.diffuser = diffuser
        self.loss_name = log_loss_name

        for split in ["train", "val"]:
            for loss_name in self.loss_name:
                setattr(self, f"{split}_{loss_name}", MeanMetric())
        self.best_val_total = MinMetric()

    def setup(self, stage: str) -> None:
        # broadcast output_dir from rank 0
        fabric = Fabric()
        fabric.launch()
        self.output_dir = fabric.broadcast(self.output_dir, src=0)
        self.val_output_dir = fabric.broadcast(self.val_output_dir, src=0)
        self.test_output_dir = fabric.broadcast(self.test_output_dir, src=0)
    def forward(
        self, 
        batch: Dict[str, Any]
    ):
        return self.score_network(**batch)

    def sampling(self, batch, output_dir):

        output = self.score_network.reverse_sample(**batch)

        for i in range(batch["aatype"].shape[0]):
            padding_mask = batch["padding_mask"][i]
            aatype = batch["aatype"][i][padding_mask].cpu().numpy()
            atom37 = output["atom37"][i][padding_mask].cpu().numpy()
            atom37_mask = output["atom37_mask"][i][padding_mask].cpu().numpy()
            res_idx = np.arange(aatype.shape[0])

            chain_name = batch["chain_name"][i]
            output_name = batch["output_name"][i]
            folder_path = f"{output_dir}/{chain_name}/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path,exist_ok=True)

            gen_protein = Protein(
                aatype=aatype,
                atom_positions=atom37,
                atom_mask=atom37_mask,
                residue_index=res_idx + 1,
                chain_index=np.zeros_like(aatype),
                b_factors=np.zeros_like(atom37_mask),
            )
            with open(f"{folder_path}/{output_name}", "w") as fp:
                fp.write(protein.to_pdb(gen_protein))

    def on_train_start(self):
        """Called at the beginning of training after sanity check.

        NOTE: by default lightning executes validation step sanity checks before training starts,
              so it's worth to make sure validation metrics don't store results from these checks
        """
        local_rank = int(dist.get_rank())
        seed_everything(42 + local_rank, workers=True)

        for split in ["train", "val"]:
            for loss_name in self.loss_name:
                getattr(self, f"{split}_{loss_name}").reset()

        self.best_val_total.reset()

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, aux_info = self.forward(batch)

        for loss_name in self.loss_name:
            getattr(self, f"train_{loss_name}").update(aux_info[loss_name])
        return loss

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ):
        if dataloader_idx == 0:
            loss, aux_info = self.forward(batch)
            for loss_name in self.loss_name:
                getattr(self, f"val_{loss_name}").update(aux_info[loss_name])
            return loss

        elif (not self.trainer.sanity_checking) and (
            (self.current_epoch + 1) % self.hparams.val_gen_every_n_epochs == 0 # type: ignore
        ):
            # inference on val_gen dataset
            output_dir = f"{self.val_output_dir}/epoch{self.current_epoch}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            self.sampling(batch, output_dir)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, sync_dist=True)
        log_info = f"Current epoch: {self.current_epoch:d}, step: {self.global_step:d}, lr: {lr:.8f}, "

        for split in ["train", "val"]:
            for loss_name in self.loss_name:
                epoch_loss = getattr(self, f"{split}_{loss_name}").compute()
                self.log(f"{split}/{loss_name}_loss", epoch_loss, sync_dist=True)
                log_info += f"{split}/{loss_name}_loss: {epoch_loss:.8f}, "
                getattr(self, f"{split}_{loss_name}").reset()
                if split == "val" and loss_name == "total":
                    self.best_val_total.update(epoch_loss)
                    self.log(
                        "val/best_val_total_loss",
                        self.best_val_total.compute(),
                        sync_dist=True,
                    )
        logger.info(log_info)
        
        dist.barrier()
        # evaluate val_gen results
        output_dir = f"{self.val_output_dir}/epoch{self.current_epoch}"
        if self.trainer.is_global_zero and os.path.exists(output_dir):
            try:
                log_stats, log_dist = eval_gen_conf(
                    output_root=output_dir,
                    csv_fpath=self.trainer.datamodule.val_gen_dataset.csv_path,
                    ref_root=self.trainer.datamodule.val_gen_dataset.data_dir,
                    num_samples=self.trainer.datamodule.val_gen_dataset.num_samples,
                    n_proc=1,
                )
                log_stats = {
                    f"val_gen/cameo/{name}": val for name, val in log_stats.items()
                }
                self.log_dict(log_stats, rank_zero_only=True, sync_dist=True)
            except:
                logger.warn("Failed to evaluate val_gen results, skip..")

        torch.cuda.empty_cache()

    def on_test_start(self):
        local_rank = int(dist.get_rank())
        seed_everything(42 + local_rank, workers=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        # inference on test_gen dataset
        self.sampling(batch, os.path.join(self.output_dir, self.test_output_dir))
    def on_test_end(self):
        dist.barrier()
        

    @property
    def is_epoch_based(self):
        """If the training is epoch-based or iteration-based."""
        return type(self.trainer.val_check_interval) == float and self.trainer.val_check_interval <= 1.0

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        if self.hparams.optimizer.func.__name__ == "DeepSpeedCPUAdam":
            optimizer = self.hparams.optimizer(
                model_params=self.trainer.model.parameters()
            )
        else:
            optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    'monitor': 'val/total_loss',
                    'interval': 'epoch' if self.is_epoch_based else 'step',
                    'strict': True,
                    'frequency': self.trainer.check_val_every_n_epoch if self.is_epoch_based else int(self.trainer.val_check_interval), # adjust lr_scheduler everytime run evaluation
                },
            }
        return {"optimizer": optimizer}

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps: # type: ignore
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams.lr_warmup_steps # type: ignore
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.optimizer.keywords["lr"]
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()
