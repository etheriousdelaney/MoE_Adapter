import lightning as L
import yaml
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from task.asr import AsrTask
from task.classify import ClassifyTask
from task.instruction_asr import InstructionAsrTask
from train.config import TrainConfig
from dataset.distributed_utils import DistributedOption
from train.model_factory import build_model_from_config, model_choices

task_choices = {
    "classify": ClassifyTask,
    "asr": AsrTask,
    "instruction_asr": InstructionAsrTask,
}

class LitModel(L.LightningModule):
    def __init__(
            self,
            config: TrainConfig):
        super().__init__()
        self.save_hyperparameters(
            {
                "task": config.task,
                "model_name": config.model_name,
                "exp_tag": config.exp_tag,
                "output_dir": config.output_dir,
            }
        )
        self.config = config
        self.task_class = task_choices[config.task]
        self.optimizer_config = config.optimizer
        self.model = build_model_from_config(config=config)

        if self.global_rank == 0:
            # Save config to make it compatible with ESPnet inference
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            with (Path(config.output_dir) / "config.yaml").open(
                "w", encoding="utf-8"
            ) as f:
                yaml.dump(
                    config.to_yaml_dict(),
                    f,
                    indent=4,
                    sort_keys=False,
                    allow_unicode=True,
                )
                
    def _step(
            self,
            batch,
            batch_idx,
            mode
    ):
        outputs = self.model(batch)
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if not hasattr(value, "detach"):
                    continue
                if getattr(value, "ndim", 0) != 0:
                    continue
                self.log(
                    f"{mode}/{key}",
                    value,
                    on_step=(mode == "train"),
                    on_epoch=True,
                    prog_bar=(key in {"loss", "acc", "ctc_loss", "lm_loss"}),
                    logger=True,
                    sync_dist=(mode != "train"),
                    batch_size=len(batch[0]) if isinstance(batch, tuple) else None,
                )
            return outputs

        self.log(
            f"{mode}/loss",
            outputs,
            on_step=(mode == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=(mode != "train"),
            batch_size=len(batch[0]) if isinstance(batch, tuple) else None,
        )
        return {"loss": outputs}
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="valid") 

    def configure_optimizers(self):
        trainable_parameters = [param for param in self.model.parameters() if param.requires_grad]
        if len(trainable_parameters) == 0:
            raise RuntimeError("No trainable parameters found for optimizer setup")
        optimizer = AdamW(
            trainable_parameters,
            lr=self.optimizer_config.lr,
            betas=(self.optimizer_config.adam_beta1, self.optimizer_config.adam_beta2),
            foreach=False
        )

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = max(0, int(self.optimizer_config.warmup_steps))
        stable_steps = max(0, int(self.optimizer_config.stable_steps))
        min_lr_ratio = float(self.optimizer_config.min_lr_ratio)

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps)

            if current_step < warmup_steps + stable_steps:
                return 1.0

            decay_start = warmup_steps + stable_steps
            decay_steps = max(1, total_steps - decay_start)
            decay_progress = min(1.0, max(0.0, (current_step - decay_start) / decay_steps))
            return max(min_lr_ratio, 1.0 - decay_progress * (1.0 - min_lr_ratio))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        train_iter_factory = self.task_class.build_iter_factory(
            config=self.config,
            distributed_option=DistributedOption(distributed=True),
            mode="train",
        )
        return train_iter_factory.build_iter(epoch=self.current_epoch)

    def val_dataloader(self):
        valid_iter_factory = self.task_class.build_iter_factory(
            config=self.config,
            distributed_option=DistributedOption(distributed=True),
            mode="valid",
        )
        return valid_iter_factory.build_iter(epoch=self.current_epoch)
