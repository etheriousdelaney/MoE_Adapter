import lightning as L
import torch
import yaml
from pathlib import Path
from task.instruction import InstructionTask
from train.config import TrainConfig
from dataset.distributed_utils import DistributedOption
from train.model_factory import build_model_from_config


optim_classes = dict(
    adam=torch.optim.Adam,
    adamw=torch.optim.AdamW,
    sgd=torch.optim.SGD,
    adadelta=torch.optim.Adadelta,
    adagrad=torch.optim.Adagrad,
    adamax=torch.optim.Adamax,
    asgd=torch.optim.ASGD,
    lbfgs=torch.optim.LBFGS,
    rmsprop=torch.optim.RMSprop,
    rprop=torch.optim.Rprop,
    nadam=torch.optim.NAdam,
    radam=torch.optim.RAdam,
)

scheduler_classes = dict(
    lambdalr=torch.optim.lr_scheduler.LambdaLR,
    steplr=torch.optim.lr_scheduler.StepLR,
    multisteplr=torch.optim.lr_scheduler.MultiStepLR,
    exponentiallr=torch.optim.lr_scheduler.ExponentialLR,
    cosineannealinglr=torch.optim.lr_scheduler.CosineAnnealingLR,
    reducelronplateau=torch.optim.lr_scheduler.ReduceLROnPlateau,
)

task_choices = {
    "instruction": InstructionTask,
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
        self.scheduler_config = config.scheduler
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
            self._log_accumulation_state(batch, batch_idx, mode)
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
        self._log_accumulation_state(batch, batch_idx, mode)
        return {"loss": outputs}

    def _log_accumulation_state(self, batch, batch_idx: int, mode: str) -> None:
        if mode != "train":
            return
        accum_grad = max(1, int(self.config.accum_grad))
        accum_step = batch_idx % accum_grad + 1
        batch_size = len(batch[0]) if isinstance(batch, tuple) else None
        self.log(
            "train/accum_step",
            float(accum_step),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "train/accum_grad",
            float(accum_grad),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "train/optimizer_step",
            float(self.global_step),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="valid") 

    def configure_optimizers(self):
        trainable_parameters = [param for param in self.model.parameters() if param.requires_grad]
        if len(trainable_parameters) == 0:
            raise RuntimeError("No trainable parameters found for optimizer setup")
        optimizer_name = str(self.optimizer_config.type).lower()
        optim_class = optim_classes.get(optimizer_name)
        if optim_class is None:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_config.type}")
        optimizer = optim_class(
            trainable_parameters,
            **self.optimizer_config.to_kwargs(optimizer_name),
        )

        scheduler_name = str(self.scheduler_config.type).lower()
        if scheduler_name == "none":
            return optimizer

        scheduler = self._build_scheduler(optimizer, scheduler_name)
        interval = self.scheduler_config.interval
        if scheduler_name == "reducelronplateau" and interval == "step":
            interval = "epoch"
        scheduler_entry = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": int(self.scheduler_config.frequency),
        }
        if scheduler_name == "reducelronplateau":
            scheduler_entry["monitor"] = self.scheduler_config.monitor
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_entry,
        }

    def _build_scheduler(self, optimizer, scheduler_name: str):
        if scheduler_name == "warmup_linear":
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=self._warmup_linear_lambda(),
            )
        if scheduler_name == "lambdalr":
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=self._warmup_linear_lambda(),
            )
        if scheduler_name not in scheduler_classes:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_config.type}")
        kwargs = self.scheduler_config.to_kwargs()
        if scheduler_name == "cosineannealinglr":
            kwargs.setdefault("T_max", max(1, int(self.trainer.estimated_stepping_batches)))
        return scheduler_classes[scheduler_name](optimizer, **kwargs)

    def _warmup_linear_lambda(self):
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
        return lr_lambda

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
