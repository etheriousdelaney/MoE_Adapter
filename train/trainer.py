import argparse
import importlib
from datetime import datetime
from loguru import logger
import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from train.LitModel import LitModel
from pathlib import Path
from train.config import (
    AdapterConfig,
    DatasetConfig,
    LlmDecoderConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
)
from train.lightning_callbacks import AverageCheckpointsCallback, ExpertHeatmapCallback
from train.model_factory import model_choices
from train.shape_files import ensure_instruction_shape_files, ensure_speech_shape_files
from utils import config_argparse


def load_init_model_weights(lit_model: LitModel, init_model: str | Path) -> None:
    init_model = Path(init_model)
    if not init_model.is_file():
        raise FileNotFoundError(f"init_model not found: {init_model}")

    checkpoint = torch.load(init_model, map_location="cpu", weights_only=False)
    if init_model.suffix == ".ckpt":
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            raise KeyError(f"Lightning checkpoint has no state_dict: {init_model}")
        incompatible = lit_model.load_state_dict(state_dict, strict=True)
        logger.info(
            "Loaded init_model={} into LitModel. missing_keys={}, unexpected_keys={}",
            init_model,
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )
        return

    if init_model.suffix == ".pth":
        state_dict = checkpoint
        incompatible = lit_model.model.load_state_dict(state_dict, strict=True)
        logger.info(
            "Loaded init_model={} into inner model. missing_keys={}, unexpected_keys={}",
            init_model,
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )
        return

    raise ValueError(f"Unsupported init_model suffix: {init_model.suffix}. Expected .ckpt or .pth")

def get_date():
    now = datetime.now()
    return now.strftime("%Y_%-m_%-d_%H_%M")

def build_parser():
    """Create the base parser with task selection."""
    class ArgumentDefaultsRawTextHelpFormatter(
            argparse.RawTextHelpFormatter,
            argparse.ArgumentDefaultsHelpFormatter,
        ):
            pass
    parser = config_argparse.ArgumentParser(
            description="base parser",
            formatter_class=ArgumentDefaultsRawTextHelpFormatter,
        )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--exp_tag",
        type=str,
        default=get_date(),
    )
    parser.add_argument(
        "--output_dir",
        type=str, 
        default=None
    )
    parser.add_argument(
        "--init_model",
        type=str,
        default="",
        help="Initialize model weights from a .ckpt or .pth without resuming optimizer/callback state.",
    )
    model_choices.add_arguments(parser)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
    )
    parser.add_argument(
        "--optimizer_conf",
        default=dict(),
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="warmup_linear",
    )
    parser.add_argument(
        "--scheduler_conf",
        default=dict(),
    )
    parser.add_argument(
        "--dataset_conf",
        default=dict(),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=314562,
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help="Lightning precision, e.g. 32-true, bf16-mixed, 16-mixed.",
    )
    parser.add_argument(
        "--accum_grad",
        type=int,
        default=1,
        help="The number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=5.0,
        help="Gradient clipping value passed to Lightning Trainer.",
    )
    parser.add_argument(
        "--grad_clip_algorithm",
        type=str,
        default="norm",
        choices=("norm", "value"),
        help="Gradient clipping algorithm passed to Lightning Trainer.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        help=(
            "Lightning distributed strategy, e.g. "
            "ddp, fsdp, deepspeed_stage_2, deepspeed_stage_3."
        ),
    )
    parser.add_argument(
        "--strategy_conf",
        default=dict(),
    )
    parser.add_argument(
        "--task",
        type=str, 
        default=None
    )
    parser.add_argument(
        "--token_type",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--token_list",
        type=str,
        default="",
    )
    parser.add_argument(
        "--non_linguistic_symbols",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--best_model_criterion",
        action="append",
        nargs=3,
        metavar=("MONITOR", "MODE", "NBEST"),
        default=None,
    )
    return parser


def build_strategy(strategy: str, strategy_conf: dict):
    strategy_name = str(strategy)
    conf = dict(strategy_conf or {})
    if strategy_name == "ddp":
        ddp_comm_hook = conf.pop("ddp_comm_hook", None)
        if ddp_comm_hook is not None:
            ddp_comm_hook = getattr(default_hooks, str(ddp_comm_hook))
        return DDPStrategy(
            ddp_comm_hook=ddp_comm_hook,
            **conf,
        )

    if strategy_name == "fsdp":
        auto_wrap_policy = _resolve_policy_set(conf.pop("auto_wrap_policy", None))
        activation_checkpointing_policy = _resolve_policy_set(
            conf.pop("activation_checkpointing_policy", None)
        )
        return FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing_policy=activation_checkpointing_policy,
            **conf,
        )

    return strategy


def _resolve_policy_set(policy_names):
    if policy_names is None or len(policy_names) == 0:
        return None
    return {
        getattr(
            importlib.import_module(".".join(str(policy).split(".")[:-1])),
            str(policy).split(".")[-1],
        )
        for policy in policy_names
    }


def main():
    torch.serialization.add_safe_globals(
        [
            TrainConfig,
            ModelConfig,
            OptimizerConfig,
            SchedulerConfig,
            DatasetConfig,
            AdapterConfig,
            LlmDecoderConfig,
        ]
    )
    args, _ = build_parser().parse_known_args()
    config = TrainConfig.from_namespace(args)
    if any(name in config.dataset.data_type for name in ("audio_context", "answer")):
        ensure_instruction_shape_files(config)
    elif "sound" in config.dataset.data_type and "fused" not in config.dataset.data_type:
        ensure_speech_shape_files(config)
    checkpoint_dir = config.checkpoint_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(config)
    logger.info("=" * 80)
    logger.info(
        "Gradient accumulation: accum_grad={} micro-batches per optimizer step. "
        "Lightning global_step counts optimizer steps, not micro-batches.",
        config.accum_grad,
    )
    logger.info(
        "Gradient clipping: grad_clip={} grad_clip_algorithm={}",
        config.grad_clip,
        config.grad_clip_algorithm,
    )
    logger.info("=" * 80)
    L.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    lit_model = LitModel(config=config)
    init_model = getattr(args, "init_model", "")

    logger.info("=" * 80)
    logger.info(lit_model)

    last_ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last="link",
        filename="step{step}",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
        save_weights_only=False,
    )

    early_stop_monitor, early_stop_mode, _ = config.best_model_criterion[0]
    patience = EarlyStopping(
        monitor=str(early_stop_monitor),
        mode=str(early_stop_mode),
        patience=args.patience,
    )

    best_ckpt_callbacks = []
    for monitor, mode, nbest in config.best_model_criterion:
        best_ckpt_callbacks.append(
            ModelCheckpoint(
                save_top_k=int(nbest),
                monitor=monitor,
                mode=mode,
                dirpath=checkpoint_dir,
                save_last=False,
                filename="epoch{epoch}_step{step}_" + str(monitor).replace("/", "."),
                auto_insert_metric_name=False,
                save_on_train_epoch_end=False,
                save_weights_only=True,
                enable_version_counter=False,
            )
        )

    ave_ckpt_callback = AverageCheckpointsCallback(
        output_dir=checkpoint_dir,
        best_ckpt_callbacks=best_ckpt_callbacks,
    )
    lr_callback = LearningRateMonitor()
    heatmap_callback = None
    if getattr(lit_model.model, "supports_expert_heatmap", False):
        heatmap_callback = ExpertHeatmapCallback(
            output_dir=Path(config.output_dir) / "image",
        )

    loggers = []
    if config.use_tensorboard:
        loggers.append(
            TensorBoardLogger(
                save_dir=config.output_dir,
                name="lightning_logs",
            )
        )

    if config.use_wandb:
        wandb_latest_id = None
        latest_run = Path(config.output_dir) / "wandb" / "latest-run"
        if latest_run.exists():
            wandb_latest_id = latest_run.resolve().name.split("-")[-1]
        loggers.append(
            WandbLogger(
                project=config.wandb_project or "MoE_Adapter",
                name=config.wandb_name or config.exp_tag,
                save_dir=config.output_dir,
                version=wandb_latest_id,
            )
        )

    trainer = L.Trainer(
        reload_dataloaders_every_n_epochs=1,
        use_distributed_sampler=False,
        strategy=build_strategy(config.strategy, config.strategy_conf),
        accelerator="auto",
        devices="auto",
        max_epochs=config.epoch,
        precision=config.precision,
        accumulate_grad_batches=config.accum_grad,
        gradient_clip_val=config.grad_clip,
        gradient_clip_algorithm=config.grad_clip_algorithm,
        log_every_n_steps=config.log_every_n_steps,
        callbacks=[
            last_ckpt_callback,
            *best_ckpt_callbacks,
            ave_ckpt_callback,
            lr_callback,
            TQDMProgressBar(refresh_rate=config.log_every_n_steps),
            patience,
            *([heatmap_callback] if heatmap_callback is not None else []),
        ],
        logger=loggers if len(loggers) > 0 else None,
    )
    last_ckpt_path = checkpoint_dir / "last.ckpt"
    ckpt_path = str(last_ckpt_path) if last_ckpt_path.exists() else None
    if ckpt_path is None:
        if init_model:
            logger.info("resume_checkpoint=None init_model={}", init_model)
            load_init_model_weights(lit_model, init_model)
        else:
            logger.info("resume_checkpoint=None start_training_from_scratch")
    else:
        logger.info("resume_checkpoint={}", ckpt_path)
    
    
    trainer.fit(lit_model, ckpt_path=ckpt_path)
    

if __name__ == "__main__":
    main()
