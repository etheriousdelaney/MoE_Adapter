from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _check_unknown_keys(data: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown_keys = sorted(set(data) - allowed)
    if unknown_keys:
        raise ValueError(f"Unknown keys in {context}: {unknown_keys}")


def _with_type(data: dict[str, Any] | None, type_name: str) -> dict[str, Any]:
    payload = dict(data or {})
    payload.setdefault("type", type_name)
    return payload


def _asdict_flat_extra(obj) -> dict[str, Any]:
    payload = asdict(obj)
    extra = payload.pop("extra", {}) or {}
    payload.update(extra)
    return payload


@dataclass
class OptimizerConfig:
    type: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    lr: float = 1e-5
    weight_decay: float = 0.0
    foreach: bool | None = False
    warmup_steps: int = 20
    stable_steps: int = 0
    min_lr_ratio: float = 0.1
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OptimizerConfig":
        payload = dict(data or {})
        known = {}
        for key in set(cls.__dataclass_fields__) - {"extra"}:
            if key in payload:
                known[key] = payload.pop(key)
        extra = dict(known.pop("extra", {}))
        extra.update(payload)
        return cls(**known, extra=extra)

    def to_kwargs(self, optimizer_name: str) -> dict[str, Any]:
        optimizer_name = optimizer_name.lower()
        kwargs = dict(self.extra)
        kwargs.setdefault("lr", self.lr)
        if optimizer_name not in {"rprop", "lbfgs"}:
            kwargs.setdefault("weight_decay", self.weight_decay)
        if optimizer_name in {"adam", "adamw", "adamax", "nadam", "radam"}:
            kwargs.setdefault("betas", (self.adam_beta1, self.adam_beta2))
        if self.foreach is not None and optimizer_name in {
            "adam",
            "adamw",
            "adamax",
            "nadam",
            "radam",
            "sgd",
        }:
            kwargs.setdefault("foreach", self.foreach)
        return kwargs


@dataclass
class AdapterConfig:
    expert_ffn_dim: int = 1280
    num_experts: int = 8
    top_k: int = 2
    qformer_model_name: str = "bert-base-uncased"
    num_query_token: int = 32
    num_hidden_layers: int = 2
    cross_attention_freq: int = 1
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AdapterConfig":
        payload = dict(data or {})
        _check_unknown_keys(payload, set(cls.__dataclass_fields__), "model_conf.adapter_conf")
        return cls(**payload)


@dataclass
class LlmDecoderConfig:
    llm_repo: str = "Qwen/Qwen3-1.7B"
    projector_hidden_dim: int = 2048
    projector_num_layers: int = 2
    dropout: float = 0.1
    max_target_length: int = 256
    prompt_text: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LlmDecoderConfig":
        payload = dict(data or {})
        _check_unknown_keys(payload, set(cls.__dataclass_fields__), "model_conf.llm_decoder_conf")
        return cls(**payload)


@dataclass
class ModelConfig:
    aux_loss_weight: float = 0.5
    model_repo: str = "moonshotai/Kimi-Audio-7B-Instruct"
    tokenizer_repo: str = "THUDM/glm-4-voice-tokenizer"
    sample_rate: int = 16000
    adapter_type: str = "moe"
    decoder_type: str = "qwen"
    freeze_frontend: bool = False
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    llm_decoder: LlmDecoderConfig = field(default_factory=LlmDecoderConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ModelConfig":
        payload = dict(data or {})
        adapter_conf = payload.pop("adapter_conf", None)
        llm_decoder_conf = payload.pop("llm_decoder_conf", None)
        _check_unknown_keys(
            payload,
            {
                "aux_loss_weight",
                "model_repo",
                "tokenizer_repo",
                "sample_rate",
                "adapter_type",
                "decoder_type",
                "freeze_frontend",
            },
            "model_conf",
        )
        return cls(
            **payload,
            adapter=AdapterConfig.from_dict(adapter_conf),
            llm_decoder=LlmDecoderConfig.from_dict(llm_decoder_conf),
        )


@dataclass
class SchedulerConfig:
    type: str = "warmup_linear"
    interval: str = "step"
    frequency: int = 1
    monitor: str = "valid/loss"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SchedulerConfig":
        payload = dict(data or {})
        known = {}
        for key in set(cls.__dataclass_fields__) - {"extra"}:
            if key in payload:
                known[key] = payload.pop(key)
        extra = dict(known.pop("extra", {}))
        extra.update(payload)
        return cls(**known, extra=extra)

    def to_kwargs(self) -> dict[str, Any]:
        return dict(self.extra)


@dataclass
class DatasetConfig:
    train_data: str = ""
    train_shape_file: list[str] = field(default_factory=list)
    valid_data: str = ""
    valid_shape_file: list[str] = field(default_factory=list)
    batch_bins: int = 1_000_000
    batch_size: int = 20
    batch_type: str = "folded"
    max_cache_size: float = 0.0
    max_cache_fd: int = 32
    num_iters_per_epoch: int | None = None
    allow_multi_rates: bool = False
    iterator_type: str = "sequence"
    fold_lengths: list[int] = field(default_factory=lambda: [80000, 150])
    sort_in_batch: str = "descending"
    sort_batch: str = "descending"
    drop_last_iter: bool = False
    train_dtype: str = "float32"
    shuffle_within_batch: bool = False
    num_workers: int = 0
    data_type: list[str] = field(default_factory=lambda: ["fused", "audio_context", "answer"])
    train_message_file: str = ""
    valid_message_file: str = ""
    instruction_source: str = "message_response"
    instruction_tasks: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DatasetConfig":
        payload = dict(data or {})
        _check_unknown_keys(payload, set(cls.__dataclass_fields__), "dataset_conf")
        return cls(**payload)


@dataclass
class TrainConfig:
    ngpu: int = 1
    exp_tag: str = ""
    output_dir: str = ""
    model_name: str = "qwen_audio_model"
    token_type: str | None = None
    token_list: str = ""
    non_linguistic_symbols: str | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    seed: int = 314562
    epoch: int = 10
    patience: int = 100
    log_every_n_steps: int = 1
    precision: str = "32-true"
    accum_grad: int = 1
    grad_clip: float = 5.0
    grad_clip_algorithm: str = "norm"
    strategy: str = "ddp_find_unused_parameters_true"
    strategy_conf: dict[str, Any] = field(default_factory=dict)
    task: str = "instruction"
    min_batch_size: int = 1
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: str = ""
    wandb_name: str = ""
    best_model_criterion: list[list[str | int]] = field(
        default_factory=lambda: [["valid/loss", "min", 3]]
    )

    @classmethod
    def from_namespace(cls, args: Namespace) -> "TrainConfig":
        defaults = cls()
        exp_tag = getattr(args, "exp_tag", defaults.exp_tag)
        output_dir = getattr(args, "output_dir", "") or f"exp/{exp_tag}"
        return cls(
            ngpu=getattr(args, "ngpu", defaults.ngpu),
            exp_tag=exp_tag,
            output_dir=output_dir,
            model_name=getattr(args, "model", defaults.model_name),
            token_type=getattr(args, "token_type", None),
            token_list=getattr(args, "token_list", ""),
            non_linguistic_symbols=getattr(args, "non_linguistic_symbols", None),
            model=ModelConfig.from_dict(getattr(args, "model_conf", None)),
            optimizer=OptimizerConfig.from_dict(
                _with_type(
                    getattr(args, "optimizer_conf", None),
                    getattr(args, "optimizer", defaults.optimizer.type),
                )
            ),
            scheduler=SchedulerConfig.from_dict(
                _with_type(
                    getattr(args, "scheduler_conf", None),
                    getattr(args, "scheduler", defaults.scheduler.type),
                )
            ),
            dataset=DatasetConfig.from_dict(getattr(args, "dataset_conf", None)),
            seed=getattr(args, "seed", defaults.seed),
            epoch=getattr(args, "epoch", defaults.epoch),
            patience=getattr(args, "patience", defaults.patience),
            log_every_n_steps=getattr(args, "log_every_n_steps", defaults.log_every_n_steps),
            precision=getattr(args, "precision", defaults.precision),
            accum_grad=getattr(args, "accum_grad", defaults.accum_grad),
            grad_clip=getattr(args, "grad_clip", defaults.grad_clip),
            grad_clip_algorithm=getattr(
                args,
                "grad_clip_algorithm",
                defaults.grad_clip_algorithm,
            ),
            strategy=getattr(args, "strategy", defaults.strategy),
            strategy_conf=dict(getattr(args, "strategy_conf", None) or {}),
            task=getattr(args, "task", defaults.task),
            min_batch_size=getattr(args, "min_batch_size", defaults.min_batch_size),
            use_tensorboard=getattr(args, "use_tensorboard", defaults.use_tensorboard),
            use_wandb=getattr(args, "use_wandb", defaults.use_wandb),
            wandb_project=getattr(args, "wandb_project", defaults.wandb_project),
            wandb_name=getattr(args, "wandb_name", defaults.wandb_name),
            best_model_criterion=_normalize_best_model_criterion(
                getattr(args, "best_model_criterion", None)
            ),
        )

    def checkpoint_dir(self) -> Path:
        return Path(self.output_dir) / "checkpoint"

    def to_yaml_dict(self) -> dict[str, Any]:
        model_dict = asdict(self.model)
        model_dict["adapter_conf"] = model_dict.pop("adapter")
        model_dict["llm_decoder_conf"] = model_dict.pop("llm_decoder")
        return {
            "model": self.model_name,
            "model_conf": model_dict,
            "optimizer": self.optimizer.type,
            "optimizer_conf": _asdict_flat_extra(self.optimizer),
            "scheduler": self.scheduler.type,
            "scheduler_conf": _asdict_flat_extra(self.scheduler),
            "dataset_conf": asdict(self.dataset),
            "token_type": self.token_type,
            "token_list": self.token_list,
            "non_linguistic_symbols": self.non_linguistic_symbols,
            "seed": self.seed,
            "epoch": self.epoch,
            "patience": self.patience,
            "log_every_n_steps": self.log_every_n_steps,
            "precision": self.precision,
            "accum_grad": self.accum_grad,
            "grad_clip": self.grad_clip,
            "grad_clip_algorithm": self.grad_clip_algorithm,
            "strategy": self.strategy,
            "strategy_conf": self.strategy_conf,
            "task": self.task,
            "ngpu": self.ngpu,
            "exp_tag": self.exp_tag,
            "output_dir": self.output_dir,
            "use_tensorboard": self.use_tensorboard,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "wandb_name": self.wandb_name,
            "best_model_criterion": self.best_model_criterion,
        }


def _normalize_best_model_criterion(
    criteria: list[list[str]] | list[tuple[str, str, str]] | None,
) -> list[list[str | int]]:
    if not criteria:
        return [["valid/loss", "min", 3]]

    normalized: list[list[str | int]] = []
    for criterion in criteria:
        if len(criterion) != 3:
            raise ValueError(f"best_model_criterion must have 3 values: {criterion}")
        monitor, mode, nbest = criterion
        normalized.append([str(monitor), str(mode), int(nbest)])
    return normalized
