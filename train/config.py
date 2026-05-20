from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _check_unknown_keys(data: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown_keys = sorted(set(data) - allowed)
    if unknown_keys:
        raise ValueError(f"Unknown keys in {context}: {unknown_keys}")


@dataclass
class OptimizerConfig:
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    lr: float = 1e-5
    warmup_steps: int = 20
    stable_steps: int = 0
    min_lr_ratio: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OptimizerConfig":
        payload = dict(data or {})
        _check_unknown_keys(payload, set(cls.__dataclass_fields__), "optimizer_conf")
        return cls(**payload)


@dataclass
class AdapterConfig:
    expert_ffn_dim: int = 1280
    num_experts: int = 8
    top_k: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AdapterConfig":
        payload = dict(data or {})
        _check_unknown_keys(payload, set(cls.__dataclass_fields__), "model_conf.adapter_conf")
        return cls(**payload)


@dataclass
class QFormerConfig:
    qformer_model_name: str = "bert-base-uncased"
    num_query_token: int = 32
    num_hidden_layers: int = 2
    cross_attention_freq: int = 1
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "QFormerConfig":
        payload = dict(data or {})
        _check_unknown_keys(payload, set(cls.__dataclass_fields__), "model_conf.qformer_conf")
        return cls(**payload)


@dataclass
class ClassifierConfig:
    classifier_hidden_dim: int = 1280
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ClassifierConfig":
        payload = dict(data or {})
        _check_unknown_keys(payload, set(cls.__dataclass_fields__), "model_conf.classfier_conf")
        return cls(**payload)


@dataclass
class AsrDecoderConfig:
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AsrDecoderConfig":
        payload = dict(data or {})
        _check_unknown_keys(payload, set(cls.__dataclass_fields__), "model_conf.asr_decoder_conf")
        return cls(**payload)


@dataclass
class LlmDecoderConfig:
    llm_repo: str = "Qwen/Qwen3-1.7B"
    projector_hidden_dim: int = 2048
    projector_num_layers: int = 2
    dropout: float = 0.1
    max_target_length: int = 256
    prompt_text: str = "Transcribe the following speech:"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LlmDecoderConfig":
        payload = dict(data or {})
        _check_unknown_keys(payload, set(cls.__dataclass_fields__), "model_conf.llm_decoder_conf")
        return cls(**payload)


@dataclass
class ModelConfig:
    num_classes: int = 4
    aux_loss_weight: float = 0.5
    model_repo: str = "moonshotai/Kimi-Audio-7B-Instruct"
    tokenizer_repo: str = "THUDM/glm-4-voice-tokenizer"
    sample_rate: int = 16000
    adapter_type: str = "moe"
    decoder_type: str = "ntp"
    freeze_frontend: bool = False
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    qformer: QFormerConfig = field(default_factory=QFormerConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    asr_decoder: AsrDecoderConfig = field(default_factory=AsrDecoderConfig)
    llm_decoder: LlmDecoderConfig = field(default_factory=LlmDecoderConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ModelConfig":
        payload = dict(data or {})
        adapter_conf = payload.pop("adapter_conf", None)
        qformer_conf = payload.pop("qformer_conf", None)
        classifier_conf = payload.pop("classfier_conf", payload.pop("classifier_conf", None))
        asr_decoder_conf = payload.pop("asr_decoder_conf", None)
        llm_decoder_conf = payload.pop("llm_decoder_conf", None)
        _check_unknown_keys(
            payload,
            {
                "num_classes",
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
            qformer=QFormerConfig.from_dict(qformer_conf),
            classifier=ClassifierConfig.from_dict(classifier_conf),
            asr_decoder=AsrDecoderConfig.from_dict(asr_decoder_conf),
            llm_decoder=LlmDecoderConfig.from_dict(llm_decoder_conf),
        )


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
    data_type: list[str] = field(default_factory=lambda: ["sound", "chime4_label"])
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
    model_name: str = "moeclassifier"
    token_type: str | None = None
    token_list: str = ""
    non_linguistic_symbols: str | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    seed: int = 314562
    epoch: int = 10
    patience: int = 100
    log_every_n_steps: int = 1
    strategy: str = "ddp_find_unused_parameters_true"
    task: str = "classify"
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
            optimizer=OptimizerConfig.from_dict(getattr(args, "optimizer_conf", None)),
            dataset=DatasetConfig.from_dict(getattr(args, "dataset_conf", None)),
            seed=getattr(args, "seed", defaults.seed),
            epoch=getattr(args, "epoch", defaults.epoch),
            patience=getattr(args, "patience", defaults.patience),
            log_every_n_steps=getattr(args, "log_every_n_steps", defaults.log_every_n_steps),
            strategy=getattr(args, "strategy", defaults.strategy),
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
        model_dict["qformer_conf"] = model_dict.pop("qformer")
        model_dict["classfier_conf"] = model_dict.pop("classifier")
        model_dict["asr_decoder_conf"] = model_dict.pop("asr_decoder")
        model_dict["llm_decoder_conf"] = model_dict.pop("llm_decoder")
        return {
            "model": self.model_name,
            "model_conf": model_dict,
            "optimizer_conf": asdict(self.optimizer),
            "dataset_conf": asdict(self.dataset),
            "token_type": self.token_type,
            "token_list": self.token_list,
            "non_linguistic_symbols": self.non_linguistic_symbols,
            "seed": self.seed,
            "epoch": self.epoch,
            "patience": self.patience,
            "log_every_n_steps": self.log_every_n_steps,
            "strategy": self.strategy,
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
