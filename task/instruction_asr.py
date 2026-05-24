from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from dataset.preprocessor import CommonPreprocessor
from task.base import InstructionIterTask
from train.config import TrainConfig


class InstructionTask(InstructionIterTask):
    @classmethod
    def build_iter_options(cls, config, distributed_option, mode: str):
        iter_options = super().build_iter_options(config, distributed_option, mode)
        iter_options.preprocess_fn = cls.build_preprocess_fn(
            config,
            train=(mode == "train"),
        )
        return iter_options

    @classmethod
    def build_preprocess_fn(
        cls, config: TrainConfig, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        return CommonPreprocessor(
            train=train,
            token_type=getattr(config, "token_type", None),
            token_list=getattr(config, "token_list", None),
            bpemodel=(
                getattr(config, "bpemodel", None)
                if getattr(config, "token_type", None) not in {"huggingface", "hf"}
                else (getattr(config, "bpemodel", None) or getattr(config, "token_list", None))
            ),
            non_linguistic_symbols=getattr(config, "non_linguistic_symbols", None),
            text_cleaner=getattr(config, "cleaner", None),
            g2p_type=getattr(config, "g2p", None),
            aux_task_names=None,
            text_name="answer",
            huggingface_max_length=getattr(config.model.llm_decoder, "max_target_length", None),
        )


InstructionAsrTask = InstructionTask
