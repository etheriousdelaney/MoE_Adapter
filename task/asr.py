from __future__ import annotations

from task.classify import ClassifyTask


class AsrTask(ClassifyTask):
    @classmethod
    def build_iter_options(cls, config, distributed_option, mode: str):
        iter_options = super().build_iter_options(config, distributed_option, mode)
        iter_options.preprocess_fn = cls.build_preprocess_fn(
            config,
            train=(mode == "train"),
        )
        return iter_options
