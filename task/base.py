from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Collection, Optional

import numpy as np
import torch
from loguru import logger
from typeguard import typechecked

from dataset.collate_fn import CommonCollateFn
from dataset.dataset import Dataset
from dataset.distributed_utils import DistributedOption
from iterators.abs_iter_factory import AbsIterFactory
from iterators.sequence_iter_factory import SequenceIterFactory
from samplers.build_batch_sampler import build_batch_sampler
from train.config import TrainConfig


@dataclass
class IteratorOptions:
    preprocess_fn: Callable | None
    collate_fn: Callable
    data: str
    shape_files: list
    batch_size: int
    batch_bins: int
    batch_type: str
    max_cache_size: float
    max_cache_fd: int
    allow_multi_rates: bool
    distributed: bool
    num_batches: Optional[int]
    num_iters_per_epoch: Optional[int]
    train: bool
    message_file: str
    instruction_source: str
    instruction_tasks: list


class InstructionIterTask:
    @classmethod
    @typechecked
    def build_iter_factory(
        cls,
        config: TrainConfig,
        distributed_option: DistributedOption,
        mode: str,
        kwargs: Optional[dict] = None,
    ) -> AbsIterFactory:
        iter_options = cls.build_iter_options(config, distributed_option, mode)
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(iter_options, key, value)
        return cls.build_sequence_iter_factory(
            config=config,
            iter_options=iter_options,
            mode=mode,
        )

    @classmethod
    @typechecked
    def build_sequence_iter_factory(
        cls,
        config: TrainConfig,
        iter_options: IteratorOptions,
        mode: str,
    ) -> AbsIterFactory:
        dataset_config = config.dataset
        batch_sampler = build_batch_sampler(
            type=iter_options.batch_type,
            shape_files=iter_options.shape_files,
            fold_lengths=dataset_config.fold_lengths,
            batch_size=iter_options.batch_size,
            batch_bins=iter_options.batch_bins,
            sort_in_batch=dataset_config.sort_in_batch,
            sort_batch=dataset_config.sort_batch,
            drop_last=dataset_config.drop_last_iter,
            min_batch_size=(
                torch.distributed.get_world_size()
                if iter_options.distributed
                else config.min_batch_size
            ),
        )

        batches = list(batch_sampler)
        if iter_options.num_batches is not None:
            batches = batches[: iter_options.num_batches]

        batch_sizes = [len(batch) for batch in batches]
        logger.info("[{}] Batch sampler: {}", mode, batch_sampler)
        logger.info(
            "[{}] mini-batch sizes summary: N-batch={}, mean={:.1f}, min={}, max={}",
            mode,
            len(batch_sizes),
            np.mean(batch_sizes),
            np.min(batch_sizes),
            np.max(batch_sizes),
        )

        if iter_options.distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            for batch in batches:
                if len(batch) < world_size:
                    raise RuntimeError(
                        "The batch-size must be equal or more than world_size: "
                        f"{len(batch)} < {world_size}"
                    )
            batches = [batch[rank::world_size] for batch in batches]

        dataset = cls.build_dataset(
            config,
            iter_options,
            set(itertools.chain(*batches)),
        )
        logger.info("[{}] dataset:\n{}", mode, dataset)

        return SequenceIterFactory(
            dataset=dataset,
            batches=batches,
            seed=config.seed,
            num_iters_per_epoch=iter_options.num_iters_per_epoch,
            shuffle=iter_options.train,
            shuffle_within_batch=dataset_config.shuffle_within_batch,
            num_workers=dataset_config.num_workers,
            collate_fn=iter_options.collate_fn,
            pin_memory=config.ngpu > 0,
        )

    @classmethod
    def build_iter_options(
        cls,
        config: TrainConfig,
        distributed_option: DistributedOption,
        mode: str,
    ) -> IteratorOptions:
        dataset_config = config.dataset
        if mode == "train":
            data = dataset_config.train_data
            shape_files = dataset_config.train_shape_file
            max_cache_size = dataset_config.max_cache_size
            num_iters_per_epoch = dataset_config.num_iters_per_epoch
            train = True
            message_file = dataset_config.train_message_file
        elif mode == "valid":
            data = dataset_config.valid_data
            shape_files = dataset_config.valid_shape_file
            max_cache_size = 0.05 * dataset_config.max_cache_size
            num_iters_per_epoch = None
            train = False
            message_file = dataset_config.valid_message_file
        else:
            raise NotImplementedError(f"mode={mode}")

        return IteratorOptions(
            preprocess_fn=None,
            collate_fn=cls.build_collate_fn(config, train=train),
            data=data,
            shape_files=shape_files,
            batch_type=dataset_config.batch_type,
            batch_size=dataset_config.batch_size,
            batch_bins=dataset_config.batch_bins,
            num_batches=None,
            max_cache_size=max_cache_size,
            max_cache_fd=dataset_config.max_cache_fd,
            allow_multi_rates=dataset_config.allow_multi_rates,
            distributed=distributed_option.distributed,
            num_iters_per_epoch=num_iters_per_epoch,
            train=train,
            message_file=message_file,
            instruction_source=dataset_config.instruction_source,
            instruction_tasks=dataset_config.instruction_tasks,
        )

    @classmethod
    @typechecked
    def build_dataset(
        cls,
        config: TrainConfig,
        iter_options: IteratorOptions,
        keys_to_load: Optional[set[int | str]] = None,
    ) -> Dataset:
        dataset_config = config.dataset
        return Dataset(
            iter_options.data,
            float_dtype=dataset_config.train_dtype,
            preprocess=iter_options.preprocess_fn,
            max_cache_size=iter_options.max_cache_size,
            max_cache_fd=iter_options.max_cache_fd,
            allow_multi_rates=iter_options.allow_multi_rates,
            keys_to_load=keys_to_load,
            data_type=dataset_config.data_type,
            message_file=iter_options.message_file,
            instruction_source=iter_options.instruction_source,
            instruction_tasks=iter_options.instruction_tasks,
        )

    @classmethod
    @typechecked
    def build_collate_fn(
        cls,
        config: TrainConfig,
        train: bool,
    ) -> Callable[[Collection[tuple[str, dict[str, np.ndarray]]]], tuple[list[str], dict[str, torch.Tensor]]]:
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)
