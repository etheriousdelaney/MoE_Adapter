from typeguard import typechecked
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, Collection
from iterators.abs_iter_factory import AbsIterFactory
from dataset.distributed_utils import DistributedOption
from dataclasses import dataclass
import torch
import numpy as np
from loguru import logger
from samplers.build_batch_sampler import build_batch_sampler
from iterators.sequence_iter_factory import SequenceIterFactory
import itertools
from dataset.collate_fn import CommonCollateFn
from dataset.preprocessor import CommonPreprocessor
from train.config import TrainConfig
from dataset.dataset import Dataset

@dataclass
class IteratorOptions:
    preprocess_fn: callable
    collate_fn: callable
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

class ClassifyTask():
    def __init__(self):
        pass

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
            for k, v in kwargs.items():
                setattr(iter_options, k, v)

        return cls.build_sequence_iter_factory(
                config=config,
                iter_options=iter_options,
                mode=mode,
            )
    
    @classmethod
    @typechecked
    def build_sequence_iter_factory(
        cls, config: TrainConfig, iter_options: IteratorOptions, mode: str
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
            )
        )

        batches = list(batch_sampler)
        if iter_options.num_batches is not None:
            batches = batches[: iter_options.num_batches]

        bs_list = [len(batch) for batch in batches]

        logger.info(f"[{mode}] Batch sampler: {batch_sampler}")
        logger.info(
            f"[{mode}] mini-batch sizes summary: N-batch={len(bs_list)}, "
            f"mean={np.mean(bs_list):.1f}, min={np.min(bs_list)}, max={np.max(bs_list)}"
        )

        # Shard mini-batches for distributed training
        if iter_options.distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            for batch in batches:
                if len(batch) < world_size:
                    raise RuntimeError(
                        f"The batch-size must be equal or more than world_size: "
                        f"{len(batch)} < {world_size}"
                    )
            batches = [batch[rank::world_size] for batch in batches]

        # Build dataset after sharding to reduce memory usage
        # This is very helpful for large-scale training
        dataset = cls.build_dataset(
            config,
            iter_options,
            set(itertools.chain(*batches)),
        )
        logger.info(f"[{mode}] dataset:\n{dataset}")

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
    ):
        dataset_config = config.dataset
        if mode == "train":
            preprocess_fn = None
            collate_fn = cls.build_collate_fn(config, train=True)
            data = dataset_config.train_data
            shape_files = dataset_config.train_shape_file
            batch_size = dataset_config.batch_size
            batch_bins = dataset_config.batch_bins
            batch_type = dataset_config.batch_type
            max_cache_size = dataset_config.max_cache_size
            max_cache_fd = dataset_config.max_cache_fd
            allow_multi_rates = dataset_config.allow_multi_rates
            distributed = distributed_option.distributed
            num_batches = None
            num_iters_per_epoch = dataset_config.num_iters_per_epoch
            train = True

        elif mode == "valid":
            preprocess_fn = None
            collate_fn = cls.build_collate_fn(config, train=False)
            data = dataset_config.valid_data
            shape_files = dataset_config.valid_shape_file

            batch_size = dataset_config.batch_size
            batch_bins = dataset_config.batch_bins
            batch_type = dataset_config.batch_type
            
            max_cache_size = 0.05 * dataset_config.max_cache_size
            max_cache_fd = dataset_config.max_cache_fd
            allow_multi_rates = dataset_config.allow_multi_rates
            distributed = distributed_option.distributed
            num_batches = None
            num_iters_per_epoch = None
            train = False

        else:
            raise NotImplementedError(f"mode={mode}")

        return IteratorOptions(
            preprocess_fn=preprocess_fn,
            collate_fn=collate_fn,
            data=data,
            shape_files=shape_files,
            batch_type=batch_type,
            batch_size=batch_size,
            batch_bins=batch_bins,
            num_batches=num_batches,
            max_cache_size=max_cache_size,
            max_cache_fd=max_cache_fd,
            allow_multi_rates=allow_multi_rates,
            distributed=distributed,
            num_iters_per_epoch=num_iters_per_epoch,
            train=train,
        )
    
    @classmethod
    @typechecked
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
            aux_task_names=getattr(config, "aux_ctc_tasks", None),
            huggingface_max_length=getattr(config.model.llm_decoder, "max_target_length", None),
        )
    
    @classmethod
    @typechecked
    def build_dataset(
        cls,
        config: TrainConfig,
        iter_options: IteratorOptions,
        keys_to_load: Optional[Set[Union[int, str]]] = None,
    ):
        dataset_config = config.dataset
        dataset_class = Dataset
        dataset = dataset_class(
            iter_options.data,
            float_dtype=dataset_config.train_dtype,
            preprocess=iter_options.preprocess_fn,
            max_cache_size=iter_options.max_cache_size,
            max_cache_fd=iter_options.max_cache_fd,
            allow_multi_rates=iter_options.allow_multi_rates,
            keys_to_load=keys_to_load,
            data_type=dataset_config.data_type
        )
        return dataset
    
    @classmethod
    @typechecked
    def build_collate_fn(cls, config: TrainConfig, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)
