import collections
import copy
import json
import logging
import numbers
from pathlib import Path
import re
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)
import humanfriendly
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typeguard import typechecked
from fileio.sound_scp import SoundScpReader
from fileio.read_text import read_2columns_text
from dataset.instruction_utils import (
    compose_instruction_sample_id,
    iter_instruction_records,
    strip_instruction_sample_id,
)
from utils.sized_dict import SizedDict

CHIME4_ENV_MAP = {
    "BUS": 0,
    "CAF": 1,
    "PED": 2,
    "STR": 3,
}

DEFAULT_AUDIO_SYSTEM_PROMPT = (
    "You are an audio understanding assistant. Answer the user's question using the audio "
    "between <start_audio> and <end_audio>."
)


def _extract_chime4_env(utt_id: str) -> str:
    parts = utt_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected CHiME4 utterance id format: {utt_id}")
    return parts[2].split(".")[0]


class AdapterForSoundScpReader(collections.abc.Mapping):
    def __init__(
        self,
        loader,
        dtype: Union[None, str] = None,
    ):
        self.loader = loader
        self.dtype = dtype
        self.rate = None

    def keys(self):
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        try:
            retval = self.loader[key]
        except KeyError:
            base_key = strip_instruction_sample_id(key)
            retval = self.loader[base_key]

        if isinstance(retval, tuple):
            assert len(retval) == 2, len(retval)
            if isinstance(retval[0], int) and isinstance(retval[1], np.ndarray):
                # sound scp case
                rate, array = retval
            elif isinstance(retval[1], int) and isinstance(retval[0], np.ndarray):
                # Extended ark format case
                array, rate = retval
            else:
                raise RuntimeError(
                    f"Unexpected type: {type(retval[0])}, {type(retval[1])}"
                )

            self.rate = rate
            # Multichannel wave fie
            # array: (NSample, Channel) or (Nsample)
            if self.dtype is not None:
                array = array.astype(self.dtype)

        else:
            # Normal ark case
            assert isinstance(retval, np.ndarray), type(retval)
            array = retval
            if self.dtype is not None:
                array = array.astype(self.dtype)

        assert isinstance(array, np.ndarray), type(array)
        return array


class AdapterForLabelScpReader(collections.abc.Mapping):
    @typechecked
    def __init__(self, loader: Mapping[str, Any], label_map: Optional[Dict[str, int]] = None):
        self.loader = loader
        self.label_map = label_map or CHIME4_ENV_MAP

    def keys(self):
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self.loader:
            raise KeyError(key)

        env = _extract_chime4_env(key)
        if env not in self.label_map:
            raise KeyError(f"Unknown CHiME4 environment '{env}' for utterance '{key}'")

        return np.array([self.label_map[env]], dtype=np.int64)


class AdapterForCategoryTextReader(collections.abc.Mapping):
    @typechecked
    def __init__(self, loader: Mapping[str, str], label_map: Optional[Dict[str, int]] = None):
        self.loader = loader
        self.label_map = label_map or CHIME4_ENV_MAP

    def keys(self):
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self.loader:
            raise KeyError(key)

        raw_value = self.loader[key].strip().upper()
        if raw_value in self.label_map:
            label_id = self.label_map[raw_value]
        else:
            label_id = int(raw_value)
        return np.array([label_id], dtype=np.int64)


class AdapterForTextReader(collections.abc.Mapping):
    def __init__(self, loader: Mapping[str, str]):
        self.loader = loader

    def keys(self):
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> str:
        if key in self.loader:
            return self.loader[key]
        base_key = strip_instruction_sample_id(key)
        return self.loader[base_key]


class AdapterForInstructionTextReader(collections.abc.Mapping):
    def __init__(self, loader: Mapping[str, str]):
        self.loader = loader

    def keys(self):
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> str:
        return self.loader[key]


class AdapterForFusedReader(collections.abc.Mapping):
    def __init__(self, loader: Mapping[str, str], dtype: Union[None, str] = None):
        self.loader = loader
        self.dtype = dtype

    def keys(self):
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self.loader:
            key = strip_instruction_sample_id(key)
        fused_path = self.loader[key]
        fused_tensor = torch.load(fused_path, map_location="cpu")
        if not isinstance(fused_tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor in {fused_path}, but got {type(fused_tensor)}")
        if fused_tensor.dtype == torch.bfloat16:
            fused_tensor = fused_tensor.float()
        array = fused_tensor.detach().cpu().numpy()
        if self.dtype is not None:
            array = array.astype(self.dtype)
        return array


def sound_loader(path, float_dtype=None, allow_multi_rates=False):
    # The file is as follows:
    #   utterance_id_A /some/where/a.wav
    #   utterance_id_B /some/where/a.flac

    # NOTE(kamo): SoundScpReader doesn't support pipe-fashion
    # like Kaldi e.g. "cat a.wav |".
    # NOTE(kamo): The audio signal is normalized to [-1,1] range.
    path = path + '/wav.scp'
    loader = SoundScpReader(path, dtype=float_dtype)

    # SoundScpReader.__getitem__() returns Tuple[int, ndarray],
    # but ndarray is desired, so Adapter class is inserted here
    return AdapterForSoundScpReader(loader)

def chime4_label_loader(path, keys_to_load=None):
    category_path = Path(path) / "utt2category"
    if category_path.exists():
        loader = read_2columns_text(category_path, keys_to_load=keys_to_load)
        return AdapterForCategoryTextReader(loader, label_map=CHIME4_ENV_MAP)

    wav_scp_path = Path(path) / "wav.scp"
    if wav_scp_path.exists():
        loader = SoundScpReader(str(wav_scp_path))
        return AdapterForLabelScpReader(loader, label_map=CHIME4_ENV_MAP)

    fused_scp_path = Path(path) / "fused.scp"
    if fused_scp_path.exists():
        loader = read_2columns_text(fused_scp_path, keys_to_load=keys_to_load)
        return AdapterForLabelScpReader(loader, label_map=CHIME4_ENV_MAP)

    text_path = Path(path) / "text"
    if text_path.exists():
        loader = read_2columns_text(text_path, keys_to_load=keys_to_load)
        return AdapterForLabelScpReader(loader, label_map=CHIME4_ENV_MAP)

    raise FileNotFoundError(f"Could not infer CHiME4 labels under {path}")


def text_loader(path, keys_to_load=None):
    text_path = path + "/text"
    loader = read_2columns_text(text_path, keys_to_load=keys_to_load)
    return AdapterForTextReader(loader)


def _normalize_instruction_keys(
    keys_to_load: Optional[Set[Union[str, int]]],
) -> Optional[Set[str]]:
    if keys_to_load is None:
        return None
    return {strip_instruction_sample_id(str(key)) for key in keys_to_load}


def _resolve_optional_data_file(path: str, file_name: str | None, default_name: str) -> Path:
    if file_name:
        file_path = Path(file_name)
        if not file_path.is_absolute():
            file_path = Path(file_name)
        return file_path
    return Path(path) / default_name


def _load_instruction_jsonl(
    path: str,
    field_name: str,
    keys_to_load: Optional[Set[Union[str, int]]] = None,
) -> AdapterForInstructionTextReader:
    response_jsonl_path = Path(path) / "response.jsonl"
    if not response_jsonl_path.exists():
        raise FileNotFoundError(f"response.jsonl not found: {response_jsonl_path}")

    sample_map: dict[str, str] = {}
    allowed_keys = None if keys_to_load is None else {str(key) for key in keys_to_load}
    for record in iter_instruction_records(response_jsonl_path):
        sample_id = record["sample_id"]
        if allowed_keys is not None and sample_id not in allowed_keys:
            continue
        sample_map[sample_id] = record[field_name]
    return AdapterForInstructionTextReader(sample_map)


def answer_loader(
    path,
    keys_to_load=None,
    instruction_source="message_response",
    instruction_tasks=None,
):
    if instruction_source == "task_specs":
        return _load_task_spec_field(
            path=path,
            field_name="answer",
            instruction_tasks=instruction_tasks,
            keys_to_load=keys_to_load,
        )
    return _load_instruction_jsonl(path, field_name="response", keys_to_load=keys_to_load)


def audio_context_loader(
    path,
    keys_to_load=None,
    message_file="",
    instruction_source="message_response",
    instruction_tasks=None,
):
    if instruction_source == "task_specs":
        return _load_task_spec_field(
            path=path,
            field_name="audio_context",
            instruction_tasks=instruction_tasks,
            keys_to_load=keys_to_load,
        )

    message_jsonl_path = _resolve_optional_data_file(path, message_file, "message.jsonl")
    if not message_jsonl_path.exists():
        raise FileNotFoundError(f"message.jsonl not found: {message_jsonl_path}")

    sample_map: dict[str, str] = {}
    allowed_keys = None if keys_to_load is None else {str(key) for key in keys_to_load}
    with message_jsonl_path.open("r", encoding="utf-8") as f:
        for record, line in zip(iter_instruction_records(Path(path) / "response.jsonl"), f):
            sample_id = record["sample_id"]
            if allowed_keys is not None and sample_id not in allowed_keys:
                continue
            payload = json.loads(line)
            sample_map[sample_id] = json.dumps(payload["messages"], ensure_ascii=False)
    return AdapterForInstructionTextReader(sample_map)


def _load_task_spec_field(
    path: str,
    field_name: str,
    instruction_tasks: Optional[List[dict[str, Any]]] = None,
    keys_to_load: Optional[Set[Union[str, int]]] = None,
) -> AdapterForInstructionTextReader:
    tasks = list(instruction_tasks or [])
    if not tasks:
        raise ValueError("instruction_source=task_specs requires non-empty instruction_tasks")

    field_maps: dict[str, Mapping[str, Any]] = {}
    answer_fields = {str(task["answer_field"]) for task in tasks if "answer_field" in task}
    for answer_field in answer_fields:
        field_maps[answer_field] = _load_task_answer_source(path, answer_field)

    allowed_keys = None if keys_to_load is None else {str(key) for key in keys_to_load}
    sample_map: dict[str, str] = {}
    for task_index, task in enumerate(tasks):
        if "prompt" not in task or "answer_field" not in task:
            raise ValueError(f"instruction task must contain prompt and answer_field: {task}")
        answer_field = str(task["answer_field"])
        source_map = field_maps[answer_field]
        for base_id in source_map:
            sample_id = compose_instruction_sample_id(str(base_id), task_index)
            if allowed_keys is not None and sample_id not in allowed_keys:
                continue
            if field_name == "audio_context":
                messages = _build_task_spec_messages(task)
                sample_map[sample_id] = json.dumps(messages, ensure_ascii=False)
            elif field_name == "answer":
                answer = _format_task_answer(source_map[base_id], task)
                sample_map[sample_id] = answer
            else:
                raise ValueError(f"Unsupported task-spec field: {field_name}")
    return AdapterForInstructionTextReader(sample_map)


def _load_task_answer_source(path: str, answer_field: str) -> Mapping[str, Any]:
    if answer_field == "chime4_label":
        return chime4_label_loader(path)
    field_path = Path(path) / answer_field
    if field_path.exists():
        return AdapterForTextReader(read_2columns_text(str(field_path)))
    metadata_path = Path(path) / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return _metadata_field_map(metadata, answer_field)
    raise FileNotFoundError(
        f"Cannot load instruction task answer_field={answer_field!r} under {path}"
    )


def _metadata_field_map(metadata: Any, answer_field: str) -> Mapping[str, str]:
    if isinstance(metadata, dict):
        items = metadata.items()
    elif isinstance(metadata, list):
        items = ((item.get("id") or item.get("utt_id") or item.get("key"), item) for item in metadata)
    else:
        raise ValueError("metadata.json must be a dict or list")
    output: dict[str, str] = {}
    for key, item in items:
        if key is None or not isinstance(item, dict) or answer_field not in item:
            continue
        output[str(key)] = str(item[answer_field])
    if not output:
        raise KeyError(f"metadata.json does not contain field {answer_field!r}")
    return output


def _build_task_spec_messages(task: Mapping[str, Any]) -> list[dict[str, str]]:
    system_prompt = str(task.get("system", DEFAULT_AUDIO_SYSTEM_PROMPT))
    prompt = str(task["prompt"]).strip()
    user_content = f"{prompt}\n<start_audio><end_audio>"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _format_task_answer(raw_value: Any, task: Mapping[str, Any]) -> str:
    label_map = task.get("label_map") or {}
    if isinstance(raw_value, np.ndarray):
        if raw_value.size == 1:
            raw_value = raw_value.reshape(-1)[0].item()
        else:
            raw_value = raw_value.tolist()
    lookup_keys = [raw_value, str(raw_value)]
    if isinstance(raw_value, (int, np.integer)):
        lookup_keys.append(int(raw_value))
    for key in lookup_keys:
        if key in label_map:
            return str(label_map[key])
    return str(raw_value)


def fused_loader(path, float_dtype=None, keys_to_load=None):
    fused_scp_path = path + "/fused.scp"
    loader = read_2columns_text(
        fused_scp_path,
        keys_to_load=_normalize_instruction_keys(keys_to_load),
    )
    return AdapterForFusedReader(loader, dtype=float_dtype)


DATA_TYPES = {
    "answer": dict(
        func=answer_loader,
        kwargs=["keys_to_load", "instruction_source", "instruction_tasks"],
    ),
    "audio_context": dict(
        func=audio_context_loader,
        kwargs=["keys_to_load", "message_file", "instruction_source", "instruction_tasks"],
    ),
    "fused": dict(
        func=fused_loader,
        kwargs=["float_dtype", "keys_to_load"],
    ),
    "sound": dict(
        func=sound_loader,
        kwargs=["float_dtype", "allow_multi_rates"],
    ),
}


class AbsDataset(Dataset, ABC):
    @abstractmethod
    def has_name(self, name) -> bool:
        raise NotImplementedError

    @abstractmethod
    def names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, uid) -> Tuple[Any, Dict[str, np.ndarray]]:
        raise NotImplementedError


class Dataset(AbsDataset):
    """Pytorch Dataset class for ESPNet.

    Examples:
        >>> dataset = ESPnetDataset([('wav.scp', 'input', 'sound'),
        ...                          ('token_int', 'output', 'text_int')],
        ...                         )
        ... uttid, data = dataset['uttid']
        {'input': per_utt_array, 'output': per_utt_array}
    """

    @typechecked
    def __init__(
        self,
        path_name: str,
        preprocess: Optional[
            Callable[[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]
        ] = None,
        float_dtype: str = "float32",
        int_dtype: str = "long",
        max_cache_size: Union[float, int, str] = 0.0,
        max_cache_fd: int = 0,
        allow_multi_rates: bool = False,
        keys_to_load: Optional[Set[Union[str, int]]] = None,
        data_type: List[str] = None,
        message_file: str = "",
        instruction_source: str = "message_response",
        instruction_tasks: Optional[List[dict[str, Any]]] = None,
    ):
        if len(path_name) == 0:
            raise ValueError(
                '1 or more elements are required for "path_name_type_list"'
            )
        if data_type is None:
            raise ValueError(
                'data type is required'
            )

        path_name = copy.deepcopy(path_name)
        self.preprocess = preprocess

        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        self.max_cache_fd = max_cache_fd
        # allow audios to have different sampling rates
        self.allow_multi_rates = allow_multi_rates

        self.loader_dict = {}
        self.debug_info = {}
        self.data_type = data_type
        self.message_file = message_file
        self.instruction_source = instruction_source
        self.instruction_tasks = instruction_tasks or []
        dataset_path = 'data/' + path_name
        if dataset_path in self.loader_dict:
            raise RuntimeError(f'"{dataset_path}" is duplicated for data-key')

        for type in self.data_type:                    
            loader = self._build_loader(dataset_path, loader_type=type, keys_to_load=keys_to_load)
            self.loader_dict[type] = loader
            self.debug_info[type] = dataset_path, type
        
        if len(self.loader_dict) == 0:
            raise RuntimeError(f"{dataset_path} has no samples")
        

        # TODO(kamo): Should check consistency of each utt-keys?
        if isinstance(max_cache_size, str):
            max_cache_size = humanfriendly.parse_size(max_cache_size)
        self.max_cache_size = max_cache_size
        if max_cache_size > 0:
            self.cache = SizedDict(shared=True)
        else:
            self.cache = None

    def _primary_loader(self):
        for preferred_name in ("answer", "audio_context"):
            if preferred_name in self.loader_dict:
                return self.loader_dict[preferred_name]
        return next(iter(self.loader_dict.values()))

    def _build_loader(
        self,
        path: str,
        loader_type: str,
        keys_to_load: Optional[Set[Union[str, int]]],
    ) -> Mapping[str, Union[np.ndarray, torch.Tensor, str, numbers.Number]]:
        """Helper function to instantiate Loader.

        Args:
            path:  The file path
            loader_type:  loader_type. fused, sound, audio_context, answer.
            keys_to_load:  The set of keys to load. If None, load all.
        """
        for key, dic in DATA_TYPES.items():
            if re.match(key, loader_type):
                kwargs = {}
                for key2 in dic["kwargs"]:
                    if key2 == "loader_type":
                        kwargs["loader_type"] = loader_type
                    elif key2 == "float_dtype":
                        kwargs["float_dtype"] = self.float_dtype
                    elif key2 == "int_dtype":
                        kwargs["int_dtype"] = self.int_dtype
                    elif key2 == "max_cache_fd":
                        kwargs["max_cache_fd"] = self.max_cache_fd
                    elif key2 == "allow_multi_rates":
                        kwargs["allow_multi_rates"] = self.allow_multi_rates
                    elif key2 == "keys_to_load":
                        kwargs["keys_to_load"] = keys_to_load
                    elif key2 == "message_file":
                        kwargs["message_file"] = self.message_file
                    elif key2 == "instruction_source":
                        kwargs["instruction_source"] = self.instruction_source
                    elif key2 == "instruction_tasks":
                        kwargs["instruction_tasks"] = self.instruction_tasks
                    else:
                        raise RuntimeError(f"Not implemented keyword argument: {key2}")

                func = dic["func"]
                try:
                    return func(path, **kwargs)
                except Exception:
                    if hasattr(func, "__name__"):
                        name = func.__name__
                    else:
                        name = str(func)
                    logging.error(f"An error happened with {name}({path})")
                    raise
        else:
            raise RuntimeError(f"Not supported: loader_type={loader_type}")

    def has_name(self, name) -> bool:
        return name in self.loader_dict

    def names(self) -> Tuple[str, ...]:
        return tuple(self.loader_dict)

    def __iter__(self):
        return iter(self._primary_loader())

    def __repr__(self):
        _mes = self.__class__.__name__
        _mes += "("
        for name, (path, _type) in self.debug_info.items():
            _mes += f'\n  {name}: {{"path": "{path}", "type": "{_type}"}}'
        _mes += f"\n  preprocess: {self.preprocess})"
        return _mes

    @typechecked
    def __getitem__(self, uid: Union[str, int]) -> Tuple[str, Dict[str, np.ndarray]]:
        # Change integer-id to string-id
        if isinstance(uid, int):
            d = self._primary_loader()
            uid = list(d)[uid]

        if self.cache is not None and uid in self.cache:
            data = self.cache[uid]
            return uid, data

        data = {}
        # 1. Load data from each loaders
        for name, loader in self.loader_dict.items():
            try:
                value = loader[uid]
                if isinstance(value, (list)):
                    value = np.array(value)
                if not isinstance(
                    value, (np.ndarray, torch.Tensor, str, numbers.Number, tuple)
                ):
                    raise TypeError(
                        (
                            "Must be ndarray, torch.Tensor, "
                            "str,  Number or tuple: {}".format(type(value))
                        )
                    )
            except Exception:
                path, _type = self.debug_info[name]
                logging.error(
                    f"Error happened with path={path}, type={_type}, id={uid}"
                )
                raise

            # torch.Tensor is converted to ndarray
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            elif isinstance(value, numbers.Number):
                value = np.array([value])
            data[name] = value

        # 2. [Option] Apply preprocessing
        #   e.g. espnet2.train.preprocessor:CommonPreprocessor
        if self.preprocess is not None:
            key_prefix = self.task + " " if hasattr(self, "task") else ""
            data = self.preprocess(key_prefix + uid, data)

        # 3. Force data-precision
        for name in data:
            value = data[name]
            if not isinstance(value, np.ndarray):
                raise RuntimeError(
                    f"All values must be converted to np.ndarray object "
                    f'by preprocessing, but "{name}" is still {type(value)}.'
                )

            # Cast to desired type
            if value.dtype.kind == "f":
                value = value.astype(self.float_dtype)
            elif value.dtype.kind == "i":
                value = value.astype(self.int_dtype)
            else:
                raise NotImplementedError(f"Not supported dtype: {value.dtype}")
            data[name] = value

        if self.cache is not None and self.cache.size < self.max_cache_size:
            self.cache[uid] = data

        retval = uid, data
        return retval
