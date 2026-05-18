import json
import logging
from pathlib import Path
from typing import Collection, Dict, Iterable, Optional, Union

import numpy as np
from typeguard import typechecked

from text.build_tokenizer import build_tokenizer
from text.cleaner import TextCleaner
from text.token_id_converter import TokenIDConverter


class CommonPreprocessor:
    start_audio_token = "<start_audio>"
    end_audio_token = "<end_audio>"

    def __init__(
        self,
        train: bool,
        token_type: Optional[str] = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: Optional[str] = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: Optional[str] = None,
        aux_task_names: Collection[str] = None,
        text_name: str = "text",
        nonsplit_symbol: Iterable[str] = None,
        whisper_language: Optional[str] = None,
        whisper_task: Optional[str] = None,
        huggingface_max_length: Optional[int] = None,
    ):
        self.train = train
        self.text_name = text_name
        self.aux_task_names = aux_task_names
        self.huggingface_max_length = huggingface_max_length
        self.is_huggingface_tokenizer = token_type in {"huggingface", "hf"}

        if token_type is not None:
            if token_list is None:
                raise ValueError("token_list is required if token_type is not None")
            self.text_cleaner = TextCleaner(text_cleaner)

            tokenizer_model = (
                token_list
                if self.is_huggingface_tokenizer and bpemodel is None
                else bpemodel
            )
            self.tokenizer = build_tokenizer(
                token_type=token_type,
                bpemodel=tokenizer_model,
                delimiter=delimiter,
                space_symbol=space_symbol,
                non_linguistic_symbols=non_linguistic_symbols,
                g2p_type=g2p_type,
                nonsplit_symbol=nonsplit_symbol,
                whisper_language=whisper_language,
                whisper_task=whisper_task,
            )
            if self.is_huggingface_tokenizer:
                self.token_id_converter = None
            else:
                self.token_id_converter = TokenIDConverter(
                    token_list=token_list,
                    unk_symbol=unk_symbol,
                    tokenizer=self.tokenizer,
                )
        else:
            self.text_cleaner = None
            self.tokenizer = None
            self.token_id_converter = None

    @typechecked
    def _speech_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        return data

    def _encode_text(self, text: str) -> list[int]:
        text = self.text_cleaner(text)
        if self.is_huggingface_tokenizer and hasattr(self.tokenizer, "encode"):
            return self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=self.huggingface_max_length,
            )

        tokens = self.tokenizer.text2tokens(text)
        return self.token_id_converter.tokens2ids(tokens)

    def _encode_context_text(self, text: str) -> list[int]:
        if self.is_huggingface_tokenizer and hasattr(self.tokenizer, "encode"):
            return self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=None,
            )

        text = self.text_cleaner(text)
        tokens = self.tokenizer.text2tokens(text)
        return self.token_id_converter.tokens2ids(tokens)

    def _apply_chat_template(self, messages) -> str:
        hf_tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        if not hasattr(hf_tokenizer, "apply_chat_template"):
            raise RuntimeError("audio_context requires a tokenizer with apply_chat_template")
        try:
            return hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _audio_context_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        if "audio_context" not in data or self.tokenizer is None:
            return data

        raw_context = data.pop("audio_context")
        if isinstance(raw_context, np.ndarray):
            raise RuntimeError("audio_context must be a JSON string before preprocessing")
        messages = json.loads(str(raw_context))
        chat_text = self._apply_chat_template(messages)

        start_index = chat_text.rfind(self.start_audio_token)
        end_index = chat_text.rfind(self.end_audio_token)
        if start_index < 0 or end_index < 0 or end_index < start_index:
            raise RuntimeError(
                "audio_context chat text must contain <start_audio> before <end_audio>"
            )

        prefix_end = start_index + len(self.start_audio_token)
        audio_prefix_text = chat_text[:prefix_end]
        audio_suffix_text = chat_text[end_index:]
        data["audio_prefix"] = np.array(
            self._encode_context_text(audio_prefix_text),
            dtype=np.int64,
        )
        data["audio_suffix"] = np.array(
            self._encode_context_text(audio_suffix_text),
            dtype=np.int64,
        )
        return data

    def _text_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if self.tokenizer is None:
            return data

        target_names = []
        for name in (self.text_name, "answer", "response", "text"):
            if name and name not in target_names:
                target_names.append(name)

        for name in target_names:
            if name not in data:
                continue
            text = data[name]
            if isinstance(text, np.ndarray):
                continue
            text_ints = self._encode_text(text)
            if len(text_ints) > 500:
                logging.warning(
                    "The length of the text output exceeds 500, "
                    "which may cause OOM on the GPU."
                    "Please ensure that the data processing is correct and verify it."
                )
            data[name] = np.array(text_ints, dtype=np.int64)
        if self.aux_task_names is not None:
            for name in self.aux_task_names:
                if name in data:
                    text = data[name]
                    if isinstance(text, np.ndarray):
                        continue
                    data[name] = np.array(self._encode_text(text), dtype=np.int64)
        return data

    @typechecked
    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:

        data = self._speech_process(data)
        data = self._audio_context_process(data)
        data = self._text_process(data)
        return data
