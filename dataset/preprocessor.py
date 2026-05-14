import logging
from pathlib import Path
from typing import Collection, Dict, Iterable, Optional, Union

import numpy as np
from typeguard import typechecked

from text.build_tokenizer import build_tokenizer
from text.cleaner import TextCleaner
from text.token_id_converter import TokenIDConverter


class CommonPreprocessor:
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

    def _text_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if self.text_name in data and self.tokenizer is not None:
            text = data[self.text_name]
            if isinstance(text, np.ndarray):
                return data
            text_ints = self._encode_text(text)
            if len(text_ints) > 500:
                logging.warning(
                    "The length of the text output exceeds 500, "
                    "which may cause OOM on the GPU."
                    "Please ensure that the data processing is correct and verify it."
                )
            data[self.text_name] = np.array(text_ints, dtype=np.int64)
        if self.aux_task_names is not None and self.tokenizer is not None:
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
        data = self._text_process(data)
        return data
