from __future__ import annotations

from typing import Iterable, List

from transformers import AutoTokenizer

from text.abs_tokenizer import AbsTokenizer


class HuggingFaceTokenizer(AbsTokenizer):
    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def text2tokens(self, line: str) -> List[str]:
        return self.tokenizer.tokenize(line)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(list(tokens))

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=max_length is not None,
            max_length=max_length,
            return_attention_mask=False,
        )
        return list(encoded["input_ids"])

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)
