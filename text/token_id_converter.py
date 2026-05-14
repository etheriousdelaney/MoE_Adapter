from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

from typeguard import typechecked


class TokenIDConverter:
    @typechecked
    def __init__(
        self,
        token_list: Union[Path, str, Iterable[str]],
        unk_symbol: str = "<unk>",
        tokenizer=None,
    ):
        if isinstance(token_list, (Path, str)):
            with Path(token_list).open("r", encoding="utf-8") as f:
                self.token_list = [line.rstrip("\n") for line in f if line.rstrip("\n")]
        else:
            self.token_list = list(token_list)

        self.tokenizer = tokenizer
        self.token2id = {token: idx for idx, token in enumerate(self.token_list)}
        if unk_symbol not in self.token2id:
            raise ValueError(f"{unk_symbol} is required in token_list")
        self.unk_symbol = unk_symbol
        self.unk_id = self.token2id[unk_symbol]

    def __len__(self) -> int:
        return len(self.token_list)

    def tokens2ids(self, tokens: Iterable[str]) -> list[int]:
        return [self.token2id.get(token, self.unk_id) for token in tokens]

    def ids2tokens(self, ids: Iterable[int]) -> list[str]:
        return [self.token_list[int(idx)] for idx in ids]
