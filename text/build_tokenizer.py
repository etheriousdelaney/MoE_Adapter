from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from typeguard import typechecked

from text.abs_tokenizer import AbsTokenizer
from text.char_tokenizer import CharTokenizer
from text.huggingface_tokenizer import HuggingFaceTokenizer

@typechecked
def build_tokenizer(
    token_type: str,
    bpemodel: Optional[Union[Path, str, Iterable[str]]] = None,
    non_linguistic_symbols: Optional[Union[Path, str, Iterable[str]]] = None,
    remove_non_linguistic_symbols: bool = False,
    space_symbol: str = "<space>",
    delimiter: Optional[str] = None,
    g2p_type: Optional[str] = None,
    nonsplit_symbol: Optional[Iterable[str]] = None,
    # tokenization encode (text2token) args, e.g. BPE dropout, only applied in training
    encode_kwargs: Optional[Dict] = None,
    # only use for whisper
    whisper_language: Optional[str] = None,
    whisper_task: Optional[str] = None,
    sot_asr: bool = False,
) -> AbsTokenizer:
    """A helper function to instantiate Tokenizer"""
    if token_type in {"huggingface", "hf"}:
        if bpemodel is None:
            raise ValueError("bpemodel must be provided for token_type=huggingface")
        return HuggingFaceTokenizer(str(bpemodel))

    return CharTokenizer(
        non_linguistic_symbols=non_linguistic_symbols,
        space_symbol=space_symbol,
        remove_non_linguistic_symbols=remove_non_linguistic_symbols,
        nonsplit_symbols=nonsplit_symbol,
    )
