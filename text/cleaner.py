from typing import Collection, Optional

# import tacotron_cleaner.cleaners
# from jaconv import jaconv
from typeguard import typechecked

# try:
#     from vietnamese_cleaner import vietnamese_cleaners
# except ImportError:
#     vietnamese_cleaners = None

# from espnet2.text.korean_cleaner import KoreanCleaner


class TextCleaner:
    """Text cleaner.

    Examples:
        >>> cleaner = TextCleaner("tacotron")
        >>> cleaner("(Hello-World);   &  jr. & dr.")
        'HELLO WORLD, AND JUNIOR AND DOCTOR'

    """

    @typechecked
    def __init__(self, cleaner_types: Optional[Collection[str]] = None):


        self.cleaner_types = []

    def __call__(self, text: str) -> str:
        for t in self.cleaner_types:
            if t == "tacotron":
                text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
            elif t == "jaconv":
                text = jaconv.normalize(text)
            elif t == "vietnamese":
                if vietnamese_cleaners is None:
                    raise RuntimeError("Please install underthesea")
                text = vietnamese_cleaners.vietnamese_cleaner(text)
            elif t == "korean_cleaner":
                text = KoreanCleaner.normalize_text(text)
            elif "whisper" in t and self.whisper_cleaner is not None:
                text = self.whisper_cleaner(text)
            else:
                raise RuntimeError(f"Not supported: type={t}")

        return text
