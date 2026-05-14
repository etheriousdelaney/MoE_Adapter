import collections.abc
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile
from typeguard import typechecked

from fileio.read_text import read_2columns_text


def soundfile_read(
    wavs: Union[str, List[str]],
    dtype=None,
    concat_axis: int = 1,
    start: int = 0,
    end: int = None,
    return_subtype: bool = False,
) -> Tuple[np.array, int]:
    if isinstance(wavs, str):
        wavs = [wavs]

    arrays = []
    subtypes = []
    prev_rate = None
    prev_wav = None
    for wav in wavs:
        with soundfile.SoundFile(wav) as f:
            f.seek(start)
            if end is not None:
                frames = end - start
            else:
                frames = -1
            if dtype == "float16":
                array = f.read(
                    frames,
                    dtype="float32",
                ).astype(dtype)
            else:
                array = f.read(frames, dtype=dtype)
            rate = f.samplerate
            subtype = f.subtype
            subtypes.append(subtype)

        if len(wavs) > 1 and array.ndim == 1 and concat_axis == 1:
            # array: (Time, Channel)
            array = array[:, None]

        if prev_wav is not None:
            if prev_rate != rate:
                raise RuntimeError(
                    f"'{prev_wav}' and '{wav}' have mismatched sampling rate: "
                    f"{prev_rate} != {rate}"
                )

            dim1 = arrays[0].shape[1 - concat_axis]
            dim2 = array.shape[1 - concat_axis]
            if dim1 != dim2:
                raise RuntimeError(
                    "Shapes must match with "
                    f"{1 - concat_axis} axis, but gut {dim1} and {dim2}"
                )

        prev_rate = rate
        prev_wav = wav
        arrays.append(array)

    if len(arrays) == 1:
        array = arrays[0]
    else:
        array = np.concatenate(arrays, axis=concat_axis)

    if return_subtype:
        return array, rate, subtypes
    else:
        return array, rate


class SoundScpReader(collections.abc.Mapping):
    """Reader class for 'wav.scp'.

    Examples:
        wav.scp is a text file that looks like the following:

        key1 /some/path/a.wav
        key2 /some/path/b.wav
        key3 /some/path/c.wav
        key4 /some/path/d.wav
        ...

        >>> reader = SoundScpReader('wav.scp')
        >>> rate, array = reader['key1']

        If multi_columns=True is given and
        multiple files are given in one line
        with space delimiter, and  the output array are concatenated
        along channel direction

        key1 /some/path/a.wav /some/path/a2.wav
        key2 /some/path/b.wav /some/path/b2.wav
        ...

        >>> reader = SoundScpReader('wav.scp', multi_columns=True)
        >>> rate, array = reader['key1']

        In the above case, a.wav and a2.wav are concatenated.

        Note that even if multi_columns=True is given,
        SoundScpReader still supports a normal wav.scp,
        i.e., a wav file is given per line,
        but this option is disable by default
        because dict[str, list[str]] object is needed to be kept,
        but it increases the required amount of memory.
    """

    @typechecked
    def __init__(
        self,
        fname,
        dtype=None,
        concat_axis=1,
    ):
        self.fname = fname
        self.dtype = dtype
        self.data = read_2columns_text(fname)
        self.concat_axis = concat_axis

    def __getitem__(self, key) -> Tuple[int, np.ndarray]:
        wavs = self.data[key]


        array, rate = soundfile_read(
            wavs,
            dtype=self.dtype,
            concat_axis=self.concat_axis,
        )
        # Returned as scipy.io.wavread's order
        return rate, array

    def get_path(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()