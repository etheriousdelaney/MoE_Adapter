import argparse
from pathlib import Path
import shutil

def rotate(path, max_num_log=1000):
    for i in range(max_num_log - 1, -1, -1):
        if i == 0:
            p = Path(path)
            pn = p.parent / (p.stem + ".1" + p.suffix)
        else:
            _p = Path(path)
            p = _p.parent / (_p.stem + f".{i}" + _p.suffix)
            pn = _p.parent / (_p.stem + f".{i + 1}" + _p.suffix)

        if p.exists():
            if i == max_num_log - 1:
                p.unlink()
            else:
                shutil.move(p, pn)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_filepath", type=str, help="Path to log-file to be rotated."
    )
    parser.add_argument(
        "--max-num-log-files",
        type=int,
        help="Maximum number of log-files to be kept.",
        default=1000,
    )
    args = parser.parse_args()

    rotate(args.log_filepath, args.max_num_log_files)


if __name__ == "__main__":
    main()