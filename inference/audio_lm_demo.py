from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import soundfile as sf
import torch

from inference.asr_inference import (
    load_inference_state_dict,
    load_state_for_inference,
    load_train_config,
    validate_asr_inference_model,
)
from train.model_factory import build_model_from_config


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a trained audio-language model on a single wav file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tag", type=str, default="", help="Experiment tag under exp/.")
    parser.add_argument("--exp_dir", type=str, default="", help="Experiment directory. Overrides --tag.")
    parser.add_argument(
        "--model",
        type=str,
        default="valid.lm_loss.ave_5best.pth",
        help="Checkpoint filename under <exp_dir>/checkpoint or <exp_dir>.",
    )
    parser.add_argument("--wav", type=str, default="", help="Input wav/flac path.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Transcribe the following speech:",
        help="Prompt prepended before the audio prefix.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="", help="cuda, cuda:0, or cpu. Empty chooses automatically.")
    parser.add_argument("--list_exps", action="store_true", help="List usable exp directories and exit.")
    parser.add_argument("--show_tokens", action="store_true", help="Print generated token strings and ids.")
    return parser


def find_experiments(exp_root: Path = Path("exp")) -> list[tuple[Path, list[str]]]:
    if not exp_root.exists():
        return []
    rows = []
    for config_path in sorted(exp_root.glob("*/config.yaml")):
        exp_dir = config_path.parent
        checkpoint_dir = exp_dir / "checkpoint"
        checkpoints = []
        if checkpoint_dir.exists():
            checkpoints.extend(path.name for path in checkpoint_dir.glob("*.pth"))
            checkpoints.extend(path.name for path in checkpoint_dir.glob("*.ckpt"))
        checkpoints.extend(path.name for path in exp_dir.glob("*.pth"))
        checkpoints.extend(path.name for path in exp_dir.glob("*.ckpt"))
        rows.append((exp_dir, sorted(set(checkpoints))))
    return rows


def list_experiments(exp_root: Path = Path("exp")) -> None:
    rows = find_experiments(exp_root)
    if not exp_root.exists():
        print(f"No exp directory found: {exp_root}")
        return

    if not rows:
        print(f"No experiments with config.yaml found under {exp_root}")
        return

    for exp_dir, checkpoints in rows:
        if checkpoints:
            print(f"{exp_dir.name}: {', '.join(checkpoints)}")
        else:
            print(f"{exp_dir.name}: config.yaml found, no checkpoint files found")


def choose_from_list(label: str, items: list[str]) -> str:
    if not items:
        raise ValueError(f"No {label} available")
    for idx, item in enumerate(items, start=1):
        print(f"[{idx}] {item}")
    while True:
        choice = input(f"Select {label} [1-{len(items)}]: ").strip()
        try:
            choice_idx = int(choice)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= choice_idx <= len(items):
            return items[choice_idx - 1]
        print(f"Please enter a number between 1 and {len(items)}.")


def choose_experiment(exp_root: Path = Path("exp")) -> tuple[Path, str]:
    rows = find_experiments(exp_root)
    if not rows:
        raise ValueError(f"No experiments with config.yaml found under {exp_root}")
    exp_dir = Path(choose_from_list("experiment", [str(path) for path, _ in rows]))
    checkpoints = dict(rows)[exp_dir]
    model_name = choose_from_list("checkpoint", checkpoints) if checkpoints else ""
    return exp_dir, model_name


def resolve_exp_dir(tag: str, exp_dir: str) -> Path:
    if exp_dir:
        resolved = Path(exp_dir)
    elif tag:
        resolved = Path("exp") / tag
    else:
        raise ValueError("Either --tag or --exp_dir is required")
    if not resolved.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {resolved}")
    if not (resolved / "config.yaml").is_file():
        raise FileNotFoundError(f"Experiment config not found: {resolved / 'config.yaml'}")
    return resolved


def resolve_model_path(exp_dir: Path, model_name: str) -> Path:
    if not model_name:
        checkpoints = dict(find_experiments(exp_dir.parent)).get(exp_dir, [])
        model_name = choose_from_list("checkpoint", checkpoints)
    candidates = [
        exp_dir / "checkpoint" / model_name,
        exp_dir / model_name,
        Path(model_name),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Model file not found. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
        + "\nUse --list_exps to see available checkpoint names."
    )


def read_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=sample_rate)
    return torch.from_numpy(audio).to(torch.float32).flatten()


def load_model(exp_dir: Path, model_path: Path, device: torch.device) -> torch.nn.Module:
    config = load_train_config(exp_dir / "config.yaml")
    inference_config = config
    inference_config.dataset.data_type = ["sound"]
    model = build_model_from_config(config=inference_config, data_type=["sound"])
    state_dict, checkpoint_format = load_inference_state_dict(model_path)
    load_state_for_inference(
        model=model,
        state_dict=state_dict,
        checkpoint_format=checkpoint_format,
        data_type=["sound"],
    )
    validate_asr_inference_model(model)
    model = model.to(device).eval()
    frontend = getattr(model, "frontend", None)
    if frontend is not None:
        frontend.device = device
        frontend.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return model


def set_prompt(model: torch.nn.Module, prompt: str) -> None:
    decoder = getattr(model, "decoder", None)
    if decoder is None or not hasattr(decoder, "tokenizer"):
        raise TypeError(f"Model {type(model).__name__} does not expose a Qwen decoder prompt")
    prompt_ids = decoder.tokenizer(
        prompt,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    decoder.prompt_text = prompt
    decoder.prompt_ids = torch.tensor(
        prompt_ids,
        dtype=torch.long,
        device=next(decoder.parameters()).device,
    )


def run_single_wav(args: argparse.Namespace) -> None:
    if not args.tag and not args.exp_dir:
        exp_dir, selected_model = choose_experiment()
        model_name = selected_model or args.model
    else:
        exp_dir = resolve_exp_dir(args.tag, args.exp_dir)
        model_name = args.model
    model_path = resolve_model_path(exp_dir, model_name)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(exp_dir=exp_dir, model_path=model_path, device=device)
    set_prompt(model, args.prompt)

    sample_rate = int(getattr(model, "frontend", None).sample_rate)
    audio = read_audio(args.wav, sample_rate=sample_rate)
    batch = (
        [Path(args.wav).stem],
        {
            "sound": audio.unsqueeze(0),
            "sound_lengths": torch.tensor([audio.numel()], dtype=torch.long),
        },
    )

    with torch.inference_mode():
        generated_ids, scores, _ = model.generate_greedy(
            batch,
            max_new_tokens=args.max_new_tokens,
        )
    token_items, text = model.ids_to_text(generated_ids[0])

    print(text)
    print(f"\nscore: {float(scores[0]):.6f}")
    if args.show_tokens:
        print("tokens:", " ".join(token_items))
        print("token_ids:", " ".join(str(token_id) for token_id in generated_ids[0]))


def main() -> None:
    args = build_argparser().parse_args()
    if args.list_exps:
        list_experiments()
        return
    if not args.wav:
        raise ValueError("--wav is required unless --list_exps is used")
    run_single_wav(args)


if __name__ == "__main__":
    main()
