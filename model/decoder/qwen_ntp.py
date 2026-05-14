from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from typing import Iterable

import torch
from huggingface_hub.utils import disable_progress_bars as hf_disable_progress_bars
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging


@dataclass
class QwenTextGenerationBundle:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device


class QwenNTPDecoder(nn.Module):
    def __init__(
        self,
        input_hidden_size: int,
        llm_repo: str = "Qwen/Qwen3-1.7B",
        projector_hidden_dim: int = 2048,
        projector_num_layers: int = 2,
        dropout: float = 0.1,
        max_target_length: int = 256,
        prompt_text: str = "Transcribe the following speech:",
    ):
        super().__init__()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        hf_disable_progress_bars()
        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_repo, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_repo,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.llm_hidden_size = int(self.llm.config.hidden_size)
        self.max_target_length = int(max_target_length)
        self.prompt_text = prompt_text
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0)
        )
        self.eos_token_id = self.tokenizer.eos_token_id

        layers = [nn.LayerNorm(input_hidden_size)]
        in_features = input_hidden_size
        for _ in range(max(1, projector_num_layers) - 1):
            layers.extend(
                [
                    nn.Linear(in_features, projector_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = projector_hidden_dim
        layers.append(nn.Linear(in_features, self.llm_hidden_size))
        self.projector = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        self.register_buffer(
            "prompt_ids",
            torch.tensor(prompt_ids, dtype=torch.long),
            persistent=False,
        )

    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.llm.parameters()).dtype

    def forward(
        self,
        audio_hidden_states: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Paper-style training: concatenate prompt + audio prefix + target text,
        # and compute causal next-token loss only on the transcript positions.
        inputs_embeds, attention_mask, labels = self._build_training_inputs(
            audio_hidden_states=audio_hidden_states,
            audio_attention_mask=audio_attention_mask,
            text_input_ids=text_input_ids,
            text_lengths=text_lengths,
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits, outputs.loss

    @torch.inference_mode()
    def greedy_generate(
        self,
        audio_hidden_states: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> tuple[list[list[int]], list[float]]:
        # Greedy decoding is implemented manually so the audio prefix can stay in
        # inputs_embeds while we append generated token embeddings step-by-step.
        prefix_embeds, prefix_mask = self._build_prefix_inputs(
            audio_hidden_states=audio_hidden_states,
            audio_attention_mask=audio_attention_mask,
        )
        batch_size = prefix_embeds.shape[0]
        generated: list[list[int]] = [[] for _ in range(batch_size)]
        scores = [0.0 for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=prefix_embeds.device)

        current_embeds = prefix_embeds
        current_mask = prefix_mask
        past_key_values = None
        input_embeddings = self.llm.get_input_embeddings()

        for _ in range(max_new_tokens):
            outputs = self.llm(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            next_token_ids = torch.argmax(log_probs, dim=-1)
            next_token_ids = torch.where(
                finished,
                torch.full_like(next_token_ids, self.pad_token_id),
                next_token_ids,
            )
            next_token_scores = log_probs.gather(1, next_token_ids.unsqueeze(1)).squeeze(1)
            for batch_idx, token_id in enumerate(next_token_ids.tolist()):
                if finished[batch_idx]:
                    continue
                if self.eos_token_id is not None and token_id == self.eos_token_id:
                    finished[batch_idx] = True
                    continue
                generated[batch_idx].append(int(token_id))
                scores[batch_idx] += float(next_token_scores[batch_idx].item())

            if bool(torch.all(finished)):
                break

            past_key_values = outputs.past_key_values
            current_embeds = input_embeddings(next_token_ids.unsqueeze(1)).to(
                device=prefix_embeds.device,
                dtype=self.model_dtype,
            )
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones(batch_size, 1, device=current_mask.device, dtype=current_mask.dtype),
                ],
                dim=1,
            )

        return generated, scores

    def ids_to_text(self, ids: Iterable[int]) -> tuple[list[str], str]:
        token_ids = [int(idx) for idx in ids]
        if not token_ids:
            return [], ""
        token_items = self.tokenizer.convert_ids_to_tokens(token_ids)
        filtered_token_ids = [
            token_id
            for token_id in token_ids
            if token_id not in set(self.tokenizer.all_special_ids)
        ]
        text = self.tokenizer.decode(filtered_token_ids, skip_special_tokens=True).strip()
        return token_items, text

    def _build_prefix_inputs(
        self,
        audio_hidden_states: torch.Tensor,
        audio_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = audio_hidden_states.shape[0]
        # projected_audio = audio_hidden_states.to(self.model_dtype)
        projected_audio = self.dropout(self.projector(audio_hidden_states)).to(self.model_dtype)
        prompt_embeds = self._prompt_embeds(batch_size, projected_audio.device)
        prompt_mask = self._prompt_mask(batch_size, projected_audio.device)
        prefix_embeds = torch.cat([prompt_embeds, projected_audio], dim=1)
        attention_mask = torch.cat(
            [prompt_mask, audio_attention_mask.to(device=projected_audio.device, dtype=torch.long)],
            dim=1,
        )
        return prefix_embeds, attention_mask

    def _build_training_inputs(
        self,
        audio_hidden_states: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_embeds, attention_mask = self._build_prefix_inputs(
            audio_hidden_states=audio_hidden_states,
            audio_attention_mask=audio_attention_mask,
        )
        target_ids, target_mask = self._build_target_ids(text_input_ids, text_lengths)
        target_embeddings = self.llm.get_input_embeddings()(target_ids).to(
            device=prefix_embeds.device,
            dtype=self.model_dtype,
        )

        inputs_embeds = torch.cat([prefix_embeds, target_embeddings], dim=1)
        attention_mask = torch.cat(
            [attention_mask, target_mask.to(device=attention_mask.device, dtype=torch.long)],
            dim=1,
        )

        # Prefix positions must not contribute to the LM objective.
        prefix_labels = torch.full(
            (prefix_embeds.shape[0], prefix_embeds.shape[1]),
            -100,
            device=prefix_embeds.device,
            dtype=torch.long,
        )
        target_labels = target_ids.masked_fill(~target_mask, -100)
        labels = torch.cat([prefix_labels, target_labels.to(prefix_embeds.device)], dim=1)
        return inputs_embeds, attention_mask, labels

    def _build_target_ids(
        self,
        text_input_ids: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = text_input_ids.shape[0]
        target_sequences: list[list[int]] = []
        for batch_idx in range(batch_size):
            length = int(text_lengths[batch_idx].item())
            token_ids = text_input_ids[batch_idx, :length].tolist()
            token_ids = token_ids[: self.max_target_length]
            if self.eos_token_id is not None:
                token_ids = token_ids + [int(self.eos_token_id)]
            if not token_ids:
                token_ids = [int(self.pad_token_id)]
            target_sequences.append([int(token_id) for token_id in token_ids])

        max_length = max(len(sequence) for sequence in target_sequences)
        padded = torch.full(
            (batch_size, max_length),
            fill_value=int(self.pad_token_id),
            dtype=torch.long,
            device=text_input_ids.device,
        )
        mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=text_input_ids.device)
        for batch_idx, sequence in enumerate(target_sequences):
            padded[batch_idx, : len(sequence)] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=text_input_ids.device,
            )
            mask[batch_idx, : len(sequence)] = True
        return padded, mask

    def _prompt_embeds(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.prompt_ids.numel() == 0:
            return torch.zeros(
                batch_size,
                0,
                self.llm_hidden_size,
                device=device,
                dtype=self.model_dtype,
            )
        prompt_embeds = self.llm.get_input_embeddings()(self.prompt_ids.to(device))
        prompt_embeds = prompt_embeds.to(dtype=self.model_dtype)
        return prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)

    def _prompt_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.ones(
            batch_size,
            int(self.prompt_ids.numel()),
            device=device,
            dtype=torch.long,
        )


def load_model(
    model: str,
    device: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = True,
    **_: object,
) -> QwenTextGenerationBundle:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    hf_disable_progress_bars()
    transformers_logging.disable_progress_bar()
    transformers_logging.set_verbosity_error()

    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(torch_dtype, resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
        torch_dtype=resolved_dtype,
    )
    llm.to(resolved_device)
    llm.eval()
    return QwenTextGenerationBundle(tokenizer=tokenizer, model=llm, device=resolved_device)


@torch.inference_mode()
def generate_response(
    model_bundle: QwenTextGenerationBundle,
    messages: list[dict],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = False,
    enable_thinking: bool = False,
    **_: object,
) -> str:
    tokenizer = model_bundle.tokenizer
    llm = model_bundle.model
    device = model_bundle.device

    prompt = _apply_chat_template(
        tokenizer=tokenizer,
        messages=messages,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generation_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(temperature)
        generation_kwargs["top_p"] = float(top_p)

    generated_ids = llm.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        **generation_kwargs,
    )
    new_ids = generated_ids[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return _strip_thinking_content(text)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(torch_dtype: str, device: torch.device) -> torch.dtype:
    if torch_dtype == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if torch_dtype not in aliases:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return aliases[torch_dtype]


def _apply_chat_template(
    tokenizer: AutoTokenizer,
    messages: list[dict],
    enable_thinking: bool,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Older chat-template implementations may not expose enable_thinking.
        # Use Qwen's soft switch as a best-effort fallback.
        fallback_messages = list(messages)
        if not enable_thinking and fallback_messages:
            fallback_messages = [dict(message) for message in fallback_messages]
            fallback_messages[-1]["content"] = str(fallback_messages[-1].get("content", "")) + "\n/no_think"
        return tokenizer.apply_chat_template(
            fallback_messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _strip_thinking_content(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("<think>"):
        return stripped

    end_tag = "</think>"
    end = stripped.find(end_tag)
    if end >= 0:
        return stripped[end + len(end_tag) :].strip()

    # If generation stopped before </think>, there is no final answer yet.
    # Return an empty direct answer instead of leaking chain-of-thought.
    return ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run direct text inference with a Qwen causal LM.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--message", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable Qwen thinking mode. Default is disabled for direct answers.",
    )
    parser.add_argument("--no-trust-remote-code", action="store_true")
    return parser


def _main() -> int:
    args = _build_parser().parse_args()
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.message})
    bundle = load_model(
        args.model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=not args.no_trust_remote_code,
    )
    print(
        generate_response(
            bundle,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            enable_thinking=args.enable_thinking,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
