import torch
from model.encoder.kimi_audio_encoder import KimiAudioFrontend
from model.encoder.kimi_audio_encoder import build_padding_mask_from_lengths
import lightning as L
from model.decoder.classfier import AudioClassfier
from model.adapter.MoEadapter import MoEAdapter
from train.config import ModelConfig

class LitMoEClassifier(L.LightningModule):
    def __init__(
            self,
            config: ModelConfig,
            use_frontend: bool = True,
            ):
        super().__init__()
        self.save_hyperparameters()
        self.frontend = (
            KimiAudioFrontend(
                model_repo=config.model_repo,
                tokenizer_repo=config.tokenizer_repo,
                sample_rate=config.sample_rate,
            )
            if use_frontend
            else None
        )
        self.hidden_size = (
            self.frontend.hidden_size if self.frontend is not None else 3584
        )
        self.aux_loss_weight = config.aux_loss_weight

        self.adapter = MoEAdapter(
            hidden_size = self.hidden_size,  #3584
            **config.adapter.__dict__
        )
        self.model_dtype = next(self.adapter.parameters()).dtype

        # This head performs masked temporal pooling before classification.
        self.decoder = AudioClassfier(
            hidden_size = self.hidden_size,
            num_classes = config.num_classes,
            **config.classifier.__dict__
        )

    def forward(
        self,
        batch,
    ) -> dict[str, torch.Tensor]:
        audio_input = self._unwrap_batch(batch)
        labels = audio_input.get("chime4_label")
        fused_input = audio_input.get("fused")
        fused_lengths = audio_input.get("fused_lengths")
        if fused_input is not None:
            fused_states = fused_input.to(self.device, dtype=self.model_dtype)
            if fused_lengths is None:
                fused_lengths = torch.full(
                    (fused_states.shape[0],),
                    fused_states.shape[1],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                fused_lengths = fused_lengths.to(self.device).long()
            padding_mask = build_padding_mask_from_lengths(
                fused_lengths,
                max_len=fused_states.shape[1],
            )
        else:
            if self.frontend is None:
                raise RuntimeError("Frontend is not initialized for online feature extraction.")
            frontend_batch = self.frontend(audio_input["sound"], audio_input["sound_lengths"])
            fused_states = frontend_batch.fused_states.to(self.device, dtype=self.model_dtype)
            padding_mask = frontend_batch.padding_mask.to(self.device)

        if labels is not None:
            labels = labels.to(self.device).view(-1).long()
        
        hidden_states, aux_loss, _selected_experts = self.adapter(
            hidden_states=fused_states,
            padding_mask=padding_mask,
        )
        logits, ce_loss = self.decoder(
            hidden_states,
            labels,
            padding_mask=padding_mask,
        )
        total_loss = ce_loss + (self.aux_loss_weight * aux_loss) if ce_loss is not None else aux_loss
        accuracy = None
        if labels is not None:
            accuracy = (logits.argmax(dim=-1) == labels).float().mean()

        return {
            "loss": total_loss,
            "ce_loss": ce_loss if ce_loss is not None else total_loss.new_zeros(()),
            "aux_loss": aux_loss,
            "acc": accuracy if accuracy is not None else total_loss.new_zeros(()),
        }

    @staticmethod
    def _unwrap_batch(batch) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            return batch
        if isinstance(batch, (list, tuple)) and len(batch) >= 2 and isinstance(batch[1], dict):
            return batch[1]
        raise TypeError(f"Unsupported batch type for LitMoEClassifier: {type(batch)}")



        
