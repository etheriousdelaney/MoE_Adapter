from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
import lightning as L

class TestModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Linear(1,1)

    def forward(
        self,
        batch,
    ) -> dict[str, torch.Tensor]:
        if self.frontend is None:
            raise RuntimeError("Frontend is not initialized for online feature extraction.")
        audio_path = batch['audio_path']
        labels = batch['label']
        frontend_batch = self.model(audio_path)

        if labels is not None:
            labels = labels.to(self.device)
        

        return frontend_batch



        