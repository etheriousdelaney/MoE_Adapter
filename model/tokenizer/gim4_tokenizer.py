import torch
import librosa
import os

from transformers import WhisperFeatureExtractor
from .modeling_whisper import WhisperVQEncoder
from .utils import extract_speech_token
from torch import nn


class Glm4Tokenizer(nn.Module):
    def __init__(self, tokenizer_path):
        super().__init__()
        self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().cuda()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

    def tokenize(self, audio_info=None):
        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, [audio_info]
        )[0]
        audio_tokens = torch.tensor(audio_tokens).unsqueeze(0)
        return audio_tokens