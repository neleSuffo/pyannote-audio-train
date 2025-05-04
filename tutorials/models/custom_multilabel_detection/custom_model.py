from typing import Optional
import torch
import torch.nn as nn
from pyannote.audio import Model
from pyannote.audio.core.task import Task
from torchaudio.models import wav2vec2_base

class CustomMultilabelModel(Model):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
        hidden_size: int = 256,
        num_transformer_layers: int = 4,
        output_size: int = 5,  # kchi, och, mal, fem, ovh
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)
        self.save_hyperparameters("hidden_size", "num_transformer_layers", "output_size")

        # Pre-trained Wav2Vec 2.0 for feature extraction
        self.feature_extractor = wav2vec2_base()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze pre-trained weights

        # Transformer encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=768,  # Wav2Vec 2.0 output size
            nhead=8,
            dim_feedforward=hidden_size,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)

        # Pooling and classification head
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, waveforms: torch.Tensor, **kwargs):
        # waveforms: (batch_size, channels, samples)
        features = self.feature_extractor(waveforms.squeeze(1))  # (batch_size, seq_len, 768)
        features = features.transpose(0, 1)  # ( seq_len, batch_size, 768)
        transformer_out = self.transformer(features)  # ( seq_len, batch_size, 768)
        pooled = self.pooling(transformer_out.permute(1, 2, 0)).squeeze(-1)  # (batch_size, 768)
        logits = self.head(pooled)  # (batch_size, output_size)
        return logits