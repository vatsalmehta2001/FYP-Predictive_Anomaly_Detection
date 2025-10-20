"""Lightweight temporal autoencoder for anomaly detection.

Deprecated import path; prefer `fyp.anomaly.autoencoder`.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class TemporalAEDataset(Dataset):
    """Dataset for temporal autoencoder training."""

    def __init__(self, sequences: list[np.ndarray], window_size: int = 48):
        self.sequences = sequences
        self.window_size = window_size
        self.windows = self._create_windows()

    def _create_windows(self) -> list[np.ndarray]:
        """Create sliding windows from sequences."""
        windows = []

        for seq in self.sequences:
            if len(seq) < self.window_size:
                continue

            for i in range(len(seq) - self.window_size + 1):
                window = seq[i : i + self.window_size]
                windows.append(window)

        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = torch.FloatTensor(self.windows[idx])
        return window, window  # Input and target are the same for autoencoder


class TemporalAutoencoder(nn.Module):
    """Lightweight temporal autoencoder for anomaly detection."""

    def __init__(
        self,
        input_size: int = 48,
        hidden_sizes: list[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        if hidden_sizes is None:
            hidden_sizes = [32, 16, 8]
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        # Encoder
        encoder_layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            encoder_layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU() if activation == "relu" else nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_size = hidden_size

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (reverse of encoder)
        decoder_layers = []
        reversed_sizes = list(reversed(hidden_sizes[:-1])) + [input_size]

        for i, hidden_size in enumerate(reversed_sizes):
            decoder_layers.append(nn.Linear(prev_size, hidden_size))

            if i < len(reversed_sizes) - 1:  # No activation on final layer
                decoder_layers.extend(
                    [
                        nn.ReLU() if activation == "relu" else nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )

            prev_size = hidden_size

        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # x: [batch_size, input_size]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """Get encoded representation."""
        return self.encoder(x)


class AutoencoderAnomalyDetector:
    """Autoencoder-based anomaly detector."""

    def __init__(
        self,
        window_size: int = 48,
        hidden_sizes: list[int] = None,
        learning_rate: float = 1e-3,
        max_epochs: int = 20,
        batch_size: int = 32,
        contamination: float = 0.05,
        device: str = "cpu",
    ):
        if hidden_sizes is None:
            hidden_sizes = [32, 16, 8]
        self.config = {
            "window_size": window_size,
            "hidden_sizes": hidden_sizes,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "contamination": contamination,
        }

        self.device = torch.device(device)
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.threshold = None
        self.training_history = []

    def _normalize_data(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize data."""
        if fit:
            self.scaler_mean = np.mean(data)
            self.scaler_std = np.std(data)
            if self.scaler_std == 0:
                self.scaler_std = 1.0

        return (data - self.scaler_mean) / self.scaler_std

    def fit(self, sequences: list[np.ndarray]) -> dict:
        """Fit autoencoder on normal data."""
        logger.info("Training temporal autoencoder")

        if not sequences:
            raise ValueError("No training sequences provided")

        # Normalize data
        np.concatenate(sequences)
        normalized_sequences = [
            self._normalize_data(seq, fit=(i == 0)) for i, seq in enumerate(sequences)
        ]

        # Create dataset
        dataset = TemporalAEDataset(normalized_sequences, self.config["window_size"])

        if len(dataset) == 0:
            raise ValueError("No valid windows created from sequences")

        # Create model
        self.model = TemporalAutoencoder(
            input_size=self.config["window_size"],
            hidden_sizes=self.config["hidden_sizes"],
        ).to(self.device)

        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        criterion = nn.MSELoss()

        # Training
        train_loader = DataLoader(
            dataset, batch_size=self.config["batch_size"], shuffle=True
        )

        for epoch in range(self.config["max_epochs"]):
            self.model.train()
            epoch_loss = 0.0

            for batch_input, batch_target in train_loader:
                batch_input = batch_input.to(self.device)
                batch_target = batch_target.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_input)
                loss = criterion(outputs, batch_target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            self.training_history.append({"epoch": epoch, "loss": epoch_loss})

            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: loss={epoch_loss:.6f}")

        # Calculate threshold based on training reconstruction errors
        self._calculate_threshold(dataset)

        return {
            "final_loss": epoch_loss,
            "epochs_trained": len(self.training_history),
            "threshold": self.threshold,
        }

    def _calculate_threshold(self, dataset: TemporalAEDataset) -> None:
        """Calculate anomaly threshold from training reconstruction errors."""
        self.model.eval()
        errors = []

        data_loader = DataLoader(dataset, batch_size=self.config["batch_size"])

        with torch.no_grad():
            for batch_input, batch_target in data_loader:
                batch_input = batch_input.to(self.device)
                batch_target = batch_target.to(self.device)

                outputs = self.model(batch_input)
                mse = torch.mean((outputs - batch_target) ** 2, dim=1)
                errors.extend(mse.cpu().detach().numpy())

        # Set threshold at percentile based on contamination rate
        percentile = (1 - self.config["contamination"]) * 100
        self.threshold = np.percentile(errors, percentile)

        logger.info(
            f"Anomaly threshold set to {self.threshold:.6f} ({percentile:.1f}th percentile)"
        )

    def predict_scores(self, data: np.ndarray) -> np.ndarray:
        """Generate anomaly scores for time series."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        # Normalize data
        normalized_data = self._normalize_data(data)

        # Create sliding windows
        window_size = self.config["window_size"]
        scores = np.zeros(len(data))

        self.model.eval()
        with torch.no_grad():
            for i in range(len(data) - window_size + 1):
                window = normalized_data[i : i + window_size]
                window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)

                # Get reconstruction
                reconstruction = self.model(window_tensor)

                # Calculate reconstruction error
                mse = torch.mean((reconstruction - window_tensor) ** 2).item()

                # Standardize score
                score = mse / (self.threshold + 1e-8)

                # Assign score to center of window
                center_idx = i + window_size // 2
                scores[center_idx] = score

        # Fill in edges with nearest values
        for i in range(window_size // 2):
            scores[i] = scores[window_size // 2]
            scores[-(i + 1)] = scores[-(window_size // 2 + 1)]

        return scores

    def predict_labels(
        self, data: np.ndarray, threshold_multiplier: float = 1.0
    ) -> np.ndarray:
        """Generate binary anomaly labels."""
        scores = self.predict_scores(data)
        threshold = self.threshold * threshold_multiplier
        return (scores > threshold).astype(int)


def create_autoencoder_config(use_samples: bool = False) -> dict:
    """Create autoencoder configuration."""
    if use_samples:
        # Fast config for CI/samples
        return {
            "window_size": 24,
            "hidden_sizes": [16, 8],
            "max_epochs": 2,
            "batch_size": 8,
            "learning_rate": 1e-2,
            "contamination": 0.1,
        }
    else:
        # Full config for real training
        return {
            "window_size": 48,
            "hidden_sizes": [32, 16, 8],
            "max_epochs": 30,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "contamination": 0.05,
        }
