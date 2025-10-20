"""PatchTST-inspired model with quantile regression for energy forecasting."""

import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class EnergyPatchDataset(Dataset):
    """Dataset for patch-based time series training."""

    def __init__(
        self,
        sequences: list[np.ndarray],
        targets: list[np.ndarray],
        patch_len: int = 16,
        stride: int = 8,
    ):
        self.sequences = sequences
        self.targets = targets
        self.patch_len = patch_len
        self.stride = stride
        self.patches, self.patch_targets = self._create_patches()

    def _create_patches(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Create patches from sequences."""
        all_patches = []
        all_targets = []

        for seq, target in zip(self.sequences, self.targets, strict=False):
            if len(seq) < self.patch_len:
                continue

            # Create overlapping patches
            patches = []
            for i in range(0, len(seq) - self.patch_len + 1, self.stride):
                patch = seq[i : i + self.patch_len]
                patches.append(patch)

            if patches:
                all_patches.append(np.array(patches))
                all_targets.append(target)

        return all_patches, all_targets

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patches = torch.FloatTensor(self.patches[idx])  # [n_patches, patch_len]
        target = torch.FloatTensor(self.patch_targets[idx])
        return patches, target


class PatchEmbedding(nn.Module):
    """Patch embedding layer."""

    def __init__(self, patch_len: int, d_model: int, norm_layer=None):
        super().__init__()
        self.patch_len = patch_len
        self.linear = nn.Linear(patch_len, d_model)
        self.norm = norm_layer(d_model) if norm_layer else nn.Identity()

    def forward(self, x):
        # x: [batch_size, n_patches, patch_len]
        return self.norm(self.linear(x))  # [batch_size, n_patches, d_model]


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, : x.size(1)]


class EnergyPatchTST(nn.Module):
    """PatchTST-inspired model for energy forecasting with quantile outputs."""

    def __init__(
        self,
        patch_len: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        forecast_horizon: int = 48,
        quantiles: list[float] = None,
        activation: str = "gelu",
    ):
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.quantiles = sorted(quantiles)

        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_len, d_model, nn.LayerNorm)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Forecasting head with quantile outputs
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, forecast_horizon * len(self.quantiles)),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # x: [batch_size, n_patches, patch_len]
        batch_size, n_patches, _ = x.shape

        # Patch embedding
        x = self.patch_embedding(x)  # [batch_size, n_patches, d_model]

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)  # [batch_size, n_patches, d_model]

        # Global pooling
        x = x.mean(dim=1)  # [batch_size, d_model]

        # Forecasting
        forecast = self.forecast_head(x)  # [batch_size, horizon * n_quantiles]

        # Reshape to quantiles
        forecast = forecast.view(batch_size, self.forecast_horizon, len(self.quantiles))

        return forecast


class PatchTSTForecaster:
    """PatchTST forecaster with quantile regression."""

    def __init__(
        self,
        patch_len: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        forecast_horizon: int = 48,
        quantiles: list[float] = None,
        learning_rate: float = 1e-3,
        max_epochs: int = 20,
        batch_size: int = 32,
        early_stopping_patience: int = 5,
        device: str = "cpu",
    ):
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.config = {
            "patch_len": patch_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "forecast_horizon": forecast_horizon,
            "quantiles": quantiles,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "early_stopping_patience": early_stopping_patience,
        }

        self.device = torch.device(device)
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.training_history = []

    def _normalize_data(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize data to zero mean, unit variance."""
        if fit:
            self.scaler_mean = np.mean(data)
            self.scaler_std = np.std(data)
            if self.scaler_std == 0:
                self.scaler_std = 1.0

        return (data - self.scaler_mean) / self.scaler_std

    def _denormalize_data(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale."""
        return data * self.scaler_std + self.scaler_mean

    def _pinball_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate pinball loss for quantile regression."""
        # y_pred: [batch_size, horizon, n_quantiles]
        # y_true: [batch_size, horizon]

        y_true = y_true.unsqueeze(-1)  # [batch_size, horizon, 1]
        residual = y_true - y_pred  # [batch_size, horizon, n_quantiles]

        quantiles = torch.tensor(self.config["quantiles"], device=y_pred.device)
        quantiles = quantiles.view(1, 1, -1)  # [1, 1, n_quantiles]

        loss = torch.max(quantiles * residual, (quantiles - 1) * residual)

        return loss.mean()

    def fit(self, windows: list[dict], validation_split: float = 0.2) -> dict:
        """Fit the model on forecasting windows."""
        logger.info("Training PatchTST model")

        if not windows:
            raise ValueError("No training windows provided")

        # Prepare data
        sequences = [w["history_energy"] for w in windows]
        targets = [w["target_energy"] for w in windows]

        # Normalize data
        np.concatenate([np.concatenate(sequences), np.concatenate(targets)])
        normalized_sequences = [
            self._normalize_data(seq, fit=(i == 0)) for i, seq in enumerate(sequences)
        ]
        normalized_targets = [self._normalize_data(target) for target in targets]

        # Train/validation split
        n_train = int(len(normalized_sequences) * (1 - validation_split))
        train_sequences = normalized_sequences[:n_train]
        train_targets = normalized_targets[:n_train]
        val_sequences = normalized_sequences[n_train:]
        val_targets = normalized_targets[n_train:]

        # Create datasets
        train_dataset = EnergyPatchDataset(
            train_sequences,
            train_targets,
            self.config["patch_len"],
            stride=self.config["patch_len"] // 2,
        )

        if val_sequences:
            val_dataset = EnergyPatchDataset(
                val_sequences,
                val_targets,
                self.config["patch_len"],
                stride=self.config["patch_len"],
            )
        else:
            val_dataset = None

        # Create model
        self.model = EnergyPatchTST(
            patch_len=self.config["patch_len"],
            d_model=self.config["d_model"],
            n_heads=self.config["n_heads"],
            n_layers=self.config["n_layers"],
            forecast_horizon=self.config["forecast_horizon"],
            quantiles=self.config["quantiles"],
        ).to(self.device)

        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config["learning_rate"]
        )

        # Training loop
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["batch_size"], shuffle=True
        )
        val_loader = (
            DataLoader(val_dataset, batch_size=self.config["batch_size"])
            if val_dataset
            else None
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config["max_epochs"]):
            # Training
            self.model.train()
            train_loss = 0.0

            for patches, targets in train_loader:
                patches = patches.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(patches)
                loss = self._pinball_loss(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for patches, targets in val_loader:
                        patches = patches.to(self.device)
                        targets = targets.to(self.device)
                        outputs = self.model(patches)
                        loss = self._pinball_loss(outputs, targets)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config["early_stopping_patience"]:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            # Log progress
            self.training_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

            if epoch % 5 == 0:
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

        return {
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "epochs_trained": len(self.training_history),
        }

    def predict(
        self,
        history: np.ndarray,
        steps: int,
        return_quantiles: bool = True,
    ) -> dict[str, np.ndarray]:
        """Generate forecasts with quantile predictions."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()

        # Normalize input
        normalized_history = self._normalize_data(history)

        # Create patches
        dataset = EnergyPatchDataset(
            [normalized_history],
            [np.zeros(steps)],  # Dummy target
            self.config["patch_len"],
            stride=self.config["patch_len"],
        )

        if len(dataset) == 0:
            # Fallback for short sequences
            median_forecast = np.full(steps, np.median(history))
            if return_quantiles:
                return {
                    str(q): median_forecast.copy() for q in self.config["quantiles"]
                }
            else:
                return {"point": median_forecast}

        # Make prediction
        with torch.no_grad():
            patches, _ = dataset[0]
            patches = patches.unsqueeze(0).to(self.device)  # Add batch dimension
            outputs = self.model(patches)  # [1, horizon, n_quantiles]

            # Denormalize
            outputs_np = outputs.cpu().numpy()[0]  # [horizon, n_quantiles]

            results = {}
            for i, quantile in enumerate(self.config["quantiles"]):
                forecast = self._denormalize_data(outputs_np[:, i])
                # Ensure non-negative energy
                forecast = np.maximum(forecast, 0.0)
                results[str(quantile)] = forecast[:steps]

            if not return_quantiles:
                # Return median as point forecast
                results = {"point": results["0.5"]}

        return results


def create_patchtst_config(use_samples: bool = False) -> dict:
    """Create PatchTST configuration based on environment."""
    if use_samples:
        # Fast config for CI/samples
        return {
            "patch_len": 8,
            "d_model": 32,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 64,
            "forecast_horizon": 16,
            "max_epochs": 2,
            "batch_size": 8,
            "learning_rate": 1e-2,
            "early_stopping_patience": 2,
        }
    else:
        # Full config for real training
        return {
            "patch_len": 16,
            "d_model": 128,
            "n_heads": 8,
            "n_layers": 4,
            "d_ff": 256,
            "forecast_horizon": 48,
            "max_epochs": 50,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "early_stopping_patience": 10,
        }
