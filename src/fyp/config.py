"""Configuration management for models and experiments."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ForecastingConfig(BaseModel):
    """Configuration for forecasting models."""

    # Model architecture
    model_type: str = Field(default="patchtst", description="Model type")
    patch_len: int = Field(default=16, description="Patch length for PatchTST")
    d_model: int = Field(default=128, description="Model dimension")
    n_heads: int = Field(default=8, description="Number of attention heads")
    n_layers: int = Field(default=4, description="Number of transformer layers")
    d_ff: int = Field(default=256, description="Feed-forward dimension")
    dropout: float = Field(default=0.1, description="Dropout rate")

    # Forecasting settings
    forecast_horizon: int = Field(default=48, description="Forecast horizon")
    context_length: int = Field(default=96, description="Input context length")
    quantiles: list[float] = Field(
        default=[0.1, 0.5, 0.9], description="Quantiles to predict"
    )

    # Training settings
    learning_rate: float = Field(default=1e-3, description="Learning rate")
    max_epochs: int = Field(default=50, description="Maximum training epochs")
    batch_size: int = Field(default=32, description="Batch size")
    early_stopping_patience: int = Field(
        default=10, description="Early stopping patience"
    )
    validation_split: float = Field(default=0.2, description="Validation split ratio")

    # Hardware
    device: str = Field(default="cpu", description="Device to use (cpu/cuda)")


class AnomalyConfig(BaseModel):
    """Configuration for anomaly detection models."""

    # Model architecture
    model_type: str = Field(default="autoencoder", description="Model type")
    window_size: int = Field(default=48, description="Window size for autoencoder")
    hidden_sizes: list[int] = Field(
        default=[32, 16, 8], description="Hidden layer sizes"
    )
    dropout: float = Field(default=0.1, description="Dropout rate")
    activation: str = Field(default="relu", description="Activation function")

    # Anomaly detection settings
    contamination: float = Field(
        default=0.05, description="Expected contamination rate"
    )
    threshold_multiplier: float = Field(default=1.0, description="Threshold multiplier")

    # Training settings
    learning_rate: float = Field(default=1e-3, description="Learning rate")
    max_epochs: int = Field(default=30, description="Maximum training epochs")
    batch_size: int = Field(default=32, description="Batch size")

    # Hardware
    device: str = Field(default="cpu", description="Device to use (cpu/cuda)")


class ExperimentConfig(BaseModel):
    """Overall experiment configuration."""

    # Data settings
    dataset: str = Field(description="Dataset to use")
    use_samples: bool = Field(default=False, description="Use sample data")
    start_date: str | None = Field(default=None, description="Start date filter")
    end_date: str | None = Field(default=None, description="End date filter")
    entities: list[str] | None = Field(
        default=None, description="Entity IDs to process"
    )

    # Model configurations
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)

    # Output settings
    output_dir: str = Field(
        default="data/derived/evaluation", description="Output directory"
    )
    save_models: bool = Field(default=True, description="Save trained models")
    create_plots: bool = Field(default=True, description="Create evaluation plots")

    # MLflow settings
    mlflow_experiment: str = Field(
        default="energy_forecasting", description="MLflow experiment name"
    )
    mlflow_run_name: str | None = Field(default=None, description="MLflow run name")


def load_config(config_path: Path | None = None) -> ExperimentConfig:
    """Load configuration from file or create default."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        return ExperimentConfig(**config_dict)
    else:
        return ExperimentConfig(dataset="lcl")


def create_sample_config() -> ExperimentConfig:
    """Create fast configuration for samples/CI."""
    config = ExperimentConfig(dataset="lcl", use_samples=True)

    # Fast forecasting config
    config.forecasting.patch_len = 8
    config.forecasting.d_model = 32
    config.forecasting.n_heads = 2
    config.forecasting.n_layers = 1
    config.forecasting.d_ff = 64
    config.forecasting.forecast_horizon = 16
    config.forecasting.max_epochs = 2
    config.forecasting.batch_size = 8
    config.forecasting.learning_rate = 1e-2
    config.forecasting.early_stopping_patience = 2

    # Fast anomaly config
    config.anomaly.window_size = 24
    config.anomaly.hidden_sizes = [16, 8]
    config.anomaly.max_epochs = 2
    config.anomaly.batch_size = 8
    config.anomaly.learning_rate = 1e-2

    return config


def save_config(config: ExperimentConfig, config_path: Path) -> None:
    """Save configuration to YAML file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, indent=2)


def get_config_from_env() -> ExperimentConfig:
    """Create configuration from environment variables."""
    config = ExperimentConfig(dataset="lcl")

    # Check for CI environment
    if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
        config = create_sample_config()

    # Override with environment variables
    if os.getenv("USE_SAMPLES"):
        config.use_samples = os.getenv("USE_SAMPLES").lower() == "true"

    if os.getenv("DEVICE"):
        config.forecasting.device = os.getenv("DEVICE")
        config.anomaly.device = os.getenv("DEVICE")

    return config
