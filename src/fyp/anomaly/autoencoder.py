"""Compatibility shim for autoencoder anomaly detector.

This module re-exports AutoencoderAnomalyDetector from fyp.models.autoencoder
to provide a more intuitive import path.
"""

from fyp.models.autoencoder import AutoencoderAnomalyDetector

__all__ = ["AutoencoderAnomalyDetector"]
