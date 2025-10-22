"""Self-play reinforcement learning system for energy forecasting.

This module implements a three-component self-play architecture inspired by
Absolute Zero Reasoning (AZR) for energy consumption forecasting and anomaly
detection.

Components:
    - ProposerAgent: Generates challenging but physically-plausible scenarios
    - SolverAgent: Forecasts consumption under proposed scenarios
    - VerifierAgent: Validates forecasts using physics constraints
    - SelfPlayTrainer: Orchestrates the propose→solve→verify training loop

BDH Enhancements (Optional, requires PyTorch):
    - HebbianVerifier: Constraint adaptation via synaptic-like plasticity
    - GraphBasedProposer: Scenario sampling from causal relationship graph
    - SparseActivationMonitor: Track activation sparsity for interpretability
    - create_bdh_enhanced_trainer: Helper to create BDH-enhanced trainer

References:
    Kosowski et al. (2025). The Dragon Hatchling: The Missing Link between
    the Transformer and Models of the Brain. arXiv:2509.26507
"""

from fyp.selfplay.proposer import ProposerAgent, ScenarioProposal
from fyp.selfplay.solver import SolverAgent
from fyp.selfplay.trainer import SelfPlayTrainer
from fyp.selfplay.verifier import (
    Constraint,
    HouseholdMaxConstraint,
    NonNegativityConstraint,
    RampRateConstraint,
    VerifierAgent,
)

# BDH enhancements (optional imports - won't break if not used)
try:
    from fyp.selfplay.bdh_enhancements import (
        GraphBasedProposer,
        HebbianVerifier,
        SparseActivationMonitor,
        create_bdh_enhanced_trainer,
    )

    _bdh_available = True
except ImportError:
    _bdh_available = False

__all__ = [
    # Core components
    "ProposerAgent",
    "ScenarioProposal",
    "SolverAgent",
    "VerifierAgent",
    "SelfPlayTrainer",
    "Constraint",
    "NonNegativityConstraint",
    "HouseholdMaxConstraint",
    "RampRateConstraint",
]

# Add BDH enhancements if available
if _bdh_available:
    __all__.extend(
        [
            "HebbianVerifier",
            "GraphBasedProposer",
            "SparseActivationMonitor",
            "create_bdh_enhanced_trainer",
        ]
    )
