"""Self-play reinforcement learning system for energy forecasting.

This module implements a three-component self-play architecture inspired by
Absolute Zero Reasoning (AZR) for energy consumption forecasting and anomaly
detection.

Components:
    - ProposerAgent: Generates challenging but physically-plausible scenarios
    - SolverAgent: Forecasts consumption under proposed scenarios
    - VerifierAgent: Validates forecasts using physics constraints
    - SelfPlayTrainer: Orchestrates the propose→solve→verify training loop
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

__all__ = [
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
