"""Verifier agent for validating energy consumption forecasts using physics constraints.

This module implements various physics and statistical constraints for evaluating
the plausibility of energy consumption forecasts, following UK electrical standards
and SSEN network constraints.
"""

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from fyp.selfplay.proposer import ScenarioProposal


class Constraint(ABC):
    """Abstract base class for physics/statistical constraints."""

    @abstractmethod
    def evaluate(
        self,
        forecast: np.ndarray,
        context: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> tuple[float, list[str]]:
        """Evaluate constraint on forecast.

        Args:
            forecast: Predicted consumption values
            context: Historical context (optional)
            timestamps: Timestamps for temporal constraints (optional)

        Returns:
            (score, violations): Score in [-1, 0] where 0=perfect compliance,
                               list of violation descriptions
        """
        pass


class NonNegativityConstraint(Constraint):
    """Energy consumption cannot be negative (hard physics constraint)."""

    def evaluate(
        self,
        forecast: np.ndarray,
        context: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> tuple[float, list[str]]:
        """Check for negative consumption values."""
        violations = np.sum(forecast < 0)
        if violations > 0:
            penalty = -1.0 * (violations / len(forecast))
            min_value = np.min(forecast)
            messages = [
                f"Negative consumption: {violations}/{len(forecast)} intervals",
                f"Minimum value: {min_value:.3f} kWh",
            ]
            return penalty, messages
        return 0.0, []


class HouseholdMaxConstraint(Constraint):
    """UK household fuse rating limit (BS 7671:2018).

    Typical max: 7.5 kWh/30min (15 kW continuous)
    Absolute max: 50 kWh/30min (100 A fuse @ 230V)
    """

    def __init__(self, typical_max_kwh: float = 7.5, absolute_max_kwh: float = 50.0):
        """Initialize with UK standard limits.

        Args:
            typical_max_kwh: Typical household maximum (default 7.5 kWh/30min)
            absolute_max_kwh: Absolute physics limit (default 50 kWh/30min)
        """
        self.typical_max = typical_max_kwh
        self.absolute_max = absolute_max_kwh

    def evaluate(
        self,
        forecast: np.ndarray,
        context: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> tuple[float, list[str]]:
        """Check household consumption limits."""
        violations = []
        penalty = 0.0

        # Absolute violations (physics impossible)
        absolute_violations = np.sum(forecast > self.absolute_max)
        if absolute_violations > 0:
            penalty = -1.0  # Maximum penalty for physics violation
            max_value = np.max(forecast)
            violations.append(
                f"Physics violation: {absolute_violations} intervals > {self.absolute_max} kWh "
                f"(max: {max_value:.2f} kWh)"
            )

        # Typical violations (suspicious but possible)
        typical_violations = np.sum(
            (forecast > self.typical_max) & (forecast <= self.absolute_max)
        )
        if typical_violations > 0:
            penalty += -0.5 * (typical_violations / len(forecast))
            violations.append(
                f"Unusual consumption: {typical_violations} intervals > {self.typical_max} kWh"
            )

        return max(penalty, -1.0), violations


class RampRateConstraint(Constraint):
    """Realistic limits on consumption change rates.

    Sudden changes in consumption should be physically plausible based on
    typical household appliances and their switching behavior.
    """

    def __init__(self, max_ramp_kwh_per_interval: float = 5.0):
        """Initialize ramp rate constraint.

        Args:
            max_ramp_kwh_per_interval: Maximum change between consecutive intervals
        """
        self.max_ramp = max_ramp_kwh_per_interval

    def evaluate(
        self,
        forecast: np.ndarray,
        context: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> tuple[float, list[str]]:
        """Check for unrealistic ramp rates."""
        if len(forecast) < 2:
            return 0.0, []

        ramp_rates = np.abs(np.diff(forecast))
        violations_count = np.sum(ramp_rates > self.max_ramp)

        if violations_count > 0:
            penalty = -0.3 * (violations_count / len(ramp_rates))
            max_ramp = np.max(ramp_rates)
            avg_ramp = np.mean(ramp_rates[ramp_rates > self.max_ramp])
            messages = [
                f"Excessive ramp rate: {violations_count} intervals",
                f"Max ramp: {max_ramp:.2f} kWh/30min (limit: {self.max_ramp})",
                f"Average violation: {avg_ramp:.2f} kWh/30min",
            ]
            return penalty, messages
        return 0.0, []


class TemporalPatternConstraint(Constraint):
    """Validate temporal patterns match typical consumption profiles.

    Checks that daily/weekly patterns are realistic based on historical data.
    """

    def __init__(
        self,
        min_daily_consumption: float = 2.0,
        max_daily_consumption: float = 50.0,
        night_day_ratio_range: tuple[float, float] = (0.3, 0.8),
    ):
        """Initialize temporal pattern constraints.

        Args:
            min_daily_consumption: Minimum daily total (kWh)
            max_daily_consumption: Maximum daily total (kWh)
            night_day_ratio_range: Expected range for night/day consumption ratio
        """
        self.min_daily = min_daily_consumption
        self.max_daily = max_daily_consumption
        self.night_day_range = night_day_ratio_range

    def evaluate(
        self,
        forecast: np.ndarray,
        context: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> tuple[float, list[str]]:
        """Check temporal pattern plausibility."""
        violations = []
        penalty = 0.0

        # Calculate daily totals (48 intervals per day)
        n_days = len(forecast) // 48
        if n_days == 0:
            return 0.0, []

        daily_totals = []
        for day in range(n_days):
            start_idx = day * 48
            end_idx = start_idx + 48
            if end_idx <= len(forecast):
                daily_total = np.sum(forecast[start_idx:end_idx])
                daily_totals.append(daily_total)

        daily_totals = np.array(daily_totals)

        # Check daily consumption bounds
        low_days = np.sum(daily_totals < self.min_daily)
        high_days = np.sum(daily_totals > self.max_daily)

        if low_days > 0:
            penalty += -0.2 * (low_days / len(daily_totals))
            violations.append(
                f"Abnormally low daily consumption: {low_days} days < {self.min_daily} kWh"
            )

        if high_days > 0:
            penalty += -0.2 * (high_days / len(daily_totals))
            violations.append(
                f"Abnormally high daily consumption: {high_days} days > {self.max_daily} kWh"
            )

        # Check night/day ratio if we have full days
        if n_days > 0 and timestamps is not None:
            # Define night hours (23:00 - 06:00) and day hours (07:00 - 22:00)
            night_consumption = 0
            day_consumption = 0

            for i, ts in enumerate(timestamps[: len(forecast)]):
                hour = ts.hour if hasattr(ts, "hour") else (i % 48) / 2
                if hour >= 23 or hour < 6:
                    night_consumption += forecast[i]
                elif 7 <= hour <= 22:
                    day_consumption += forecast[i]

            if day_consumption > 0:
                night_day_ratio = night_consumption / day_consumption
                if night_day_ratio < self.night_day_range[0]:
                    penalty += -0.1
                    violations.append(
                        f"Unusual night/day ratio: {night_day_ratio:.2f} "
                        f"(expected: {self.night_day_range[0]}-{self.night_day_range[1]})"
                    )
                elif night_day_ratio > self.night_day_range[1]:
                    penalty += -0.1
                    violations.append(
                        f"Unusual night/day ratio: {night_day_ratio:.2f} "
                        f"(expected: {self.night_day_range[0]}-{self.night_day_range[1]})"
                    )

        return max(penalty, -1.0), violations


class PowerFactorConstraint(Constraint):
    """Validate power factor remains within SSEN G59/3 limits.

    UK standard requires power factor between 0.95 lagging and unity.
    """

    def __init__(self, min_power_factor: float = 0.95):
        """Initialize power factor constraint.

        Args:
            min_power_factor: Minimum acceptable power factor (default 0.95)
        """
        self.min_pf = min_power_factor

    def evaluate(
        self,
        forecast: np.ndarray,
        context: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> tuple[float, list[str]]:
        """Check power factor compliance.

        Note: This is a simplified check as we don't have reactive power data.
        We assume violations only for very high consumption spikes.
        """
        # Simplified: assume power factor degrades with very high consumption
        high_load_threshold = 10.0  # kWh per 30min
        high_load_intervals = forecast > high_load_threshold

        if np.any(high_load_intervals):
            violation_rate = np.sum(high_load_intervals) / len(forecast)
            if violation_rate > 0.1:  # More than 10% high load
                penalty = -0.2 * violation_rate
                messages = [
                    f"Potential power factor violation: {np.sum(high_load_intervals)} "
                    f"intervals with load > {high_load_threshold} kWh"
                ]
                return penalty, messages

        return 0.0, []


class VoltageConstraint(Constraint):
    """Validate voltage remains within SSEN limits.

    UK standard: 230V +10%/-6% (216.2V to 253V)
    """

    def __init__(
        self,
        nominal_voltage: float = 230.0,
        tolerance: tuple[float, float] = (-0.06, 0.10),
    ):
        """Initialize voltage constraint.

        Args:
            nominal_voltage: Nominal UK voltage (230V)
            tolerance: Lower and upper tolerance (default -6%, +10%)
        """
        self.nominal = nominal_voltage
        self.min_voltage = nominal_voltage * (1 + tolerance[0])
        self.max_voltage = nominal_voltage * (1 + tolerance[1])

    def evaluate(
        self,
        forecast: np.ndarray,
        context: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> tuple[float, list[str]]:
        """Check voltage compliance through load analysis.

        Note: This is indirect validation based on load levels that might
        cause voltage violations in typical distribution networks.
        """
        # Estimate voltage drop based on load (simplified model)
        # Assume 1% voltage drop per 5 kWh/30min load
        estimated_voltage_drop = forecast / 5.0 * 0.01 * self.nominal
        estimated_voltage = self.nominal - estimated_voltage_drop

        low_voltage_violations = np.sum(estimated_voltage < self.min_voltage)

        if low_voltage_violations > 0:
            penalty = -0.3 * (low_voltage_violations / len(forecast))
            min_est_voltage = np.min(estimated_voltage)
            messages = [
                f"Potential voltage violations: {low_voltage_violations} intervals",
                f"Minimum estimated voltage: {min_est_voltage:.1f}V (limit: {self.min_voltage:.1f}V)",
            ]
            return penalty, messages

        return 0.0, []


class VerifierAgent:
    """Composite constraint validator with weighted rewards."""

    def __init__(self, ssen_constraints_path: str):
        """Initialize verifier with SSEN constraints.

        Args:
            ssen_constraints_path: Path to SSEN constraints JSON file
        """
        self.constraints_path = ssen_constraints_path
        self.constraints = self._initialize_constraints()
        self.weights = {
            "non_negativity": 1.0,  # Hard constraint
            "household_max": 1.0,  # Hard constraint
            "ramp_rate": 0.5,  # Soft constraint
            "temporal_pattern": 0.3,  # Soft constraint
            "power_factor": 0.4,  # Soft constraint
            "voltage": 0.6,  # Medium constraint
        }

        logger.info(
            f"Initialized VerifierAgent with {len(self.constraints)} constraints"
        )

    def _initialize_constraints(self) -> dict[str, Constraint]:
        """Load SSEN constraints and create constraint objects."""
        try:
            with open(self.constraints_path) as f:
                ssen_data = json.load(f)
        except FileNotFoundError:
            logger.warning(f"SSEN constraints file not found: {self.constraints_path}")
            logger.warning("Using default constraint values")
            ssen_data = {
                "household_limits": {
                    "typical_max_kwh_30min": 7.5,
                    "absolute_max_kwh_30min": 50.0,
                },
                "voltage_limits": {
                    "nominal_v": 230,
                    "min_percent": -6,
                    "max_percent": 10,
                },
                "power_factor": {"min_lagging": 0.95},
            }

        constraints = {
            "non_negativity": NonNegativityConstraint(),
            "household_max": HouseholdMaxConstraint(
                typical_max_kwh=ssen_data["household_limits"]["typical_max_kwh_30min"],
                absolute_max_kwh=ssen_data["household_limits"][
                    "absolute_max_kwh_30min"
                ],
            ),
            "ramp_rate": RampRateConstraint(max_ramp_kwh_per_interval=5.0),
            "temporal_pattern": TemporalPatternConstraint(),
            "power_factor": PowerFactorConstraint(
                min_power_factor=ssen_data["power_factor"]["min_lagging"]
            ),
            "voltage": VoltageConstraint(
                nominal_voltage=ssen_data["voltage_limits"]["nominal_v"],
                tolerance=(
                    ssen_data["voltage_limits"]["min_percent"] / 100,
                    ssen_data["voltage_limits"]["max_percent"] / 100,
                ),
            ),
        }

        return constraints

    def evaluate(
        self,
        forecast: np.ndarray,
        scenario: Optional["ScenarioProposal"] = None,
        timestamps: np.ndarray | None = None,
        return_details: bool = False,
    ) -> float | tuple[float, dict]:
        """Evaluate forecast against all constraints.

        Args:
            forecast: Predicted consumption values
            scenario: Scenario proposal (optional)
            timestamps: Timestamps for temporal constraints (optional)
            return_details: Whether to return detailed breakdown

        Returns:
            reward: Combined reward score in [-1, +1]
                   -1 = maximum violation
                    0 = perfect compliance
                   +1 = perfect + difficulty bonus
            details: (Optional) Per-constraint scores and violations
        """
        total_penalty = 0.0
        details = {}

        # Get context from scenario if available
        context = scenario.baseline_context if scenario else None

        for name, constraint in self.constraints.items():
            score, violations = constraint.evaluate(forecast, context, timestamps)
            weighted_score = score * self.weights[name]
            total_penalty += weighted_score

            details[name] = {
                "score": score,
                "violations": violations,
                "weight": self.weights[name],
                "weighted_score": weighted_score,
            }

        # Normalize to [-1, 0] range
        reward = max(total_penalty, -1.0)

        # Optional difficulty bonus (reward challenging but valid scenarios)
        if scenario is not None and reward == 0.0 and scenario.difficulty_score > 0.5:
            bonus = 0.1 * scenario.difficulty_score
            reward += bonus
            details["difficulty_bonus"] = {
                "score": bonus,
                "violations": [],
                "weight": 1.0,
                "weighted_score": bonus,
            }

        if return_details:
            return reward, details
        return reward

    def validate_single_constraint(
        self,
        constraint_name: str,
        forecast: np.ndarray,
        context: np.ndarray | None = None,
    ) -> tuple[float, list[str]]:
        """Validate forecast against a single named constraint.

        Args:
            constraint_name: Name of constraint to check
            forecast: Predicted consumption values
            context: Historical context (optional)

        Returns:
            (score, violations) tuple
        """
        if constraint_name not in self.constraints:
            raise ValueError(f"Unknown constraint: {constraint_name}")

        return self.constraints[constraint_name].evaluate(forecast, context)

    def get_constraint_summary(self) -> dict[str, dict[str, float]]:
        """Get summary of all constraints and their weights.

        Returns:
            Dictionary mapping constraint names to their properties
        """
        summary = {}
        for name, constraint in self.constraints.items():
            summary[name] = {
                "weight": self.weights[name],
                "type": constraint.__class__.__name__,
                "is_hard": self.weights[name] >= 1.0,
            }
        return summary
