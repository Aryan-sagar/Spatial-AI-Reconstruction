"""Uncertainty model: interval half-width grows as evidence weakens (tier prior, fit noise, low support) and, when a
calibration file (empirical error quantiles from ground truth) is supplied, uses it instead of the prior."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STATUSES = ("observed", "estimated", "unobserved")


def conformal_quantile(abs_residuals, alpha: float) -> float:
    """Split-conformal quantile with the finite-sample (n+1) correction."""
    r = np.sort(np.asarray(abs_residuals, float))
    n = len(r)
    if n == 0:
        raise ValueError("no residuals")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(r[min(k, n) - 1])


class UncertaintyModel:
    def __init__(self, cfg: dict, tier: str):
        u = cfg["uncertainty"]
        self.level, self.z, self.tier = u["confidence_level"], u["z"], tier
        self.prior = u["priors"][tier]
        self.inflate = u["low_support_inflation"]
        self.cal = None
        if u.get("calibration_file"):
            self.cal = json.loads(Path(u["calibration_file"]).read_text())

    def interval(self, kind: str, value: float, fit_sigma: float = 0.0, n_support: int | None = None, min_support: int = 200) -> dict:
        basis, low_cal = "uncalibrated_prior", None
        entry = (self.cal or {}).get(self.tier, {}).get(kind)
        if entry:
            hw, n = float(entry["quantile"]), int(entry["n"])
            basis = f"empirical_quantile(n={n})"
            if n < 20:
                hw *= self.inflate
                basis += "+small_sample_inflation"
        else:
            rel = self.prior["rel"] * (2.0 if kind.endswith("area") else 1.0)
            hw = self.prior["abs_m"] + rel * abs(value) + self.z * fit_sigma
        if n_support is not None and n_support < min_support:
            hw *= self.inflate
            basis += "+low_support_inflation"
        return {"lower": float(value - hw), "upper": float(value + hw), "confidence_level": self.level, "half_width": float(hw), "basis": basis}

    def measurement(self, kind: str, value: float, unit: str, method: str, status: str = "observed", fit_sigma: float = 0.0,
                    n_support: int | None = None, artifacts: list[str] | None = None) -> dict:
        assert status in STATUSES
        ci = self.interval(kind, value, fit_sigma, n_support)
        return {"value": float(value), "unit": unit,
                "confidence_interval": {"lower": ci["lower"], "upper": ci["upper"], "confidence_level": ci["confidence_level"]},
                "status": status, "method": method, "supporting_artifacts": artifacts or [], "uncertainty_basis": ci["basis"]}

    def bounded(self, value: float, lower: float, upper: float, unit: str, method: str, basis: str, reasons: list[str],
                artifacts: list[str] | None = None) -> dict:
        """Status 'estimated' measurement whose interval is [lower, upper] from evidence bounds (widened to contain the value), flagged
        unreliable with the reasons. Used when a quality check says the tier-prior interval would be confident garbage."""
        lo, hi = min(lower, value), max(upper, value)
        return {"value": float(value), "unit": unit, "confidence_interval": {"lower": float(lo), "upper": float(hi), "confidence_level": self.level},
                "status": "estimated", "method": method, "supporting_artifacts": artifacts or [], "uncertainty_basis": basis,
                "reliable": False, "unreliable_reasons": list(reasons)}

    @staticmethod
    def unobserved(unit: str, method: str, reason: str) -> dict:
        return {"value": None, "unit": unit, "confidence_interval": None, "status": "unobserved", "method": method,
                "supporting_artifacts": [], "reason": reason}
