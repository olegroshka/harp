"""Interface definitions for the replication package."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd


# ── Enums ──────────────────────────────────────────────────

class ActorType(str, Enum):
    """Actor type classification."""
    GLOBAL_SHOCK = "global_shock"
    CENTRAL_BANK = "central_bank"
    REGULATOR = "regulator"
    THINK_TANK = "think_tank"
    INTL_ORG = "intl_org"
    LARGE_FIRM = "large_firm"
    BANK = "bank"
    SECTOR_LEADER = "sector_leader"
    SME = "sme"
    MUNICIPALITY = "municipality"
    HOUSEHOLD = "household"
    RETAIL_INVESTOR = "retail_investor"


class Layer(int, Enum):
    """Four-layer hierarchy."""
    EXOGENOUS = 0
    UPSTREAM = 1
    TRANSMISSION = 2
    DOWNSTREAM = 3


class DecompositionMethod(str, Enum):
    """Spectral decomposition methods."""
    SCHUR = "schur"
    DIRECTED_VARIATION = "directed_variation"
    POLAR = "polar"
    HERMITIAN_DILATION = "hermitian_dilation"
    DMD = "dmd"
    EXTENDED_DMD = "extended_dmd"


@dataclass
class ModalFrame:
    """Result of spectral decomposition: modal basis U_t in R^{N x K}."""
    basis: NDArray[np.floating]               # (N, K)
    eigenvalues: NDArray[np.complexfloating]   # (K,)
    method: DecompositionMethod
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def K(self) -> int:
        return self.basis.shape[1]

    @property
    def N(self) -> int:
        return self.basis.shape[0]


@dataclass(frozen=True)
class Actor:
    """Single actor in the investment system."""
    actor_id: str
    name: str
    actor_type: ActorType
    layer: Layer
    geography: str
    sector: str
    external_ids: dict[str, str] = field(default_factory=dict)


@dataclass
class DateRange:
    """Time range for estimation or evaluation."""
    start: pd.Timestamp
    end: pd.Timestamp
    frequency: str = "QS"
