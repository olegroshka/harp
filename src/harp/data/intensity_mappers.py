"""Investment intensity mappers for the SMIM framework.

Each mapper implements the ``InvestmentIntensityMapper`` protocol from
``smim/interfaces.py`` and normalises a raw actor-specific observable to
the interval $[0, 1]$ (or NaN for missing observations).

Assumption A2 (Typed comparability) requires that normalisation is
performed **within** each ActorType so that $y_{i,t}$ values are
comparable across actors of the same type but not across types.

Mapper conventions
------------------
- ``raw_data``: a ``pd.DataFrame`` with a **date index** and **actor_ids
  as columns**.  Each cell contains the pre-computed raw metric for that
  actor at that date.  NaN cells are allowed and propagated.
- ``actor``: the specific ``Actor`` whose intensity series is returned.
- Returns: ``pd.Series`` indexed by date, values in $[0, 1]$ (or NaN).

Three concrete mappers are provided here:

CorporateCapexMapper
    Normalises CapEx/Assets ratio to [0,1] by **cross-sectional
    percentile rank** at each date (higher CapEx/Assets → higher
    intensity).  All columns in ``raw_data`` are treated as the
    cross-section; only the target actor's column is returned.

BankCreditMapper
    Normalises YoY asset growth rate to [0,1] via **cross-sectional
    percentile rank** at each date (same approach as CorporateCapexMapper).
    ``raw_data`` must contain all peer bank actors.

AgencyBudgetMapper
    Normalises budget-share values to [0,1] via **min-max** across the
    cross-section at each date (all columns in ``raw_data`` treated as
    the reference distribution).

Usage::

    from harp.data.intensity_mappers import (
        CorporateCapexMapper, BankCreditMapper, AgencyBudgetMapper,
        MapperRegistry,
    )
    mapper = MapperRegistry().get(ActorType.LARGE_FIRM)
    y = mapper.compute(raw_data, actor)
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from harp.interfaces import Actor, ActorType


# ═══════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════

def _sigmoid(x: pd.Series) -> pd.Series:
    """Element-wise sigmoid: σ(x) = 1 / (1 + exp(-x)).  NaN-safe."""
    return 1.0 / (1.0 + np.exp(-x))


def _cross_section_percentile_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise (cross-sectional) percentile rank, scaled to [0, 1].

    Ties are averaged.  Rows with all-NaN are left as NaN.
    Returns a DataFrame of the same shape as ``df``.
    """
    return df.rank(axis=1, pct=True, na_option="keep")


def _min_max_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise min-max normalisation to [0, 1].

    At each date, maps the minimum value to 0 and the maximum to 1.
    Rows where min == max are mapped to 0.5 (no information).
    Source NaN values are propagated (not replaced by 0.5).
    """
    row_min = df.min(axis=1)
    row_max = df.max(axis=1)
    span = row_max - row_min
    normalised = df.subtract(row_min, axis=0).divide(
        span.replace(0.0, np.nan), axis=0
    )
    # For constant rows (span==0) only fill cells that were NOT NaN in the source
    constant_rows = span == 0.0
    if constant_rows.any():
        source_not_nan = ~df.loc[constant_rows].isna()
        normalised.loc[constant_rows] = normalised.loc[constant_rows].where(
            ~source_not_nan, 0.5
        )
    return normalised


# ═══════════════════════════════════════════════════════════
# Concrete mappers
# ═══════════════════════════════════════════════════════════

class CorporateCapexMapper:
    """Normalises CapEx/Assets to [0, 1] via cross-sectional percentile rank.

    Intended for ``ActorType.LARGE_FIRM`` and ``ActorType.SECTOR_LEADER``.

    ``raw_data`` must have actor_ids as columns and dates as index;
    each cell should be the CapEx/Assets ratio (a non-negative float).
    Higher ratios receive higher normalised intensity.
    """

    @property
    def actor_type(self) -> ActorType:
        return ActorType.LARGE_FIRM

    def compute(
        self,
        raw_data: pd.DataFrame,
        actor: Actor,
    ) -> pd.Series:
        """Return cross-sectionally ranked intensity for ``actor``.

        Args:
            raw_data: DataFrame (dates × actor_ids) of CapEx/Assets ratios.
            actor: The actor whose intensity series to return.

        Returns:
            pd.Series indexed by date, values in [0, 1] (NaN where raw is NaN).

        Raises:
            KeyError: If actor.actor_id is not a column in raw_data.
        """
        if actor.actor_id not in raw_data.columns:
            raise KeyError(
                f"Actor {actor.actor_id!r} not found in raw_data columns."
            )
        ranked = _cross_section_percentile_rank(raw_data)
        series = ranked[actor.actor_id].rename(actor.actor_id)
        series.index = raw_data.index
        return series


class BankCreditMapper:
    """Normalises YoY asset growth rate to [0, 1] via cross-sectional percentile rank.

    Intended for ``ActorType.BANK``.

    ``raw_data`` must have actor_ids as columns and dates as index;
    each cell should be the YoY (4-quarter) asset growth rate.  At each
    date the row is ranked cross-sectionally across **all bank actors in
    the panel**, so the result is a relative measure of investment intensity
    within the bank peer group.  A bank that consistently grows faster than
    its peers receives higher intensity; a bank that shrinks or grows slowly
    receives lower intensity.

    This approach mirrors ``CorporateCapexMapper`` and guarantees rank
    stability by construction for any persistent differences in bank growth
    levels.

    Output is in [0, 1] inclusive (can reach exactly 0.0 and 1.0).
    Missing values (NaN) are propagated.

    Important: ``raw_data`` must contain the full cross-section of bank
    actors, not just the target actor, for the rank to be meaningful.
    """

    @property
    def actor_type(self) -> ActorType:
        return ActorType.BANK

    def compute(
        self,
        raw_data: pd.DataFrame,
        actor: Actor,
    ) -> pd.Series:
        """Return cross-sectionally ranked intensity for ``actor``.

        Args:
            raw_data: DataFrame (dates × actor_ids) of YoY asset growth rates.
                      Must contain all peer bank actors for meaningful ranking.
            actor: The actor whose intensity series to return.

        Returns:
            pd.Series indexed by date, values in [0, 1] (NaN where raw is NaN).

        Raises:
            KeyError: If actor.actor_id is not a column in raw_data.
        """
        if actor.actor_id not in raw_data.columns:
            raise KeyError(
                f"Actor {actor.actor_id!r} not found in raw_data columns."
            )
        ranked = _cross_section_percentile_rank(raw_data)
        return ranked[actor.actor_id].rename(actor.actor_id)


class AgencyBudgetMapper:
    """Normalises budget-share values to [0, 1] via cross-sectional min-max.

    Intended for ``ActorType.REGULATOR``, ``ActorType.INTL_ORG``, and
    ``ActorType.MUNICIPALITY``.

    ``raw_data`` must have actor_ids as columns and dates as index;
    each cell should be the actor's budget share or spending level.
    At each date the row is min-max normalised over all actors present,
    so the result is a relative measure within the group.
    """

    @property
    def actor_type(self) -> ActorType:
        return ActorType.REGULATOR

    def compute(
        self,
        raw_data: pd.DataFrame,
        actor: Actor,
    ) -> pd.Series:
        """Return min-max normalised intensity for ``actor``.

        Args:
            raw_data: DataFrame (dates × actor_ids) of budget-share values.
            actor: The actor whose intensity series to return.

        Returns:
            pd.Series indexed by date, values in [0, 1] (NaN where raw is NaN).

        Raises:
            KeyError: If actor.actor_id is not a column in raw_data.
        """
        if actor.actor_id not in raw_data.columns:
            raise KeyError(
                f"Actor {actor.actor_id!r} not found in raw_data columns."
            )
        normalised = _min_max_cross_section(raw_data)
        return normalised[actor.actor_id].rename(actor.actor_id)


# ═══════════════════════════════════════════════════════════
# Mapper registry
# ═══════════════════════════════════════════════════════════

class MapperRegistry:
    """Maps ActorType → InvestmentIntensityMapper instance.

    Provides a single ``get(actor_type)`` entry point.  Unmapped types
    raise ``KeyError`` so missing coverage is detected early (A2 check).

    The default registry covers the actor types present in the MVP energy
    taxonomy.  Extend by calling ``register(actor_type, mapper)`` before
    use.
    """

    def __init__(self) -> None:
        _capex = CorporateCapexMapper()
        _credit = BankCreditMapper()
        _budget = AgencyBudgetMapper()

        self._registry: dict[ActorType, object] = {
            # Layer 2 — transmission
            ActorType.LARGE_FIRM:    _capex,
            ActorType.SECTOR_LEADER: _capex,
            ActorType.BANK:          _credit,
            # Layer 1 — upstream
            ActorType.REGULATOR:     _budget,
            ActorType.INTL_ORG:      _budget,
            ActorType.CENTRAL_BANK:  _budget,
            ActorType.THINK_TANK:    _budget,
            # Layer 3 — downstream
            ActorType.SME:           _capex,
            ActorType.MUNICIPALITY:  _budget,
            # Layer 3 — additional downstream types
            ActorType.HOUSEHOLD:       _budget,
            ActorType.RETAIL_INVESTOR: _budget,
            # Layer 0 — exogenous (shocks are not investment actors, but
            # include for completeness; callers should use a dedicated shock
            # adapter instead)
            ActorType.GLOBAL_SHOCK:    _budget,
        }

    def get(self, actor_type: ActorType) -> object:
        """Return the mapper for the given ActorType.

        Args:
            actor_type: One of the ActorType enum values.

        Returns:
            The registered mapper instance (satisfies InvestmentIntensityMapper).

        Raises:
            KeyError: If no mapper is registered for actor_type.
        """
        try:
            return self._registry[actor_type]
        except KeyError:
            raise KeyError(
                f"No InvestmentIntensityMapper registered for {actor_type!r}. "
                "Register one via MapperRegistry.register() before use."
            ) from None

    def register(self, actor_type: ActorType, mapper: object) -> None:
        """Register or replace the mapper for actor_type.

        Args:
            actor_type: ActorType enum value.
            mapper: Object satisfying the InvestmentIntensityMapper protocol.
        """
        self._registry[actor_type] = mapper

    def registered_types(self) -> list[ActorType]:
        """Return all registered ActorType values."""
        return list(self._registry.keys())
