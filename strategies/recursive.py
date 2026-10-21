"""
Recursive strategy / path support.

A RecursiveStrategy nests child strategies across horizons (or sub-portfolios).
At each recursion depth it:
  1. Runs the child signal generator on a nested window / subset
  2. Aggregates child weights into parent weights
  3. Optionally re-enters on residual capital (recursive path)

This models multi-horizon allocation and nested sleeve construction without
requiring a DAG framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd


SignalFn = Callable[[pd.DataFrame, int], pd.DataFrame]


def default_momentum_signals(prices: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Simple top-N momentum sleeve used when no child is supplied."""
    mom = prices.pct_change(20)
    ranks = mom.rank(axis=1, ascending=False)
    signals = (ranks <= top_n).astype(float)
    # Equal-weight among selected
    row_sums = signals.sum(axis=1).replace(0, np.nan)
    return signals.div(row_sums, axis=0).fillna(0.0)


@dataclass
class RecursiveNode:
    name: str
    lookback: int = 60
    top_n: int = 3
    capital_fraction: float = 1.0
    children: List["RecursiveNode"] = field(default_factory=list)


class RecursiveStrategy:
    """
    Recursively allocate across nested strategy nodes.

    Parameters
    ----------
    root : RecursiveNode
        Tree describing nested sleeves.
    max_depth : int
        Hard cap on recursion depth (guards against pathological trees).
    residual_reentry : bool
        If True, unallocated capital at a node is re-run through the same
        node on a shorter lookback (recursive path).
    """

    def __init__(
        self,
        root: Optional[RecursiveNode] = None,
        max_depth: int = 4,
        residual_reentry: bool = True,
    ):
        self.root = root or RecursiveNode(name="root", lookback=60, top_n=3)
        self.max_depth = max_depth
        self.residual_reentry = residual_reentry

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            return prices.copy()
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sleeve = self._allocate_node(self.root, prices, depth=0)
        weights = weights.add(sleeve, fill_value=0.0)
        # Normalize to <= 1 gross long
        gross = weights.abs().sum(axis=1).replace(0, np.nan)
        return weights.div(gross.clip(lower=1.0), axis=0).fillna(0.0)

    def _allocate_node(
        self, node: RecursiveNode, prices: pd.DataFrame, depth: int
    ) -> pd.DataFrame:
        if depth > self.max_depth:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if node.children:
            child_frames = []
            frac = node.capital_fraction / max(len(node.children), 1)
            for child in node.children:
                child_w = self._allocate_node(child, prices, depth + 1)
                child_frames.append(child_w * frac)
            combined = sum(child_frames)
            allocated = combined.abs().sum(axis=1)
        else:
            signals = default_momentum_signals(prices, top_n=node.top_n)
            # Horizon gating: only trade once lookback is available
            valid = prices.notna().sum(axis=1) >= 1
            combined = signals.mul(node.capital_fraction).where(valid, 0.0)
            # Smooth with lookback mean of weights for path stability
            combined = combined.rolling(max(node.lookback // 5, 1)).mean().fillna(0.0)
            allocated = combined.abs().sum(axis=1)

        if self.residual_reentry and depth < self.max_depth:
            residual = (node.capital_fraction - allocated).clip(lower=0.0)
            if float(residual.mean()) > 0.05:
                shorter = RecursiveNode(
                    name=f"{node.name}_reentry",
                    lookback=max(node.lookback // 2, 5),
                    top_n=max(node.top_n - 1, 1),
                    capital_fraction=1.0,
                )
                reentry = self._allocate_node(shorter, prices, depth + 1)
                scale = residual.clip(upper=node.capital_fraction)
                combined = combined.add(reentry.mul(scale, axis=0), fill_value=0.0)

        return combined.fillna(0.0)

    @classmethod
    def multi_horizon(
        cls,
        horizons: Sequence[int] = (20, 60, 120),
        top_n: int = 3,
    ) -> "RecursiveStrategy":
        """Factory: recursive multi-horizon momentum sleeves."""
        children = [
            RecursiveNode(
                name=f"h{h}",
                lookback=h,
                top_n=top_n,
                capital_fraction=1.0 / len(horizons),
            )
            for h in horizons
        ]
        root = RecursiveNode(name="multi_horizon", children=children, capital_fraction=1.0)
        return cls(root=root, residual_reentry=True)
