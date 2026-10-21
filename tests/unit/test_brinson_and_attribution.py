"""Brinson attribution unit coverage (single-period Series API)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attribution.brinson import BrinsonAttribution


@pytest.mark.parametrize("n_sectors", [2, 3, 4, 5, 8, 10])
def test_brinson_effects_identity(n_sectors):
    sectors = [f"S{i}" for i in range(n_sectors)]
    rng = np.random.default_rng(n_sectors)
    port_w = pd.Series(rng.dirichlet(np.ones(n_sectors)), index=sectors)
    bench_w = pd.Series(1 / n_sectors, index=sectors)
    port_r = pd.Series(rng.normal(0.01, 0.02, size=n_sectors), index=sectors)
    bench_r = pd.Series(rng.normal(0.008, 0.015, size=n_sectors), index=sectors)
    out = BrinsonAttribution(port_r, bench_r, port_w, bench_w).calculate_attribution()
    assert set(out) >= {"allocation", "selection", "interaction", "total"}
    recon = out["allocation"] + out["selection"] + out["interaction"]
    assert abs(recon - out["total"]) < 1e-9


def test_brinson_equal_weights_selection_only():
    sectors = ["A", "B", "C"]
    w = pd.Series(1 / 3, index=sectors)
    port_r = pd.Series([0.02, 0.01, -0.005], index=sectors)
    bench_r = pd.Series([0.01, 0.01, 0.01], index=sectors)
    out = BrinsonAttribution(port_r, bench_r, w, w).calculate_attribution()
    assert abs(out["allocation"]) < 1e-12
    assert abs(out["total"] - (out["selection"] + out["interaction"])) < 1e-12


@pytest.mark.parametrize(
    "port_w,bench_w",
    [
        ([0.5, 0.5], [0.5, 0.5]),
        ([0.7, 0.3], [0.5, 0.5]),
        ([1.0, 0.0], [0.5, 0.5]),
        ([0.2, 0.8], [0.8, 0.2]),
    ],
)
def test_brinson_two_asset_grid(port_w, bench_w):
    idx = ["X", "Y"]
    out = BrinsonAttribution(
        pd.Series([0.03, -0.01], index=idx),
        pd.Series([0.01, 0.01], index=idx),
        pd.Series(port_w, index=idx),
        pd.Series(bench_w, index=idx),
    ).calculate_attribution()
    assert np.isfinite(out["total"])
