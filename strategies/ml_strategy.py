"""Machine-learning strategy path (sklearn momentum classifier)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MLMomentumStrategy:
    """
    Walk-forward ML strategy: predict next-period outperformance vs cross-section median.

    Features per symbol/date: returns(5/10/20), RSI-like rolling mean, volatility.
    """

    def __init__(
        self,
        model: str = "logreg",
        train_window: int = 126,
        min_train: int = 60,
        top_n: int = 3,
        random_state: int = 42,
    ):
        self.model_name = model
        self.train_window = train_window
        self.min_train = min_train
        self.top_n = top_n
        self.random_state = random_state

    def _make_model(self):
        if self.model_name == "gbm":
            clf = GradientBoostingClassifier(random_state=self.random_state)
        else:
            clf = LogisticRegression(max_iter=500, random_state=self.random_state)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    @staticmethod
    def _feature_frame(prices: pd.DataFrame) -> dict:
        rets = prices.pct_change()
        feats = {
            "r5": prices.pct_change(5),
            "r10": prices.pct_change(10),
            "r20": prices.pct_change(20),
            "vol20": rets.rolling(20).std(),
            "ma_gap": prices / prices.rolling(20).mean() - 1.0,
        }
        return feats

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        prices = prices.dropna(how="all")
        feats = self._feature_frame(prices)
        fwd = prices.pct_change().shift(-1)
        cross_med = fwd.median(axis=1)
        labels = fwd.gt(cross_med, axis=0).astype(float)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        symbols = list(prices.columns)

        # Walk-forward by date; retrain periodically for speed
        retrain_every = 5
        model = None
        feature_cols = ["r5", "r10", "r20", "vol20", "ma_gap"]

        for i, dt in enumerate(prices.index):
            if i < self.min_train:
                continue
            if model is None or (i - self.min_train) % retrain_every == 0:
                start = max(0, i - self.train_window)
                rows = []
                y = []
                for sym in symbols:
                    for j in range(start, i):
                        row = [feats[c].iloc[j][sym] for c in feature_cols]
                        lab = labels.iloc[j][sym]
                        if any(pd.isna(row)) or pd.isna(lab):
                            continue
                        rows.append(row)
                        y.append(lab)
                if len(rows) < self.min_train:
                    continue
                model = self._make_model()
                model.fit(np.asarray(rows), np.asarray(y))

            if model is None:
                continue

            scores = {}
            for sym in symbols:
                row = [feats[c].iloc[i][sym] for c in feature_cols]
                if any(pd.isna(row)):
                    continue
                proba = model.predict_proba(np.asarray(row).reshape(1, -1))[0]
                # probability of class 1 (outperform)
                scores[sym] = float(proba[1]) if len(proba) > 1 else float(proba[0])

            if not scores:
                continue
            ranked = sorted(scores, key=scores.get, reverse=True)[: self.top_n]
            w = 1.0 / len(ranked)
            for sym in ranked:
                weights.loc[dt, sym] = w

        return weights.fillna(0.0)
