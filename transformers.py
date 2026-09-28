"""
Custom sklearn transformers for the house price pipeline.

They live in their own module (not __main__) so that joblib/pickle can
reload artifacts/house_model.joblib from any script (e.g. app.py).
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileCapper(BaseEstimator, TransformerMixin):
    """Clip selected columns to [q_low, q_high] quantiles learned on train.

    Fit inside the Pipeline -> the bounds come only from training data
    (no leakage), and transform() applies them everywhere else.
    """

    def __init__(self, columns: list[str], q_low: float = 0.01, q_high: float = 0.99):
        self.columns = columns
        self.q_low = q_low
        self.q_high = q_high

    def fit(self, X: pd.DataFrame, y=None):
        X_ = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns)
        self.bounds_ = {}
        for c in self.columns:
            lo, hi = X_[c].quantile([self.q_low, self.q_high])
            self.bounds_[c] = (float(lo), float(hi))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            X[c] = X[c].clip(lo, hi)
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds rooms-per-household and bedrooms-per-room ratio features."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["RoomsPerHousehold"] = X["AveRooms"] / X["AveOccup"].clip(lower=0.5)
        X["BedroomsPerRoom"] = X["AveBedrms"] / X["AveRooms"].clip(lower=0.5)
        return X
