"""Section 8 CatBoost regression of Delta VX.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

_SEED = 20260525
_CONFIG = dict(
    iterations=500,
    depth=4,
    learning_rate=0.03,
    l2_leaf_reg=10.0,
    loss_function="RMSE",
    eval_metric="MAE",
    thread_count=1,
    allow_writing_files=False,
    verbose=False,
)


def fit_regression(
    X: pd.DataFrame,
    y,
    *,
    eval_set: tuple | None = None,
    early_stopping_rounds: int | None = None,
    seed: int = _SEED,
    **over,
) -> CatBoostRegressor:
    """Fit a CatBoostRegressor of Delta VX; optional early-stop on `eval_set=(Xval,yval)`."""
    model = CatBoostRegressor(random_seed=seed, **{**_CONFIG, **over})
    extra: dict = {}
    if eval_set is not None:
        Xval, yval = eval_set
        extra["eval_set"] = (Xval, np.asarray(yval, float))
        extra["use_best_model"] = True
        if early_stopping_rounds is not None:
            extra["early_stopping_rounds"] = int(early_stopping_rounds)
    model.fit(X, np.asarray(y, float), **extra)
    return model


def predict(model: CatBoostRegressor, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X), float)
