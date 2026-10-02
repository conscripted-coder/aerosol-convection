"""Repeated-split linear regression of environmental predictors on radar/microphysical
labels.
"""
# Every label shares the same ``n_bootstraps`` 70/30 splits (random_state = split index, so
# deterministic). Nothing is resampled with replacement, so the 2.5/97.5 percentiles are the
# range 95% of splits fall inside, not a bootstrap confidence interval.


import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split


# Exported so callers can label axes and size arrays without restating them
DEFAULT_PERCENTILES  = (2.5, 25, 50, 75, 97.5)
DEFAULT_N_BOOTSTRAPS = 5000
DEFAULT_TRAIN_SIZE   = 0.7


def fit_bootstrap_regression(
    df,
    predictors,
    labels,
    *,
    standardize_x=True,
    standardize_y=True,
    score_on='test',
    n_bootstraps=DEFAULT_N_BOOTSTRAPS,
    train_size=DEFAULT_TRAIN_SIZE,
    percentiles=DEFAULT_PERCENTILES,
):
    """Model ``labels`` (y) on ``predictors`` (X) over an ensemble of train/test splits.

    ``df`` is the per-track feature set, free of NaN/Inf in the ``predictors`` and
    ``labels`` columns. ``standardize_x`` / ``standardize_y`` z-score on the train split
    only; ``score_on`` ('test' or 'train') is the split R^2 is computed on. The scheme
    is recorded in ``.attrs``.

    Returns a Dataset of ``coefficient`` (percentile, label, predictor) and
    ``r_squared`` (percentile, label).
    """
    if score_on not in ('test', 'train'):
        raise ValueError(
            f"score_on must be 'test' or 'train', got {score_on!r}."
        )

    predictors  = list(predictors)
    labels      = list(labels)
    percentiles = list(percentiles)

    # float64: sources are float32, and unstandardized X is ill-conditioned (CCN ~1000s
    # beside RH ~10s), which amplifies float32 round-off into the coefficients
    x_all = df[predictors].to_numpy(dtype=float)
    y_all = df[labels].to_numpy(dtype=float)

    n_obs   = x_all.shape[0]
    n_pct   = len(percentiles)
    n_pred  = len(predictors)
    n_label = len(labels)

    boot_coefs = np.empty((n_bootstraps, n_label, n_pred))
    boot_r2    = np.empty((n_bootstraps, n_label))

    # Split row indices so one lstsq solve per split fits every label (columns of y are
    # independent, so this matches label-by-label fits)
    row_index = np.arange(n_obs)

    for c in range(n_bootstraps):
        train_idx, test_idx = train_test_split(
            row_index, train_size=train_size, random_state=c
        )
        X_train, X_test = x_all[train_idx], x_all[test_idx]
        Y_train, Y_test = y_all[train_idx], y_all[test_idx]

        # z-score with train-split mean and std (ddof=0); std 0 -> 1 avoids divide-by-zero
        if standardize_x:
            x_mu    = X_train.mean(axis=0)
            x_sd    = X_train.std(axis=0)
            x_sd    = np.where(x_sd == 0, 1.0, x_sd)
            X_train = (X_train - x_mu) / x_sd
            X_test  = (X_test - x_mu) / x_sd

        if standardize_y:
            y_mu    = Y_train.mean(axis=0)
            y_sd    = Y_train.std(axis=0)
            y_sd    = np.where(y_sd == 0, 1.0, y_sd)
            Y_train = (Y_train - y_mu) / y_sd
            Y_test  = (Y_test - y_mu) / y_sd

        # Leading column of ones is the intercept; beta is (1 + n_pred, n_label)
        design_train  = np.column_stack([np.ones(len(train_idx)), X_train])
        beta, *_      = np.linalg.lstsq(design_train, Y_train, rcond=None)
        boot_coefs[c] = beta[1:].T  # drop the intercept row

        if score_on == 'test':
            design_score, Y_score = (
                np.column_stack([np.ones(len(test_idx)), X_test]), Y_test
            )
        else:
            design_score, Y_score = design_train, Y_train

        # R^2 per label, about the mean of the scored split
        residual   = Y_score - design_score @ beta
        ss_res     = (residual ** 2).sum(axis=0)
        ss_tot     = ((Y_score - Y_score.mean(axis=0)) ** 2).sum(axis=0)
        boot_r2[c] = 1.0 - ss_res / ss_tot

    coef = np.percentile(boot_coefs, percentiles, axis=0)  # (n_pct, n_label, n_pred)
    r2   = np.percentile(boot_r2, percentiles, axis=0)     # (n_pct, n_label)

    return xr.Dataset(
        {
            'coefficient': (('percentile', 'label', 'predictor'), coef),
            'r_squared':   (('percentile', 'label'), r2),
        },
        coords={
            'percentile': percentiles,
            'label': labels,
            'predictor': predictors,
        },
        attrs={
            'standardize_x': int(standardize_x),
            'standardize_y': int(standardize_y),
            'score_on': score_on,
            'n_bootstraps': n_bootstraps,
            'train_size': train_size,
        },
    )
