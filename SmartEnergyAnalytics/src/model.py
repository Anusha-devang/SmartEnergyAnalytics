"""
model.py
--------
Stage 4: learn what a normal hour looks like, so that the streaming layer can
recognise an abnormal one.

The model predicts total_kwh from the clock, the outside temperature and the
occupancy. Its job is not to be clever; its job is to give the anomaly detector
a trustworthy expectation and an honest error band.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


def build_pipeline():
    """
    Scale, then fit a straight-line model.

    StandardScaler matters because temperature (about 30), occupancy (about 70)
    and the sine terms (between -1 and 1) live on wildly different scales.
    Scaling puts them on equal footing so the coefficients can be compared.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("reg", LinearRegression()),
    ])


def train(df):
    """
    Evaluate with K-fold cross-validation, then refit on everything.

    Why not a plain train/test split? The baseline is a single 24-hour day.
    A chronological split would put the whole evening into the test set and the
    model would never have seen an evening hour. Cross-validation lets every
    hour take a turn as unseen data while still training on the rest.
    """
    X = df[config.FEATURES]
    y = df[config.TARGET]

    pipe = build_pipeline()
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True,
               random_state=config.RANDOM_STATE)

    # Out-of-fold predictions: every row predicted by a model that did not see it
    oof = cross_val_predict(pipe, X, y, cv=cv)
    residuals = y - oof

    metrics = {
        "rows": len(df),
        "folds": config.CV_FOLDS,
        "mae": round(float(mean_absolute_error(y, oof)), 3),
        "rmse": round(float(np.sqrt(np.mean(residuals ** 2))), 3),
        "r2": round(float(r2_score(y, oof)), 3),
        "mape": round(float(np.mean(np.abs(residuals / y)) * 100), 2),
        "sigma": round(float(residuals.std(ddof=1)), 3),
    }

    print("\n" + "=" * 70)
    print("FORECAST MODEL TRAINING")
    print("=" * 70)
    print(f"Rows / folds        : {metrics['rows']} / {metrics['folds']}")
    print(f"MAE                 : {metrics['mae']} kWh")
    print(f"RMSE                : {metrics['rmse']} kWh")
    print(f"MAPE                : {metrics['mape']} %")
    print(f"R-squared           : {metrics['r2']}")
    print(f"Residual sigma      : {metrics['sigma']} kWh  <- the anomaly yardstick")
    print(f"  WATCH band beyond : {config.WATCH_SIGMA * metrics['sigma']:.2f} kWh error")
    print(f"  ALERT band beyond : {config.ALERT_SIGMA * metrics['sigma']:.2f} kWh error")

    # Refit on the complete baseline for deployment
    pipe.fit(X, y)
    joblib.dump({"pipeline": pipe, "sigma": metrics["sigma"], "metrics": metrics},
                config.MODEL_FILE)
    print(f"[model] saved to {config.MODEL_FILE.name}")

    return pipe, metrics


def coefficients(pipe):
    """Read the fitted weights back out; scaled inputs make them comparable."""
    coefs = pipe.named_steps["reg"].coef_
    table = (pd.DataFrame({"feature": config.FEATURES, "coefficient": coefs.round(3)})
             .assign(strength=lambda d: d["coefficient"].abs())
             .sort_values("strength", ascending=False)
             .drop(columns="strength")
             .reset_index(drop=True))
    print("\nWhat the model learned (kWh change per 1 standard deviation):")
    print(table.to_string(index=False))
    return table


def load():
    """Load the saved bundle, or explain how to create it."""
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError("Model not trained yet. Run main.py first.")
    return joblib.load(config.MODEL_FILE)


def forecast_one(pipe, row):
    """Expected total_kwh for a single incoming reading."""
    frame = row.to_frame().T[config.FEATURES].astype(float)
    return float(pipe.predict(frame)[0])


def classify_deviation(actual, expected, sigma):
    """
    Compare what happened against what was expected, in sigma units.

    Using sigma rather than a fixed kWh threshold means the detector
    automatically becomes stricter when the model is accurate and more
    forgiving when it is not.
    """
    error = actual - expected
    z = error / sigma if sigma else 0.0
    if abs(z) >= config.ALERT_SIGMA:
        status = "ALERT"
    elif abs(z) >= config.WATCH_SIGMA:
        status = "WATCH"
    else:
        status = "NORMAL"
    pct = (error / expected * 100) if expected else 0.0
    return {
        "expected_kwh": round(expected, 2),
        "deviation_kwh": round(error, 2),
        "deviation_pct": round(pct, 1),
        "sigma_score": round(z, 2),
        "status": status,
    }


if __name__ == "__main__":
    import data_loader
    import tariff_engine

    frame = tariff_engine.price_dataframe(data_loader.get_baseline())
    model, _ = train(frame)
    coefficients(model)
