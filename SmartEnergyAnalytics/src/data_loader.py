"""
data_loader.py
--------------
Stage 1: read the meter file, prove it is usable, and derive the columns the
tariff engine and the forecast model need.

Time-series data brings two jobs that ordinary tabular data does not:
parsing the timestamp properly, and turning the clock into something a model
can actually learn from.
"""

import numpy as np
import pandas as pd

import config


def load_csv(path):
    """Read a meter CSV and fail loudly if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"[data_loader] Loaded {len(df)} readings from {path.name} "
          f"({df['timestamp'].min()} to {df['timestamp'].max()})")
    return df


def validate(df):
    """Data-quality gate. Returns the cleaned frame plus a small report."""
    report = {
        "readings": len(df),
        "duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "negative_readings": int((df[config.LOAD_COLUMNS] < 0).sum().sum()),
    }

    # A meter can never run backwards; a negative value is a sensor fault.
    for col in config.LOAD_COLUMNS:
        df.loc[df[col] < 0, col] = np.nan

    # Duplicate timestamps mean the meter reported twice: keep the first.
    if report["duplicate_timestamps"]:
        df = df.drop_duplicates(subset="timestamp", keep="first")

    # A dropped reading is filled by interpolating between its neighbours,
    # which suits time series far better than a column median.
    numeric = config.LOAD_COLUMNS + ["temperature_c", "occupancy"]
    before = int(df[numeric].isna().sum().sum())
    if before:
        df[numeric] = df[numeric].interpolate(limit_direction="both")
    report["interpolated"] = before

    # Readings must be in chronological order for any time-based logic.
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Sampling interval check: are the readings really hourly?
    gaps = df["timestamp"].diff().dropna().dt.total_seconds() / 3600
    report["interval_hours"] = float(gaps.mode().iloc[0]) if len(gaps) else 0.0
    report["gaps_in_series"] = int((gaps != report["interval_hours"]).sum())

    print(f"[data_loader] Validation -> {report}")
    return df, report


def engineer_features(df):
    """Add the derived columns used everywhere downstream."""
    df = df.copy()

    # Total draw of the building for that hour: the headline number.
    df["total_kwh"] = df[config.LOAD_COLUMNS].sum(axis=1).round(2)

    # Calendar parts pulled out of the timestamp.
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.date
    df["is_working_hour"] = df["hour"].isin(config.WORKING_HOURS).astype(int)

    # Cyclical encoding. Hour 23 and hour 0 are one hour apart, but as plain
    # numbers they look 23 apart. Mapping the clock onto a circle with sine and
    # cosine tells the model that midnight follows 23:00.
    angle = 2 * np.pi * df["hour"] / 24
    df["hour_sin"] = np.sin(angle).round(4)
    df["hour_cos"] = np.cos(angle).round(4)

    # Share of the total taken by each sub-meter, useful for the charts.
    for col in config.LOAD_COLUMNS:
        df[col.replace("_kwh", "_share")] = (df[col] / df["total_kwh"]).round(3)

    print(f"[data_loader] Engineered features; total load "
          f"{df['total_kwh'].sum():.1f} kWh over {len(df)} hours")
    return df


def get_baseline():
    """Full pipeline for the historical day used to learn normal behaviour."""
    df, _ = validate(load_csv(config.BASELINE_CSV))
    return engineer_features(df)


def get_stream():
    """Same treatment for the live feed, so the model sees identical inputs."""
    df, _ = validate(load_csv(config.STREAM_CSV))
    return engineer_features(df)


if __name__ == "__main__":
    data = get_baseline()
    print(data[["timestamp", "total_kwh", "hour_sin", "hour_cos",
                "is_working_hour"]].head(8).to_string(index=False))
