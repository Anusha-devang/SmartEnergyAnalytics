"""
main.py
-------
Entry point. Runs the five stages in order:

    ingest -> tariff costing -> EDA -> forecast model -> real-time stream

Usage:
    python main.py                # full run, 1 second between meter readings
    python main.py --fast         # no delay
    python main.py --no-charts    # console only, skip matplotlib
"""

import argparse
from datetime import datetime

import config
import data_loader
import eda
import model as model_module
import stream_simulator
import tariff_engine


def banner(step, title):
    print("\n" + "#" * 70)
    print(f"# STEP {step}: {title}")
    print("#" * 70)


def write_summary(priced, metrics, coefs, kpi):
    lines = [
        "SMART ENERGY ANALYTICS DASHBOARD - RUN SUMMARY",
        f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "1. BASELINE DAY",
        f"   Readings           : {len(priced)} hourly",
        f"   Consumption        : {priced['total_kwh'].sum():.1f} kWh",
        f"   Cost               : Rs.{priced['total_cost'].sum():,.2f}",
        f"   CO2                : {priced['co2_kg'].sum():.1f} kg",
        f"   Peak hour          : {int(priced.loc[priced['total_kwh'].idxmax(), 'hour']):02d}:00"
        f" at {priced['total_kwh'].max():.1f} kWh",
        f"   Load factor        : {priced['total_kwh'].mean() / priced['total_kwh'].max():.2f}",
        "",
        "2. FORECAST MODEL",
        f"   MAE / RMSE         : {metrics['mae']} / {metrics['rmse']} kWh",
        f"   MAPE               : {metrics['mape']} %",
        f"   R-squared          : {metrics['r2']}",
        f"   Residual sigma     : {metrics['sigma']} kWh",
        "",
        "   Coefficients (per 1 standard deviation of the input):",
    ]
    for _, r in coefs.iterrows():
        lines.append(f"     {r['feature']:<18} {r['coefficient']:>8}")

    lines += [
        "",
        "3. REAL-TIME STREAM",
        f"   Readings processed : {kpi['readings']}",
        f"   Energy             : {kpi['kwh']:.1f} kWh",
        f"   Cost               : Rs.{kpi['cost']:,.2f} (Rs.{kpi['avg_rate']:.2f}/kWh)",
        f"   Demand penalty     : Rs.{kpi['penalty']:,.2f}",
        f"   CO2                : {kpi['co2']:.1f} kg",
        f"   Peak               : {kpi['peak_kwh']:.1f} kWh at {kpi['peak_hour']:02d}:00",
        f"   Alerts / watches   : {kpi['alerts']} / {kpi['watches']}",
        f"   Avoidable waste    : {kpi['wasted_kwh']:.1f} kWh "
        f"(about Rs.{kpi['wasted_cost']:,.2f})",
        "",
        "Artefacts in outputs/: charts (*.png), load_forecast_model.joblib,",
        "realtime_alerts.csv, summary_report.txt",
    ]
    config.SUMMARY_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[main] summary written to {config.SUMMARY_REPORT.name}")


def main():
    parser = argparse.ArgumentParser(description="Smart Energy Analytics Dashboard")
    parser.add_argument("--fast", action="store_true", help="no delay in the stream")
    parser.add_argument("--no-charts", action="store_true", help="skip PNG generation")
    args = parser.parse_args()

    print("\nSMART ENERGY ANALYTICS DASHBOARD")
    print(f"Started at {datetime.now():%Y-%m-%d %H:%M:%S}")

    banner(1, "DATA INGESTION & FEATURE ENGINEERING")
    baseline = data_loader.get_baseline()

    banner(2, "TARIFF COSTING & OPERATING RULES")
    priced = tariff_engine.price_dataframe(baseline)
    print(priced[["timestamp", "total_kwh", "tariff_slab", "tariff_rate",
                  "energy_cost", "total_cost", "co2_kg"]].to_string(index=False))

    banner(3, "EXPLORATORY DATA ANALYSIS")
    if args.no_charts:
        eda.print_summary(priced)
    else:
        eda.run(priced)

    banner(4, "LOAD FORECAST MODEL")
    pipe, metrics = model_module.train(priced)
    coefs = model_module.coefficients(pipe)

    banner(5, "REAL-TIME METER STREAM")
    bundle = {"pipeline": pipe, "sigma": metrics["sigma"], "metrics": metrics}
    _, kpi = stream_simulator.run(bundle, delay=0 if args.fast else None)

    write_summary(priced, metrics, coefs, kpi)
    print("\nPipeline finished successfully.\n")


if __name__ == "__main__":
    main()
