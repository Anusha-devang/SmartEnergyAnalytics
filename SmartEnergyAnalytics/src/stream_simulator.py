"""
stream_simulator.py
-------------------
Stage 5: the real-time part.

A real building pushes a meter reading every few minutes over MQTT or Modbus.
We reproduce that by reading live_stream.csv one row at a time with a delay.
For every reading that arrives we:

    1. engineer exactly the same features used at training time
    2. price it through the tariff engine        -> money and carbon
    3. run the explainable operating rules       -> named waste conditions
    4. ask the model what it expected            -> deviation in sigma units
    5. combine rules and model into one status   -> NORMAL / WATCH / ALERT
    6. update rolling KPIs and append to the alert log

Only step 1's data source would change if this were fed by a real broker.
"""

import time
from datetime import datetime

import pandas as pd

import config
import data_loader
import model as model_module
import tariff_engine


def combine_status(model_status, findings):
    """
    Arbitration between the two layers.

    A named rule finding is hard evidence of a specific fault, so it can raise
    the severity on its own. The model can raise it too, but a purely
    statistical deviation with no rule behind it stays a WATCH until it is
    extreme, because the model is only as good as one day of history.
    """
    if findings and model_status == "ALERT":
        return "ALERT", "rule finding confirmed by the forecast model"
    if findings:
        return "ALERT", "operating rule breached"
    if model_status == "ALERT":
        return "WATCH", "large deviation from forecast, no rule matched"
    if model_status == "WATCH":
        return "WATCH", "mild deviation from forecast"
    return "NORMAL", "within the expected band"


class RollingKPI:
    """Running totals updated in O(1) per event, exactly like a stream job."""

    def __init__(self):
        self.readings = 0
        self.kwh = 0.0
        self.cost = 0.0
        self.co2 = 0.0
        self.penalty = 0.0
        self.alerts = 0
        self.watches = 0
        self.peak_kwh = 0.0
        self.peak_hour = None
        self.wasted_kwh = 0.0        # kWh above forecast on flagged hours

    def update(self, row, cost, status, deviation):
        self.readings += 1
        self.kwh += row["total_kwh"]
        self.cost += cost["total_cost"]
        self.co2 += cost["co2_kg"]
        self.penalty += cost["demand_penalty"]
        if status == "ALERT":
            self.alerts += 1
            if deviation["deviation_kwh"] > 0:
                self.wasted_kwh += deviation["deviation_kwh"]
        elif status == "WATCH":
            self.watches += 1
        if row["total_kwh"] > self.peak_kwh:
            self.peak_kwh = row["total_kwh"]
            self.peak_hour = int(row["hour"])

    def snapshot(self):
        return {
            "readings": self.readings,
            "kwh": round(self.kwh, 2),
            "cost": round(self.cost, 2),
            "co2": round(self.co2, 2),
            "penalty": round(self.penalty, 2),
            "alerts": self.alerts,
            "watches": self.watches,
            "peak_kwh": round(self.peak_kwh, 2),
            "peak_hour": self.peak_hour,
            "wasted_kwh": round(self.wasted_kwh, 2),
            "avg_rate": round(self.cost / self.kwh, 2) if self.kwh else 0.0,
        }


def run(bundle=None, delay=None):
    """Consume the whole stream and return (alert log, final KPI snapshot)."""
    bundle = bundle or model_module.load()
    pipe, sigma = bundle["pipeline"], bundle["sigma"]
    delay = config.STREAM_DELAY_SECONDS if delay is None else delay

    stream = data_loader.get_stream()
    kpi = RollingKPI()
    log = []

    print("\n" + "=" * 70)
    print("REAL-TIME METER STREAM")
    print("=" * 70)
    print(f"Model error band: 1 sigma = {sigma:.2f} kWh | "
          f"WATCH at {config.WATCH_SIGMA:.0f} sigma, ALERT at {config.ALERT_SIGMA:.0f} sigma")

    for _, row in stream.iterrows():
        time.sleep(delay)
        received = datetime.now().strftime("%H:%M:%S")

        cost = tariff_engine.cost_of_reading(row)                  # money
        findings = tariff_engine.check_rules(row)                  # rules
        expected = model_module.forecast_one(pipe, row)            # model
        deviation = model_module.classify_deviation(
            row["total_kwh"], expected, sigma)
        status, why = combine_status(deviation["status"], findings)
        action = tariff_engine.recommend(findings, deviation["deviation_pct"])

        kpi.update(row, cost, status, deviation)
        snap = kpi.snapshot()

        mark = {"NORMAL": "  ok  ", "WATCH": " WATCH", "ALERT": " ALERT"}[status]
        print(f"\n[{received}] {row['timestamp']:%Y-%m-%d %H:%M}  [{mark}]")
        print(f"    actual {row['total_kwh']:5.1f} kWh | expected "
              f"{deviation['expected_kwh']:5.1f} kWh | deviation "
              f"{deviation['deviation_kwh']:+5.1f} kWh "
              f"({deviation['deviation_pct']:+.0f}%, {deviation['sigma_score']:+.1f} sigma)")
        print(f"    HVAC {row['hvac_kwh']:.1f} | Light {row['lighting_kwh']:.1f} | "
              f"Mach {row['machinery_kwh']:.1f} | Plug {row['plug_load_kwh']:.1f} | "
              f"{row['temperature_c']:.1f} C | {int(row['occupancy'])} people")
        print(f"    {cost['tariff_slab']:<8} @ Rs.{cost['tariff_rate']:.2f}/kWh -> "
              f"Rs.{cost['total_cost']:.2f}"
              + (f"  (incl. Rs.{cost['demand_penalty']:.2f} demand penalty)"
                 if cost["demand_penalty"] else ""))
        for f in findings:
            print(f"    ! {f}")
        print(f"    verdict : {status} - {why}")
        print(f"    action  : {action}")
        print(f"    LIVE KPI: {snap['readings']} readings | {snap['kwh']:.1f} kWh | "
              f"Rs.{snap['cost']:,.2f} | {snap['co2']:.1f} kg CO2 | "
              f"peak {snap['peak_kwh']:.1f} kWh at {snap['peak_hour']:02d}:00 | "
              f"{snap['alerts']} alerts")

        log.append({
            "received_at": received,
            "timestamp": row["timestamp"],
            "total_kwh": row["total_kwh"],
            "expected_kwh": deviation["expected_kwh"],
            "deviation_kwh": deviation["deviation_kwh"],
            "deviation_pct": deviation["deviation_pct"],
            "sigma_score": deviation["sigma_score"],
            "tariff_slab": cost["tariff_slab"],
            "total_cost": cost["total_cost"],
            "demand_penalty": cost["demand_penalty"],
            "co2_kg": cost["co2_kg"],
            "status": status,
            "reason": why,
            "findings": " | ".join(findings) if findings else "",
            "action": action,
        })

    log_df = pd.DataFrame(log)
    log_df.to_csv(config.ALERT_LOG, index=False)
    print(f"\n[stream] alert log written to {config.ALERT_LOG.name}")

    final = kpi.snapshot()
    wasted_cost = final["wasted_kwh"] * final["avg_rate"]
    print("\n--- SESSION SUMMARY ---")
    print(f"Readings processed   : {final['readings']}")
    print(f"Energy consumed      : {final['kwh']:.1f} kWh")
    print(f"Cost incurred        : Rs.{final['cost']:,.2f} "
          f"(effective Rs.{final['avg_rate']:.2f}/kWh)")
    print(f"Demand penalty       : Rs.{final['penalty']:,.2f}")
    print(f"Carbon emitted       : {final['co2']:.1f} kg CO2")
    print(f"Peak demand          : {final['peak_kwh']:.1f} kWh at "
          f"{final['peak_hour']:02d}:00")
    print(f"Alerts / watches     : {final['alerts']} / {final['watches']}")
    print(f"Avoidable waste      : {final['wasted_kwh']:.1f} kWh "
          f"= about Rs.{wasted_cost:,.2f} on flagged hours alone")

    final["wasted_cost"] = round(wasted_cost, 2)
    return log_df, final


if __name__ == "__main__":
    run()
