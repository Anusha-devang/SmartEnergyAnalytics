"""
tariff_engine.py
----------------
Stage 2: turn kilowatt-hours into money, carbon and plain-English warnings.

This module contains no machine learning at all. It is deterministic
arithmetic that an electricity bill auditor could check by hand, which is
exactly why it sits alongside the model rather than inside it.
"""

import config


def slab_for_hour(hour):
    """Return (slab name, rupees per kWh) for a given clock hour."""
    for name, hours, rate in config.TARIFF_SLABS:
        if hour in hours:
            return name, rate
    return "NORMAL", config.DEFAULT_RATE


def cost_of_reading(row):
    """
    Price one hourly reading.

    Bill for the hour = energy charge + fixed charge + demand penalty
      energy charge  = kWh consumed x the slab rate for that hour
      fixed charge   = a flat standing charge
      demand penalty = charged only on the kWh above the contracted demand
    """
    slab, rate = slab_for_hour(int(row["hour"]))
    kwh = float(row["total_kwh"])

    energy_cost = kwh * rate
    excess = max(0.0, kwh - config.CONTRACTED_DEMAND_KWH)
    penalty = excess * config.DEMAND_PENALTY_PER_KWH

    return {
        "tariff_slab": slab,
        "tariff_rate": rate,
        "energy_cost": round(energy_cost, 2),
        "demand_excess_kwh": round(excess, 2),
        "demand_penalty": round(penalty, 2),
        "total_cost": round(energy_cost + config.FIXED_CHARGE_PER_HOUR + penalty, 2),
        "co2_kg": round(kwh * config.CO2_KG_PER_KWH, 2),
    }


def check_rules(row):
    """
    Explainable operating rules. Each one describes a specific, fixable waste
    condition, so the message can be sent straight to a facilities engineer.
    """
    findings = []
    empty = row["occupancy"] <= config.IDLE_OCCUPANCY

    if empty and row["hvac_kwh"] > config.HVAC_IDLE_LIMIT_KWH:
        findings.append(
            f"HVAC drawing {row['hvac_kwh']:.1f} kWh with {int(row['occupancy'])} "
            f"people present - suspected stuck damper or thermostat")

    if empty and row["machinery_kwh"] > config.MACHINERY_IDLE_LIMIT_KWH:
        findings.append(
            f"Machinery at {row['machinery_kwh']:.1f} kWh outside occupied hours "
            f"- equipment not shut down")

    if empty and row["lighting_kwh"] > config.LIGHTING_IDLE_LIMIT_KWH:
        findings.append(
            f"Lighting at {row['lighting_kwh']:.1f} kWh in an empty building "
            f"- lights left on")

    if row["total_kwh"] > config.CONTRACTED_DEMAND_KWH:
        findings.append(
            f"Demand {row['total_kwh']:.1f} kWh exceeds the contracted "
            f"{config.CONTRACTED_DEMAND_KWH:.0f} kWh - penalty applies")

    slab, _ = slab_for_hour(int(row["hour"]))
    if slab == "PEAK" and row["machinery_kwh"] > 12.0:
        findings.append(
            "Heavy machinery running inside the peak tariff window - "
            "consider shifting this batch to off-peak")

    return findings


def recommend(findings, deviation_pct):
    """Turn findings into the single most useful next action."""
    if not findings:
        if deviation_pct is not None and deviation_pct < -15:
            return "Consumption well below forecast - verify the meter is reporting"
        return "No action required"
    text = " ".join(findings).lower()
    if "hvac" in text:
        return "Dispatch technician to inspect the HVAC unit now"
    if "machinery" in text and "peak" in text:
        return "Reschedule the machinery batch to the off-peak window after 22:00"
    if "machinery" in text:
        return "Shut down idle machinery from the building management system"
    if "lighting" in text:
        return "Switch off unoccupied-zone lighting"
    if "demand" in text:
        return "Shed non-critical load to stay under the contracted demand"
    return "Review the flagged meter"


def price_dataframe(df):
    """Add all costing columns to a whole DataFrame at once."""
    priced = df.apply(cost_of_reading, axis=1, result_type="expand")
    out = df.join(priced)
    print(f"[tariff_engine] Priced {len(out)} readings -> "
          f"Rs.{out['total_cost'].sum():,.2f} total, "
          f"{out['co2_kg'].sum():.1f} kg CO2")
    return out


if __name__ == "__main__":
    import data_loader

    demo = price_dataframe(data_loader.get_baseline())
    print(demo[["timestamp", "total_kwh", "tariff_slab", "tariff_rate",
                "total_cost", "co2_kg"]].head(8).to_string(index=False))
