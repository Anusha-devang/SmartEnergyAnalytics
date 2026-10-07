"""
eda.py
------
Stage 3: describe the day in numbers, then in four pictures.

Uses matplotlib only, with the Agg backend so it runs on a server, in Docker
and in Colab without trying to open a window.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

COLOURS = {
    "hvac_kwh": "#2b6cb0",
    "lighting_kwh": "#d69e2e",
    "machinery_kwh": "#c53030",
    "plug_load_kwh": "#2f855a",
}
SLAB_COLOUR = {"OFF-PEAK": "#2f855a", "NORMAL": "#d69e2e", "PEAK": "#c53030"}


def print_summary(df):
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    total = df["total_kwh"].sum()
    peak_row = df.loc[df["total_kwh"].idxmax()]
    base_row = df.loc[df["total_kwh"].idxmin()]

    print(f"Readings              : {len(df)} hourly")
    print(f"Total consumption     : {total:.1f} kWh")
    print(f"Average per hour      : {df['total_kwh'].mean():.2f} kWh")
    print(f"Peak hour             : {peak_row['hour']:02d}:00 at "
          f"{peak_row['total_kwh']:.1f} kWh")
    print(f"Base load (min hour)  : {base_row['hour']:02d}:00 at "
          f"{base_row['total_kwh']:.1f} kWh")
    print(f"Load factor           : {df['total_kwh'].mean() / peak_row['total_kwh']:.2f}"
          f"   (average / peak; 1.0 would be perfectly flat)")
    print(f"Total cost            : Rs.{df['total_cost'].sum():,.2f}")
    print(f"Total CO2             : {df['co2_kg'].sum():.1f} kg")

    print("\n-- Consumption by sub-meter --")
    for col in config.LOAD_COLUMNS:
        kwh = df[col].sum()
        print(f"  {config.LOAD_LABELS[col]:<11} {kwh:7.1f} kWh   {kwh / total:6.1%}")

    print("\n-- Cost and energy by tariff slab --")
    by_slab = df.groupby("tariff_slab").agg(
        hours=("total_kwh", "size"),
        kwh=("total_kwh", "sum"),
        cost=("total_cost", "sum"),
    ).sort_values("cost", ascending=False)
    for slab, r in by_slab.iterrows():
        print(f"  {slab:<9} {int(r['hours']):2d} h   {r['kwh']:6.1f} kWh   "
              f"Rs.{r['cost']:8,.2f}   {r['cost'] / df['total_cost'].sum():5.1%} of bill")

    print("\n-- What drives the total (correlation) --")
    corr = df[["temperature_c", "occupancy", "is_working_hour",
               "total_kwh"]].corr(numeric_only=True)["total_kwh"].drop("total_kwh")
    for name, v in corr.sort_values(ascending=False).items():
        print(f"  {name:<16} {v:+.3f}")


def _save(fig, name):
    path = config.OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[eda] saved {path.name}")


def build_charts(df):
    hours = df["hour"]

    # 1. Stacked load curve -- the single most important energy chart --------
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bottom = 0
    for col in config.LOAD_COLUMNS:
        ax.fill_between(hours, bottom, bottom + df[col], label=config.LOAD_LABELS[col],
                        color=COLOURS[col], alpha=0.85, linewidth=0)
        bottom = bottom + df[col]
    ax.axhline(config.CONTRACTED_DEMAND_KWH, color="black", ls="--", lw=1.1,
               label=f"Contracted demand ({config.CONTRACTED_DEMAND_KWH:.0f} kWh)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("kWh")
    ax.set_title("Daily load curve by sub-meter")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "01_load_curve.png")

    # 2. Cost per hour, coloured by tariff slab ------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(hours, df["total_cost"],
           color=[SLAB_COLOUR[s] for s in df["tariff_slab"]])
    for slab, colour in SLAB_COLOUR.items():
        ax.bar(0, 0, color=colour, label=slab)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Rupees")
    ax.set_title("Hourly cost, coloured by time-of-day tariff slab")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=8)
    _save(fig, "02_cost_by_slab.png")

    # 3. Temperature vs HVAC -------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 3.6))
    sc = ax.scatter(df["temperature_c"], df["hvac_kwh"], c=df["occupancy"],
                    cmap="viridis", s=55, edgecolors="white")
    fig.colorbar(sc, ax=ax, label="Occupancy")
    ax.set_xlabel("Outside temperature (C)")
    ax.set_ylabel("HVAC consumption (kWh)")
    ax.set_title("HVAC load vs temperature")
    _save(fig, "03_hvac_vs_temperature.png")

    # 4. Share of consumption per sub-meter ----------------------------------
    fig, ax = plt.subplots(figsize=(5, 3.6))
    totals = [df[c].sum() for c in config.LOAD_COLUMNS]
    labels = [config.LOAD_LABELS[c] for c in config.LOAD_COLUMNS]
    ax.barh(labels, totals, color=[COLOURS[c] for c in config.LOAD_COLUMNS])
    for i, v in enumerate(totals):
        ax.text(v + 0.6, i, f"{v:.1f} kWh ({v / sum(totals):.0%})",
                va="center", fontsize=8.5)
    ax.set_xlim(0, max(totals) * 1.35)
    ax.set_xlabel("kWh over the day")
    ax.set_title("Where the energy goes")
    _save(fig, "04_consumption_share.png")


def run(df):
    print_summary(df)
    build_charts(df)


if __name__ == "__main__":
    import data_loader
    import tariff_engine

    run(tariff_engine.price_dataframe(data_loader.get_baseline()))
