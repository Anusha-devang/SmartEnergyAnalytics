"""
dashboard.py  (OPTIONAL BONUS MODULE)
-------------------------------------
A live control-room view built on the same src/ modules. Run it only after
`python src/main.py` has produced outputs/load_forecast_model.joblib.

    pip install streamlit
    streamlit run app/dashboard.py

No analytics logic is duplicated here - every number comes from the same
tariff engine and the same forecast model used by the console pipeline.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import config                     # noqa: E402
import data_loader                # noqa: E402
import model as model_module      # noqa: E402
import tariff_engine              # noqa: E402
from stream_simulator import RollingKPI, combine_status   # noqa: E402

st.set_page_config(page_title="Smart Energy Control Room", layout="wide")
st.title("Smart Energy Analytics Dashboard")

bundle = model_module.load()
pipe, sigma = bundle["pipeline"], bundle["sigma"]

tab_live, tab_day, tab_what = st.tabs(
    ["Live meter stream", "Baseline day", "What-if calculator"])

# ------------------------------------------------------------- live stream --
with tab_live:
    delay = st.slider("Seconds between meter readings", 0.0, 3.0, 1.0, 0.5)
    if st.button("Start stream", type="primary"):
        stream = data_loader.get_stream()
        kpi = RollingKPI()
        c1, c2, c3, c4, c5 = st.columns(5)
        m = [c.empty() for c in (c1, c2, c3, c4, c5)]
        chart_slot = st.empty()
        alert_slot = st.container()
        table_slot = st.empty()
        rows, series = [], []

        for _, row in stream.iterrows():
            time.sleep(delay)
            cost = tariff_engine.cost_of_reading(row)
            findings = tariff_engine.check_rules(row)
            expected = model_module.forecast_one(pipe, row)
            dev = model_module.classify_deviation(row["total_kwh"], expected, sigma)
            status, why = combine_status(dev["status"], findings)
            action = tariff_engine.recommend(findings, dev["deviation_pct"])
            kpi.update(row, cost, status, dev)
            snap = kpi.snapshot()

            m[0].metric("Readings", snap["readings"])
            m[1].metric("Energy", f"{snap['kwh']:.1f} kWh")
            m[2].metric("Cost", f"Rs.{snap['cost']:,.0f}")
            m[3].metric("CO2", f"{snap['co2']:.0f} kg")
            m[4].metric("Alerts", snap["alerts"])

            series.append({"hour": f"{int(row['hour']):02d}:00",
                           "Actual kWh": row["total_kwh"],
                           "Expected kWh": dev["expected_kwh"]})
            chart_slot.line_chart(pd.DataFrame(series).set_index("hour"))

            if status == "ALERT":
                alert_slot.error(f"{int(row['hour']):02d}:00 - {action}\n\n"
                                 + "\n\n".join(findings))

            rows.append({
                "Hour": f"{int(row['hour']):02d}:00",
                "Actual": row["total_kwh"], "Expected": dev["expected_kwh"],
                "Dev %": dev["deviation_pct"], "Sigma": dev["sigma_score"],
                "Slab": cost["tariff_slab"], "Cost": cost["total_cost"],
                "Status": status, "Action": action,
            })
            table_slot.dataframe(pd.DataFrame(rows), width="stretch")

# ------------------------------------------------------------ baseline day --
with tab_day:
    priced = tariff_engine.price_dataframe(data_loader.get_baseline())
    a, b, c, d = st.columns(4)
    a.metric("Consumption", f"{priced['total_kwh'].sum():.1f} kWh")
    b.metric("Cost", f"Rs.{priced['total_cost'].sum():,.0f}")
    c.metric("CO2", f"{priced['co2_kg'].sum():.0f} kg")
    d.metric("Peak", f"{priced['total_kwh'].max():.1f} kWh")
    st.subheader("Load curve by sub-meter")
    st.area_chart(priced.set_index("hour")[config.LOAD_COLUMNS])
    st.subheader("Cost per hour")
    st.bar_chart(priced.set_index("hour")["total_cost"])
    st.dataframe(priced[["timestamp", "total_kwh", "tariff_slab", "tariff_rate",
                         "total_cost", "co2_kg"]], width="stretch")

# ------------------------------------------------------- what-if calculator --
with tab_what:
    st.write("Estimate the bill for a single hour under any conditions.")
    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Hour of day", 0, 23, 14)
        hvac = st.number_input("HVAC kWh", 0.0, 40.0, 8.0, 0.1)
        light = st.number_input("Lighting kWh", 0.0, 20.0, 3.5, 0.1)
    with col2:
        mach = st.number_input("Machinery kWh", 0.0, 40.0, 9.5, 0.1)
        plug = st.number_input("Plug load kWh", 0.0, 20.0, 3.8, 0.1)
        occ = st.slider("Occupancy", 0, 120, 65)

    total = hvac + light + mach + plug
    row = pd.Series({"hour": hour, "total_kwh": round(total, 2),
                     "hvac_kwh": hvac, "lighting_kwh": light,
                     "machinery_kwh": mach, "plug_load_kwh": plug,
                     "occupancy": occ})
    cost = tariff_engine.cost_of_reading(row)
    findings = tariff_engine.check_rules(row)

    x, y, z = st.columns(3)
    x.metric("Total demand", f"{total:.1f} kWh")
    y.metric(f"{cost['tariff_slab']} rate", f"Rs.{cost['tariff_rate']:.2f}/kWh")
    z.metric("Hour bill", f"Rs.{cost['total_cost']:,.2f}")
    if cost["demand_penalty"]:
        st.warning(f"Demand penalty of Rs.{cost['demand_penalty']:,.2f} on "
                   f"{cost['demand_excess_kwh']:.1f} kWh above contract")
    for f in findings:
        st.info(f)
    st.caption(f"Carbon for this hour: {cost['co2_kg']:.2f} kg CO2")
