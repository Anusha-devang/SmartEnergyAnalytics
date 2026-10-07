"""
config.py
---------
One place for every path, tariff rate, threshold and column list used by the
project. If a number could be argued about by an energy manager, it lives here
and nowhere else.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# 1. FOLDER LAYOUT
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

BASELINE_CSV = DATA_DIR / "energy_readings.csv"
STREAM_CSV = DATA_DIR / "live_stream.csv"

MODEL_FILE = OUTPUT_DIR / "load_forecast_model.joblib"
ALERT_LOG = OUTPUT_DIR / "realtime_alerts.csv"
SUMMARY_REPORT = OUTPUT_DIR / "summary_report.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 2. COLUMN GROUPS
# ----------------------------------------------------------------------------
# The four sub-meters installed in the building
LOAD_COLUMNS = ["hvac_kwh", "lighting_kwh", "machinery_kwh", "plug_load_kwh"]

# Friendly labels for charts and console output
LOAD_LABELS = {
    "hvac_kwh": "HVAC",
    "lighting_kwh": "Lighting",
    "machinery_kwh": "Machinery",
    "plug_load_kwh": "Plug load",
}

TARGET = "total_kwh"

# Inputs given to the forecasting model
FEATURES = [
    "hour_sin",          # cyclical encoding of the clock
    "hour_cos",
    "temperature_c",
    "occupancy",
    "is_working_hour",
]

# ----------------------------------------------------------------------------
# 3. TARIFF: TIME-OF-DAY SLABS (Indian TOD structure, rupees per kWh)
# ----------------------------------------------------------------------------
# Each entry: (slab name, set of hours it covers, rate per kWh)
TARIFF_SLABS = [
    ("PEAK",     {9, 10, 11, 18, 19, 20},                          10.80),
    ("NORMAL",   {6, 7, 8, 12, 13, 14, 15, 16, 17, 21},             7.50),
    ("OFF-PEAK", {22, 23, 0, 1, 2, 3, 4, 5},                        5.20),
]
DEFAULT_RATE = 7.50

FIXED_CHARGE_PER_HOUR = 12.0        # standing charge billed every hour
CO2_KG_PER_KWH = 0.82               # Indian grid emission factor

# ----------------------------------------------------------------------------
# 4. OPERATING RULES
# ----------------------------------------------------------------------------
CONTRACTED_DEMAND_KWH = 25.0        # exceeding this in one hour draws a penalty
DEMAND_PENALTY_PER_KWH = 45.0       # rupees for every kWh above the contract

WORKING_HOURS = range(8, 19)        # 08:00 to 18:00 inclusive of 18

# Rule-engine thresholds (used by tariff_engine.check_rules)
IDLE_OCCUPANCY = 2                  # at or below this, the building is "empty"
HVAC_IDLE_LIMIT_KWH = 3.0           # HVAC above this with nobody in = fault
MACHINERY_IDLE_LIMIT_KWH = 1.5      # machinery running out of hours
LIGHTING_IDLE_LIMIT_KWH = 1.5       # lights left on

# ----------------------------------------------------------------------------
# 5. MODEL AND ANOMALY SETTINGS
# ----------------------------------------------------------------------------
CV_FOLDS = 5
RANDOM_STATE = 42

# Deviation measured in standard deviations of the model's own error
WATCH_SIGMA = 2.0                   # 2 sigma -> keep an eye on it
ALERT_SIGMA = 3.0                   # 3 sigma -> raise an alert

STREAM_DELAY_SECONDS = 1.0          # pause between two simulated readings
