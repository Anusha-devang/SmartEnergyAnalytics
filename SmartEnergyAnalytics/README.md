# Smart Energy Analytics Dashboard

A real-time data analytics project. One day of hourly smart-meter readings is
analysed to learn what "normal" looks like, then new readings are scored one by
one as they arrive: priced against a time-of-day tariff, checked against
operating rules, compared with a load forecast, and turned into alerts and
recommended actions.

Read `Smart_Energy_Analytics_Dashboard.pdf` first. It contains the full
requirement specification, the architecture, and a line-by-line explanation of
every file in `src/`.

---

## 1. Quick start (3 commands)

```bash
pip install -r requirements.txt
cd src
python main.py
```

That is the whole project. It runs all five stages and writes every artefact
into `outputs/`.

| Command | What it does |
|---|---|
| `python main.py` | Full run, 1 second pause between meter readings |
| `python main.py --fast` | Same, no pause (good while debugging) |
| `python main.py --no-charts` | Console output only, skips matplotlib |

Each module also runs standalone:

```bash
python data_loader.py       # ingestion + time-series features
python tariff_engine.py     # costing and operating rules
python eda.py               # regenerate the four charts
python model.py             # retrain the forecast model
python stream_simulator.py  # replay the live meter stream
```

---

## 2. Folder structure

```
SmartEnergyAnalytics/
├── README.md
├── requirements.txt
├── Smart_Energy_Analytics_Dashboard.pdf   <- the handbook, start here
├── data/
│   ├── energy_readings.csv    24 hourly readings = one baseline day
│   └── live_stream.csv        12 incoming readings for the live feed
├── src/
│   ├── config.py              paths, tariff slabs, thresholds
│   ├── data_loader.py         ingestion, validation, time features
│   ├── tariff_engine.py       cost, carbon and explainable rules
│   ├── eda.py                 console statistics + 4 charts
│   ├── model.py               load forecast + anomaly band
│   ├── stream_simulator.py    real-time loop with rolling KPIs
│   └── main.py                orchestrator (run this)
├── app/
│   └── dashboard.py           OPTIONAL Streamlit control room
├── sample_outputs/            reference results so you can check your run
└── outputs/                   empty until you run the project
```

---

## 3. What gets generated in `outputs/`

| File | Contents |
|---|---|
| `01_load_curve.png` | Stacked daily load curve by sub-meter |
| `02_cost_by_slab.png` | Hourly cost coloured by tariff slab |
| `03_hvac_vs_temperature.png` | HVAC load against outside temperature |
| `04_consumption_share.png` | Share of the day's energy per sub-meter |
| `load_forecast_model.joblib` | Fitted pipeline plus its error band (sigma) |
| `realtime_alerts.csv` | One audit row per streamed reading |
| `summary_report.txt` | Plain-text run summary |

`sample_outputs/` holds the exact files a correct run produces, plus
`console_output.txt` with the full reference log. Compare your `outputs/`
against it — only the timestamps should differ.

---

## 4. Optional dashboard

```bash
pip install streamlit
python src/main.py          # must run once so the model file exists
streamlit run app/dashboard.py
```

Three tabs: a live meter stream with animated KPI tiles and an
actual-vs-expected chart, a baseline-day explorer, and a what-if bill
calculator.

---

## 5. Requirements

Python 3.9 or newer. Dependencies are in `requirements.txt`
(pandas, numpy, scikit-learn, matplotlib, joblib). No internet access,
no database and no API key is required.
