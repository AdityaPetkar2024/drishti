# Drishti: Parking Enforcement Intelligence for Bengaluru


**🔗 Live demo: https://drishti-xxxxx.streamlit.app**
**Gridlock Hackathon 2.0 · Theme 1: Poor Visibility on Parking-Induced Congestion**

Drishti turns 248,376 confirmed parking-violation records into a ranked, time-aware
enforcement plan, plus an AI model that forecasts where violations will concentrate next.
Instead of reactive, patrol-based enforcement spread evenly across the city, it tells
Bengaluru Traffic Police **where** illegal parking concentrates, **when** it peaks, **how
many officers** to deploy, and **where to pre-position them** ahead of time.

Live data partner: Bengaluru Traffic Police (BTP). Host: Flipkart.

## The core insight

- The **top 50 hotspot zones**, under 1% of all locations, produce **33.6%** of every
  violation in the city. The top 500 zones reach **77%**.
- Violations peak at **10:00-11:00 IST** and on **Sundays**.
- **49.7%** of violations happen away from monitored junctions: mid-road and street
  parking that junction-based enforcement misses entirely.

Concentrating enforcement on these zones and windows is the single highest-leverage change.

## What it does (5 views)

1. **Command Overview**: headline metrics and the concentration + timing story.
2. **Hotspot Map**: the 500 most active parking zones citywide, hoverable, priority-coloured,
   bounds-locked to Bengaluru.
3. **When Violations Happen**: hour, day, violation-type and vehicle-type patterns (IST).
4. **Deployment Plan**: a ranked schedule with peak windows, target vehicle types, police
   station, and a recommended officer count per zone. Includes a coverage curve and CSV export.
5. **AI Forecast**: a LightGBM model predicting violation intensity per zone/hour/day,
   validated on a held-out future month.

## The AI model

A LightGBM gradient-boosting regressor forecasts violation intensity for every zone, hour,
and day of week. Validation uses a strict **temporal hold-out**: the final 30 days are never
seen during training, and every historical feature is computed from the training period only,
so there is no leakage.

- **Ranking AUC: 0.87** on the held-out future month.
- **Top 10% of predicted windows capture 56%** of all actual violations in that month.
- **MAE: 0.34** violations per zone-hour.

This is what shifts enforcement from reactive to proactive: deploy before violations occur.

## Methodology notes (why the numbers are trustworthy)

- **Data quality:** rejected and duplicate citations are removed (16.8%). Unvalidated
  (null-status) records are kept: they are time-clustered after Jan 2024, when BTP's review
  workflow tapered off, so they are unreviewed but genuine violations. Dropping them would
  delete three months of real recent data.
- **Timezone:** raw timestamps are UTC. Every timestamp is converted to **IST (UTC+5:30)**
  before any temporal analysis. Skipping this makes peak hours look like pre-dawn, which is wrong.
- **Coordinate validation:** all rows confirmed within Bengaluru bounds (12.7-13.3N, 77.3-77.9E).
- **Priority score:** a transparent blend of violation volume (60%, log-scaled),
  repeat-offender concentration (25%), and breadth of offending vehicles (15%). No black box.
- **Officer recommendation:** tiered by daily violation rate (25+/day = 3 officers,
  10+/day = 2, else 1).

## Run locally

```bash
pip install -r requirements.txt
python process_data.py            # builds hotspot/pattern artifacts in ./data/
python train_forecast_model.py    # trains the forecast model, writes predictions to ./data/
streamlit run app.py              # launches the dashboard
```

The raw file `jan to may police violation_anonymized791b166.csv` must be in the project root
to run the two scripts. The processed artifacts in `./data/` are committed, so the dashboard
runs immediately without re-processing.

## Stack

Python · pandas · NumPy · LightGBM · Streamlit · Plotly

## Project structure

```
app.py                    # Streamlit dashboard (5 views)
process_data.py           # data cleaning, hotspot detection, scoring -> ./data/
train_forecast_model.py   # LightGBM forecast model -> ./data/
data/                     # generated artifacts (committed for instant deploy)
requirements.txt
```
