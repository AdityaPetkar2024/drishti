"""
Drishti — Data Processing & Hotspot Intelligence Pipeline
Generates all artifacts the dashboard needs from the raw violations CSV.

Run once:  python process_data.py
Outputs (saved to ./data/):
    hotspots.csv          — ranked enforcement hotspots with scores
    hourly_pattern.csv    — violations by hour (IST)
    daily_pattern.csv     — violations by day of week
    vehicle_pattern.csv   — violations by vehicle type
    violation_types.csv   — exploded individual violation counts
    station_load.csv      — violations per police station
    heatmap_points.csv    — downsampled lat/lon for map rendering
    summary_stats.json    — headline numbers for the dashboard
"""

import pandas as pd
import numpy as np
import ast
import json
import os

RAW_CSV = "jan to may police violation_anonymized791b166.csv"
OUT_DIR = "data"
GRID = 0.0015          # ~165m grid cells for hotspot aggregation
TOP_N_HOTSPOTS = 50
TOP_N_MAP = 500        # broad citywide coverage on the map

os.makedirs(OUT_DIR, exist_ok=True)


def load_and_clean(path):
    df = pd.read_csv(path)

    # Remove confirmed-invalid citations only.
    # 'rejected' and 'duplicate' are dismissed/erroneous tickets — not real road blockages.
    # We deliberately KEEP null validation_status: those are time-clustered (BTP's review
    # workflow tapered off after Jan 2024), so they are unreviewed but complete, genuine
    # violations. Dropping them would delete ~3 months of real recent data and bias the
    # analysis toward the past.
    before = len(df)
    df = df[~df["validation_status"].isin(["rejected", "duplicate"])].copy()
    print(f"  Removed {before - len(df):,} rejected/duplicate citations "
          f"({100*(before-len(df))/before:.1f}%); kept {len(df):,} confirmed violations")

    # CRITICAL: timestamps are UTC. Bengaluru is UTC+5:30.
    # Without this conversion, peak hours are completely wrong.
    df["created_dt"] = (
        pd.to_datetime(df["created_datetime"], format="ISO8601")
        + pd.Timedelta(hours=5, minutes=30)
    )

    # Validate coordinates are within Bengaluru bounds
    lat_ok = df["latitude"].between(12.7, 13.3)
    lon_ok = df["longitude"].between(77.3, 77.9)
    df = df[lat_ok & lon_ok].copy()

    # Time features (all IST)
    df["hour"] = df["created_dt"].dt.hour
    df["dayofweek"] = df["created_dt"].dt.dayofweek
    df["date"] = df["created_dt"].dt.date

    # Parse the JSON-like violation_type field
    def parse_v(s):
        try:
            return ast.literal_eval(s)
        except Exception:
            return []
    df["violations_list"] = df["violation_type"].apply(parse_v)
    df["num_violations"] = df["violations_list"].apply(len)

    return df


def build_hotspots(df):
    """Grid-based hotspot detection + multi-factor enforcement priority score."""
    df["grid_lat"] = (df["latitude"] / GRID).round() * GRID
    df["grid_lon"] = (df["longitude"] / GRID).round() * GRID

    g = df.groupby(["grid_lat", "grid_lon"])

    hotspots = g.agg(
        violations=("id", "count"),
        unique_vehicles=("vehicle_number", "nunique"),
        peak_hour=("hour", lambda x: x.mode().iloc[0] if len(x.mode()) else -1),
        top_vehicle=("vehicle_type", lambda x: x.mode().iloc[0] if len(x.mode()) else "N/A"),
        top_location=("location", lambda x: x.mode().iloc[0] if len(x.mode()) else "N/A"),
        top_station=("police_station", lambda x: x.mode().iloc[0] if len(x.mode()) else "N/A"),
        top_junction=("junction_name", lambda x: x.mode().iloc[0] if len(x.mode()) else "N/A"),
    ).reset_index()

    # Repeat-offender ratio: how concentrated violations are among vehicles
    hotspots["repeat_ratio"] = (
        hotspots["violations"] / hotspots["unique_vehicles"]
    ).round(2)

    # --- Enforcement Priority Score (0-100) ---
    # Transparent, defensible weighting. Each component normalised 0-1.
    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    hotspots["s_volume"] = norm(np.log1p(hotspots["violations"]))      # log: avoid mega-cells dominating
    hotspots["s_repeat"] = norm(hotspots["repeat_ratio"])             # chronic-offender locations
    hotspots["s_spread"] = norm(hotspots["unique_vehicles"])          # breadth of the problem

    hotspots["priority_score"] = (
        0.60 * hotspots["s_volume"]
        + 0.25 * hotspots["s_repeat"]
        + 0.15 * hotspots["s_spread"]
    ) * 100
    hotspots["priority_score"] = hotspots["priority_score"].round(1)

    hotspots = hotspots.sort_values("priority_score", ascending=False).reset_index(drop=True)
    hotspots["rank"] = hotspots.index + 1

    # Recommended officers: simple, explainable tiers by daily violation rate
    n_days = df["date"].nunique()
    hotspots["violations_per_day"] = (hotspots["violations"] / n_days).round(1)

    def officers(vpd):
        if vpd >= 25:
            return 3
        if vpd >= 10:
            return 2
        return 1
    hotspots["recommended_officers"] = hotspots["violations_per_day"].apply(officers)

    return hotspots.head(TOP_N_HOTSPOTS), hotspots.head(TOP_N_MAP)


def main():
    print("Loading and cleaning...")
    df = load_and_clean(RAW_CSV)
    n_days = df["date"].nunique()
    print(f"  {len(df):,} valid violations over {n_days} days")

    print("Building hotspots...")
    hotspots, hotspots_map = build_hotspots(df)
    hotspots.to_csv(f"{OUT_DIR}/hotspots.csv", index=False)
    hotspots_map.to_csv(f"{OUT_DIR}/hotspots_map.csv", index=False)
    print(f"  Top {len(hotspots)} priority hotspots + {len(hotspots_map)} map zones saved")

    # Hourly pattern (IST)
    hourly = df.groupby("hour").size().reindex(range(24), fill_value=0)
    hourly.to_frame("violations").reset_index().to_csv(f"{OUT_DIR}/hourly_pattern.csv", index=False)

    # Daily pattern
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = df.groupby("dayofweek").size().reindex(range(7), fill_value=0)
    daily_df = pd.DataFrame({"day": dow_names, "violations": daily.values})
    daily_df.to_csv(f"{OUT_DIR}/daily_pattern.csv", index=False)

    # Vehicle type pattern
    veh = df["vehicle_type"].value_counts().head(15)
    veh.to_frame("violations").reset_index().rename(columns={"index": "vehicle_type"}).to_csv(
        f"{OUT_DIR}/vehicle_pattern.csv", index=False)

    # Exploded individual violation types
    all_v = []
    for vl in df["violations_list"]:
        all_v.extend(vl)
    vt = pd.Series(all_v).value_counts().head(15)
    vt.to_frame("count").reset_index().rename(columns={"index": "violation"}).to_csv(
        f"{OUT_DIR}/violation_types.csv", index=False)

    # Police station load
    st = df["police_station"].value_counts().head(20)
    st.to_frame("violations").reset_index().rename(columns={"index": "station"}).to_csv(
        f"{OUT_DIR}/station_load.csv", index=False)

    # Heatmap points — downsample to keep the map fast (max 15k points)
    sample = df.sample(min(15000, len(df)), random_state=42)
    sample[["latitude", "longitude"]].to_csv(f"{OUT_DIR}/heatmap_points.csv", index=False)

    # Cumulative coverage curve — how much of all violations the top-N zones capture
    df["grid_lat"] = (df["latitude"] / GRID).round() * GRID
    df["grid_lon"] = (df["longitude"] / GRID).round() * GRID
    zone_counts = (
        df.groupby(["grid_lat", "grid_lon"]).size().sort_values(ascending=False).values
    )
    total = zone_counts.sum()
    cum = np.cumsum(zone_counts)
    checkpoints = [10, 25, 50, 100, 150, 250, 500, 750, 1000]
    rows = []
    for n in checkpoints:
        if n <= len(zone_counts):
            rows.append({"zones": n, "coverage_pct": round(100 * cum[n - 1] / total, 1)})
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/coverage_curve.csv", index=False)

    # Headline summary
    peak_hour = int(hourly.idxmax())
    busiest_day = dow_names[int(daily.idxmax())]
    top50_share = hotspots["violations"].sum() / len(df) * 100
    summary = {
        "total_violations": int(len(df)),
        "date_start": str(df["date"].min()),
        "date_end": str(df["date"].max()),
        "days_covered": int(n_days),
        "avg_per_day": int(round(len(df) / n_days)),
        "unique_locations": int(df.groupby(["latitude", "longitude"]).ngroups),
        "named_junctions": int((df["junction_name"] != "No Junction").sum()),
        "no_junction_share": round((df["junction_name"] == "No Junction").mean() * 100, 1),
        "peak_hour_ist": peak_hour,
        "peak_hour_label": f"{peak_hour:02d}:00–{peak_hour+1:02d}:00 IST",
        "busiest_day": busiest_day,
        "top_violation": vt.index[0],
        "top50_hotspot_share": round(top50_share, 1),
        "police_stations": int(df["police_station"].nunique()),
    }
    with open(f"{OUT_DIR}/summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nAll artifacts written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
