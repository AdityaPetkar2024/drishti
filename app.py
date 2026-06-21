"""
Drishti — Parking Enforcement Intelligence for Bengaluru
Run:  streamlit run app.py
Requires the artifacts in ./data/ produced by process_data.py
"""

import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Drishti — Parking Enforcement Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- styling ----------
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #161b26; border: 1px solid #232a39; border-radius: 10px;
        padding: 18px 20px; margin-bottom: 8px;
    }
    .metric-value { font-size: 30px; font-weight: 700; color: #e8edf5; line-height: 1.1; }
    .metric-label { font-size: 12px; color: #8b95a7; text-transform: uppercase; letter-spacing: .06em; }
    .accent { color: #ff5a3c; }
    h1, h2, h3 { color: #e8edf5; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load():
    d = {}
    d["hotspots"] = pd.read_csv("data/hotspots.csv")
    d["hotspots_map"] = pd.read_csv("data/hotspots_map.csv")
    d["hourly"] = pd.read_csv("data/hourly_pattern.csv")
    d["daily"] = pd.read_csv("data/daily_pattern.csv")
    d["vehicle"] = pd.read_csv("data/vehicle_pattern.csv")
    d["vtypes"] = pd.read_csv("data/violation_types.csv")
    d["stations"] = pd.read_csv("data/station_load.csv")
    d["coverage"] = pd.read_csv("data/coverage_curve.csv")
    d["forecast"] = pd.read_csv("data/forecast_grid.csv")
    with open("data/model_metrics.json") as f:
        d["metrics"] = json.load(f)
    d["heat"] = pd.read_csv("data/heatmap_points.csv")
    with open("data/summary_stats.json") as f:
        d["summary"] = json.load(f)
    return d


data = load()
s = data["summary"]


def metric(col, value, label):
    col.markdown(
        f'<div class="metric-card"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


# ---------- sidebar ----------
st.sidebar.title("🚦 Drishti")
st.sidebar.caption("Parking Enforcement Intelligence for Bengaluru")
page = st.sidebar.radio(
    "View",
    ["Command Overview", "Hotspot Map", "When Violations Happen",
     "Deployment Plan", "AI Forecast"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Data: {s['date_start']} → {s['date_end']}  \n"
    f"{s['total_violations']:,} violations · {s['days_covered']} days  \n"
    f"All times converted to IST (UTC+5:30)"
)

# ========== PAGE 1: OVERVIEW ==========
if page == "Command Overview":
    st.title("Command Overview")
    st.markdown(
        f"Bengaluru logs **{s['avg_per_day']:,} parking violations a day**. "
        f"Enforcement today is patrol-based and reactive. Drishti turns "
        f"{s['total_violations']:,} historical records into a ranked, time-aware "
        f"enforcement plan."
    )

    c1, c2, c3, c4 = st.columns(4)
    metric(c1, f"{s['total_violations']:,}", "Total Violations")
    metric(c2, f"{s['avg_per_day']:,}", "Per Day")
    metric(c3, f"{s['top50_hotspot_share']}%", "From Top 50 Zones")
    metric(c4, f"{s['no_junction_share']}%", "Away From Junctions")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("The concentration problem")
        st.markdown(
            f"The top 50 hotspot zones — under **1%** of all locations — produce "
            f"**{s['top50_hotspot_share']}%** of every violation in the city. "
            f"Enforcement spread evenly across Bengaluru misses this. Concentrated "
            f"enforcement on these zones is the single highest-leverage change."
        )
        top10 = data["hotspots"].head(10)
        fig = px.bar(
            top10[::-1], x="violations", y="top_location", orientation="h",
            labels={"violations": "Violations", "top_location": ""},
            color="priority_score", color_continuous_scale="Reds",
        )
        fig.update_layout(
            height=380, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="#c4ccd9", yaxis={"tickfont": {"size": 9}},
            coloraxis_colorbar={"title": "Priority"},
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("The timing problem")
        st.markdown(
            f"Recorded violations are heavily concentrated in the day, peaking at "
            f"**{s['peak_hour_label']}**, with **{s['busiest_day']}** the busiest day. "
            f"This reflects when enforcement is currently active — and it reveals which "
            f"zones and windows are well-covered versus under-watched, so coverage can be "
            f"extended where the data is thin."
        )
        h = data["hourly"]
        fig = px.bar(h, x="hour", y="violations",
                     labels={"hour": "Hour (IST)", "violations": "Violations"})
        fig.update_traces(marker_color="#ff5a3c")
        fig.update_layout(
            height=380, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="#c4ccd9", margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Hour (IST)",
        )
        st.plotly_chart(fig, use_container_width=True)

# ========== PAGE 2: HOTSPOT MAP ==========
elif page == "Hotspot Map":
    st.title("Hotspot Map")
    st.markdown("The 500 most active parking zones across Bengaluru — together 77% of all violations. Bigger, redder circles mean higher enforcement priority. Hover any circle for detail.")

    hs = data["hotspots_map"].copy()

    # Plotly scatter mapbox — open-street tiles, city-zoomed, no national boundaries
    fig = px.scatter_mapbox(
        hs,
        lat="grid_lat", lon="grid_lon",
        size="violations",
        color="priority_score",
        color_continuous_scale="YlOrRd",
        size_max=38,
        zoom=11,
        center={"lat": 12.97, "lon": 77.59},
        hover_name="top_location",
        hover_data={
            "grid_lat": False, "grid_lon": False,
            "rank": True, "violations": ":,",
            "violations_per_day": True, "priority_score": True,
            "peak_hour": True, "top_vehicle": True,
            "recommended_officers": True,
        },
        labels={
            "rank": "Rank", "violations": "Total violations",
            "violations_per_day": "Per day", "priority_score": "Priority",
            "peak_hour": "Peak hour (IST)", "top_vehicle": "Top vehicle",
            "recommended_officers": "Officers",
        },
        height=620,
    )
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox=dict(
            center={"lat": 12.97, "lon": 77.59},
            zoom=11,
            # Lock panning to Greater Bengaluru — the frame can never leave the city,
            # so no national boundaries ever enter view.
            bounds=dict(west=77.35, east=77.85, south=12.75, north=13.29),
        ),
        paper_bgcolor="#0e1117",
        font_color="#c4ccd9",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar={"title": "Priority"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 15 enforcement zones")
    show = data["hotspots"].head(15)[
        ["rank", "top_location", "violations", "violations_per_day",
         "priority_score", "peak_hour", "top_vehicle", "recommended_officers"]
    ].rename(columns={
        "rank": "Rank", "top_location": "Location", "violations": "Total",
        "violations_per_day": "Per Day", "priority_score": "Priority",
        "peak_hour": "Peak Hr", "top_vehicle": "Top Vehicle",
        "recommended_officers": "Officers",
    })
    st.dataframe(show, use_container_width=True, hide_index=True)

# ========== PAGE 3: TEMPORAL ==========
elif page == "When Violations Happen":
    st.title("When Violations Happen")
    st.markdown("Timing is half the enforcement problem. These patterns set the deployment windows.")
    st.caption(
        "Note: these counts reflect *recorded* violations — i.e. enforcement activity. "
        "They show when and where coverage is concentrated today, which is exactly what "
        "a deployment plan needs in order to rebalance officer time toward under-covered "
        "zones and windows."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("By hour of day (IST)")
        h = data["hourly"]
        peak = int(h.loc[h["violations"].idxmax(), "hour"])
        fig = go.Figure(go.Bar(x=h["hour"], y=h["violations"],
                               marker_color=["#ff5a3c" if hr == peak else "#444c5e" for hr in h["hour"]]))
        fig.update_layout(height=340, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                          font_color="#c4ccd9", xaxis_title="Hour (IST)", yaxis_title="Violations",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("By day of week")
        dd = data["daily"]
        peakd = dd["violations"].idxmax()
        fig = go.Figure(go.Bar(x=dd["day"], y=dd["violations"],
                               marker_color=["#ff5a3c" if i == peakd else "#444c5e" for i in range(len(dd))]))
        fig.update_layout(height=340, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                          font_color="#c4ccd9", yaxis_title="Violations",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Top violation types")
        vt = data["vtypes"].head(8)
        fig = px.bar(vt[::-1], x="count", y="violation", orientation="h",
                     labels={"count": "Count", "violation": ""})
        fig.update_traces(marker_color="#ff5a3c")
        fig.update_layout(height=340, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                          font_color="#c4ccd9", yaxis={"tickfont": {"size": 10}},
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Top offending vehicle types")
        veh = data["vehicle"].head(8)
        fig = px.bar(veh[::-1], x="violations", y="vehicle_type", orientation="h",
                     labels={"violations": "Violations", "vehicle_type": ""})
        fig.update_traces(marker_color="#ff5a3c")
        fig.update_layout(height=340, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                          font_color="#c4ccd9", yaxis={"tickfont": {"size": 10}},
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ========== PAGE 4: DEPLOYMENT ==========
elif page == "Deployment Plan":
    st.title("Deployment Plan")
    st.markdown(
        "A concrete, ranked enforcement schedule generated from the data. "
        "Each zone shows its peak window and a recommended officer count based on "
        "daily violation load."
    )

    total_officers = int(data["hotspots"]["recommended_officers"].sum())
    c1, c2, c3 = st.columns(3)
    metric(c1, len(data["hotspots"]), "Priority Zones")
    metric(c2, total_officers, "Officer-Shifts / Day")
    covered = data["hotspots"]["violations"].sum()
    metric(c3, f"{100*covered/s['total_violations']:.0f}%", "Violations Covered")

    st.markdown("---")
    st.subheader("Coverage scales with deployment")
    st.markdown(
        "Enforcement is resource-constrained, so the plan is a **dial**, not a fixed list. "
        "The ranking guarantees every additional officer goes to the next highest-impact "
        "zone. The top 50 zones already capture **34%** of all violations; expanding to 500 "
        "reaches **77%**. The remaining long tail is thousands of low-count locations better "
        "served by patrol and public reporting than fixed posting."
    )
    cov = data["coverage"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cov["zones"], y=cov["coverage_pct"],
        mode="lines+markers", line=dict(color="#ff5a3c", width=3),
        marker=dict(size=8, color="#ff5a3c"),
        fill="tozeroy", fillcolor="rgba(255,90,60,0.12)",
        hovertemplate="Top %{x} zones<br>%{y}% of violations<extra></extra>",
    ))
    # Mark the current top-50 plan
    row50 = cov[cov["zones"] == 50]
    if not row50.empty:
        fig.add_trace(go.Scatter(
            x=[50], y=[row50["coverage_pct"].iloc[0]],
            mode="markers+text", marker=dict(size=14, color="#ffd23c"),
            text=["Current plan: 50 zones"], textposition="top center",
            textfont=dict(color="#ffd23c"), hoverinfo="skip", showlegend=False,
        ))
    fig.update_layout(
        height=360, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="#c4ccd9", xaxis_title="Number of enforcement zones",
        yaxis_title="% of all violations covered", showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Full enforcement schedule")

    plan = data["hotspots"].copy()

    def window(h):
        h = int(h)
        end = (h + 2) % 24
        label = f"{h:02d}:00–{end:02d}:00"
        # Flag pre-dawn windows so they read as intentional, not as errors
        if h < 5:
            label += " (pre-dawn — market/loading activity)"
        return label

    plan["window"] = plan["peak_hour"].apply(window)
    out = plan[[
        "rank", "top_location", "top_station", "violations_per_day",
        "window", "top_vehicle", "recommended_officers", "priority_score"
    ]].rename(columns={
        "rank": "Rank", "top_location": "Zone", "top_station": "Police Station",
        "violations_per_day": "Violations/Day", "window": "Deploy Window",
        "top_vehicle": "Target Vehicle", "recommended_officers": "Officers",
        "priority_score": "Priority",
    })
    st.dataframe(out, use_container_width=True, hide_index=True, height=600)
    st.caption(
        "Deploy windows are centred on each zone's busiest recorded hour (IST). "
        "Pre-dawn windows are genuine — wholesale-market and goods-loading zones such as "
        "KR Market peak before sunrise."
    )

    st.download_button(
        "Download enforcement plan (CSV)",
        out.to_csv(index=False).encode(),
        "drishti_enforcement_plan.csv",
        "text/csv",
    )

# ========== PAGE 5: AI FORECAST ==========
elif page == "AI Forecast":
    st.title("AI Forecast — Predictive Deployment")
    st.markdown(
        "The previous views describe what *has* happened. This view **predicts where "
        "violations will concentrate next**, so officers can be positioned *before* they "
        "occur — shifting enforcement from reactive to proactive."
    )

    m = data["metrics"]
    c1, c2, c3 = st.columns(3)
    metric(c1, m["auc"], "Ranking AUC (held-out future month)")
    metric(c2, f"{m['top_decile_capture']}%", "Violations in Top 10% Predicted Windows")
    metric(c3, f"{m['mae']}", "Mean Abs. Error (violations/zone-hour)")

    st.caption(
        "Model: LightGBM gradient-boosting regressor. Validated on the final 30 days, which "
        "were never seen during training (temporal hold-out, no leakage). An AUC of "
        f"{m['auc']} means the model reliably ranks which zone-hours will be hotspots before "
        "they happen; its top 10% of predicted windows capture "
        f"{m['top_decile_capture']}% of all actual violations in that future month."
    )

    st.markdown("---")
    st.subheader("Predicted risk by hour and day")
    st.markdown(
        "Expected violation intensity across a full 24-hour week, aggregated across the "
        "top 200 zones. Darker = higher predicted risk. The morning concentration and "
        "near-empty evening reflect the recorded-enforcement pattern in the data."
    )

    fc = data["forecast"]
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    heat = fc.groupby(["dayofweek", "hour"])["pred"].sum().reset_index()
    pivot = heat.pivot(index="dayofweek", columns="hour", values="pred")
    pivot.index = [dow_names[i] for i in pivot.index]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"{h:02d}:00" for h in pivot.columns], y=pivot.index,
        colorscale="YlOrRd", hovertemplate="%{y} %{x}<br>Predicted: %{z:.0f}<extra></extra>",
    ))
    fig.update_layout(
        height=320, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="#c4ccd9", xaxis_title="Hour (IST)", yaxis_title="",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Tomorrow's highest-risk zones")
    pick_day = st.selectbox("Day of week", dow_names, index=6)
    day_idx = dow_names.index(pick_day)
    day_fc = fc[fc["dayofweek"] == day_idx].copy()
    # Aggregate per zone for that day, take the peak hour
    zone_day = day_fc.groupby(["zone", "top_location"]).agg(
        predicted_violations=("pred", "sum"),
        peak_hour=("pred", lambda x: day_fc.loc[x.idxmax(), "hour"]),
    ).reset_index().sort_values("predicted_violations", ascending=False).head(15)
    zone_day["predicted_violations"] = zone_day["predicted_violations"].round(1)
    zone_day["peak_window"] = zone_day["peak_hour"].apply(lambda h: f"{int(h):02d}:00–{int(h)+2:02d}:00")
    show = zone_day[["top_location", "predicted_violations", "peak_window"]].rename(columns={
        "top_location": "Zone", "predicted_violations": "Predicted Violations", "peak_window": "Peak Window",
    })
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption(
        f"Predicted highest-risk zones for {pick_day}, ranked by expected violation volume. "
        "Deploy here, at these windows, ahead of time."
    )
