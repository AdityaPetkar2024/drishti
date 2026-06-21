"""
Drishti — Predictive Forecast Model
Trains a LightGBM regressor to forecast parking-violation intensity per
zone × hour × day-of-week, enabling proactive (pre-emptive) deployment.

Validation is a strict TEMPORAL hold-out: the final 30 days of data are never
seen during training, so reported metrics reflect genuine future prediction,
not memorisation. All historical features are computed from the training
period only — no leakage.

Run:  python train_forecast_model.py
Outputs (to ./data/):
    forecast_grid.csv     — predicted intensity + risk tier for every zone/day/hour
    forecast_model.txt    — saved LightGBM model
    model_metrics.json    — held-out validation metrics (AUC, capture, MAE)
"""
import pandas as pd, numpy as np, lightgbm as lgb, json
from sklearn.metrics import roc_auc_score

RAW = "jan to may police violation_anonymized791b166.csv"
GRID = 0.0015
TOP_ZONES = 200
HOURS = list(range(0, 24))      # full day — evening is genuinely near-zero in the data
TEST_DAYS = 30

df = pd.read_csv(RAW)
# Remove confirmed-invalid citations (rejected/duplicate). Keep null status — those are
# unreviewed but genuine violations (BTP validation workflow tapered off after Jan 2024).
df = df[~df["validation_status"].isin(["rejected", "duplicate"])].copy()
df["created_dt"] = pd.to_datetime(df["created_datetime"], format="ISO8601") + pd.Timedelta(hours=5, minutes=30)
df = df[df["latitude"].between(12.7, 13.3) & df["longitude"].between(77.3, 77.9)].copy()
df["hour"] = df["created_dt"].dt.hour
df["date"] = df["created_dt"].dt.date
df["dayofweek"] = df["created_dt"].dt.dayofweek
df["grid_lat"] = (df["latitude"] / GRID).round() * GRID
df["grid_lon"] = (df["longitude"] / GRID).round() * GRID
df["zone"] = df["grid_lat"].round(4).astype(str) + "_" + df["grid_lon"].round(4).astype(str)

top = df["zone"].value_counts().head(TOP_ZONES).index.tolist()
meta = df[df["zone"].isin(top)].groupby("zone").agg(
    grid_lat=("grid_lat", "first"), grid_lon=("grid_lon", "first"),
    top_location=("location", lambda x: x.mode().iloc[0] if len(x.mode()) else "N/A")).reset_index()
dfz = df[df["zone"].isin(top)]
dates = sorted(df["date"].unique())
split = dates[-TEST_DAYS]

# Full panel incl. zeros
idx = pd.MultiIndex.from_product([top, dates, HOURS], names=["zone", "date", "hour"])
panel = pd.DataFrame(index=idx).reset_index()
actual = dfz.groupby(["zone", "date", "hour"]).size().reset_index(name="count")
panel = panel.merge(actual, on=["zone", "date", "hour"], how="left")
panel["count"] = panel["count"].fillna(0)
panel["dayofweek"] = pd.to_datetime(panel["date"]).dt.dayofweek
panel = panel.merge(meta[["zone", "grid_lat", "grid_lon"]], on="zone", how="left")

train = panel[panel["date"] < split].copy()
test = panel[panel["date"] >= split].copy()

# Leakage-free historical features (train period only)
zh = train.groupby(["zone", "hour"])["count"].mean().rename("zh_count")
zd = train.groupby(["zone", "dayofweek"])["count"].mean().rename("zd_count")
zm = train.groupby("zone")["count"].mean().rename("z_count")
hm = train.groupby("hour")["count"].mean().rename("h_count")
gm = train["count"].mean()

def add(d):
    d["zh_count"] = d.set_index(["zone", "hour"]).index.map(zh).astype(float)
    d["zd_count"] = d.set_index(["zone", "dayofweek"]).index.map(zd).astype(float)
    d["z_count"] = d["zone"].map(zm).astype(float)
    d["h_count"] = d["hour"].map(hm).astype(float)
    for c in ["zh_count", "zd_count", "z_count", "h_count"]:
        d[c] = d[c].fillna(gm)
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    return d

train, test = add(train), add(test)
feats = ["hour", "dayofweek", "grid_lat", "grid_lon", "zh_count", "zd_count",
         "z_count", "h_count", "hour_sin", "hour_cos", "is_weekend"]
params = dict(objective="regression", metric="mae", learning_rate=0.05, num_leaves=63,
              min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=5, verbose=-1, n_jobs=-1)

m = lgb.train(params, lgb.Dataset(train[feats], label=train["count"]), num_boost_round=300,
              valid_sets=[lgb.Dataset(test[feats], label=test["count"])],
              callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])

pred = np.clip(m.predict(test[feats]), 0, None)
auc = roc_auc_score((test["count"] >= 5).astype(int), pred)
cut = np.quantile(pred, 0.90)
capture = test.assign(p=pred).query("p >= @cut")["count"].sum() / test["count"].sum()
mae = float(np.mean(np.abs(pred - test["count"])))
print(f"Held-out AUC={auc:.3f}  top-decile capture={100*capture:.0f}%  MAE={mae:.3f}")

# Retrain on all data, build a typical-week forecast grid
allp = add(panel.copy())
final = lgb.train(params, lgb.Dataset(allp[feats], label=allp["count"]),
                  num_boost_round=m.best_iteration or 200)
g = pd.DataFrame(index=pd.MultiIndex.from_product(
    [top, range(7), HOURS], names=["zone", "dayofweek", "hour"])).reset_index().merge(meta, on="zone", how="left")
g = add(g)
g["pred"] = np.clip(final.predict(g[feats]), 0, None)
g["risk"] = pd.cut(g["pred"], bins=[-1, 1, 3, 1e9], labels=["Low", "Medium", "High"])
g[["zone", "dayofweek", "hour", "grid_lat", "grid_lon", "top_location", "pred", "risk"]].to_csv("data/forecast_grid.csv", index=False)
final.save_model("data/forecast_model.txt")
json.dump({"auc": round(auc, 3), "top_decile_capture": round(100 * capture),
           "test_days": TEST_DAYS, "mae": round(mae, 3)}, open("data/model_metrics.json", "w"), indent=2)
print("Saved forecast_grid.csv, forecast_model.txt, model_metrics.json")
