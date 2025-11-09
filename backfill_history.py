# backfill_history.py
# One-time bootstrap of a 90-day rolling history using 180 days of ACLED events.
#
# Purpose:
#   Generate 90 daily snapshots (from 89 days ago through today) with properly computed
#   risk indicators (DCR, PRS, priority) based on rolling windows. This provides an initial
#   history that the daily pipeline will progressively replace over the next 90 days.
#
# Requires:
#   - .env with ACLED_USER, ACLED_PASS
#   - data/mex_admbnda_govmex_20210618_SHP/ (HDX administrative boundaries)
#   - out/adm2_risk_daily.csv (run pipeline.py once first to generate static inputs)
#
# Outputs:
#   - out/history/adm2_risk_daily_history.csv (90 days × ~2500 municipalities)
#   - out/history/cast_state_history.csv (90 days of state-level CAST)
#   - out/history/acled_event_counts_history.csv (90 days of national event counts)
#
# Usage:
#   python backfill_history.py

import os
import pathlib
import datetime as dt
import json
import pandas as pd
import numpy as np
import geopandas as gpd
import requests
from shapely.geometry import Point
from dotenv import load_dotenv
from libpysal.weights import Queen
from unidecode import unidecode

# ----------------- config / paths -----------------
load_dotenv()
ACLED_USER = os.getenv("ACLED_USER")
ACLED_PASS = os.getenv("ACLED_PASS")
SSL_VERIFY = os.getenv("SSL_VERIFY", "true").lower() == "true"

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
HIST_DIR = OUT_DIR / "history"
HIST_DIR.mkdir(parents=True, exist_ok=True)

FACT_CSV = OUT_DIR / "adm2_risk_daily.csv"
ADM2_SHP = DATA_DIR / "mex_admbnda_govmex_20210618_SHP" / "mex_admbnda_adm2_govmex_20210618.shp"

FACT_HISTORY_CSV = HIST_DIR / "adm2_risk_daily_history.csv"
CAST_HISTORY_CSV = HIST_DIR / "cast_state_history.csv"
EVENTS_SUMMARY_HISTORY_CSV = HIST_DIR / "acled_event_counts_history.csv"
EVENT_JOIN_CSV = HIST_DIR / "acled_events_backfill_join.csv"
ANOMALIES_CSV = HIST_DIR / "v30_anomalies_backfill.csv"

# Optional: state-level CAST history by day (if pipeline produced it)
CAST_DAILY_CSV = OUT_DIR / "cast_state_daily.csv"

ACLED_TOKEN_URL = "https://acleddata.com/oauth/token"
ACLED_READ_URL  = "https://acleddata.com/api/acled/read"
ACLED_CAST_URL = "https://acleddata.com/api/cast/read"
ACLED_FIELDS = "event_id_cnty|event_date|event_type|admin1|admin2|latitude|longitude|country"
VIOLENT = {"Violence against civilians", "Battles", "Explosions/Remote violence"}

# Weights (match pipeline.py)
W_DCR = {
    "V3m": 0.35,
    "S":   0.15,
    "A":   0.30,
    "MVI": 0.20,
}
W_PRS_CAST = {
    "V30":  0.30,
    "dV30": 0.25,
    "S":    0.10,
    "CAST": 0.18,
    "A":    0.12,
    "MVI":  0.05,
}
W_PRS_NC = {
    "V30":  0.40,
    "dV30": 0.30,
    "S":    0.10,
    "A":    0.12,
    "MVI":  0.08,
}

# ----------------- helpers -----------------
def get_token(u, p):
    r = requests.post(
        ACLED_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        data={"username": u, "password": p, "grant_type": "password", "client_id": "acled"},
        timeout=60, verify=SSL_VERIFY
    )
    r.raise_for_status()
    return r.json()["access_token"]

def acled_fetch(params, token, limit=5000, max_pages=200):
    frames = []
    for page in range(1, max_pages+1):
        q = {"_format": "json", "page": page, "limit": limit}
        q.update(params)
        r = requests.get(
            ACLED_READ_URL,
            headers={"Authorization": f"Bearer {token}", "Accept":"application/json"},
            params=q, timeout=120, verify=SSL_VERIFY
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data: break
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def winsor01(s, lo=0.05, hi=0.95):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0: return pd.Series(np.zeros(len(s)), index=s.index)
    a, b = np.nanquantile(s, lo), np.nanquantile(s, hi)
    s2 = np.clip(s, a, b)
    return (s2 - a) / (b - a) if b > a else pd.Series(np.zeros(len(s)), index=s.index)

def normalize_adm1_join_name(s: pd.Series) -> pd.Series:
    """Canonical ADM1 key to join CAST to ADM2 (matches pipeline)."""
    def canon(x: str) -> str:
        if pd.isna(x):
            return ""
        y = unidecode(str(x)).strip()
        y = " ".join(w.capitalize() for w in y.split())
        rep = {
            "Distrito Federal": "Ciudad De Mexico",
            "Mexico City": "Ciudad De Mexico",
            "Federal District": "Ciudad De Mexico",
            "Coahuila De Zaragoza": "Coahuila",
            "Michoacan De Ocampo": "Michoacan",
            "Veracruz De Ignacio De La Llave": "Veracruz",
        }
        return rep.get(y, y)
    return s.apply(canon)

def first_of_month(d: dt.date) -> dt.date:
    """Return first day of the month containing d."""
    return d.replace(day=1)

def add_month(d: dt.date) -> dt.date:
    """Return first day of the next month after d."""
    y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return dt.date(y, m, 1)

def _month_name(n: int) -> str:
    return ["January","February","March","April","May","June","July","August","September","October","November","December"][n-1]

def fetch_cast_for_month(token: str, year: int, month_num: int) -> pd.DataFrame:
    """
    Fetch CAST for Mexico for a specific forecast month/year (one-month-ahead use case).
    Returns DataFrame with columns: adm1_join, cast_state, forecast_date.
    """
    headers = {"Authorization": f"Bearer {token}"}
    frames = []
    page, limit = 1, 5000
    while True:
        params = {"_format": "csv", "country": "Mexico", "year": year, "month": _month_name(month_num), "limit": limit, "page": page}
        r = requests.get(ACLED_CAST_URL, headers=headers, params=params, timeout=60, verify=SSL_VERIFY)
        r.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(r.text))
        if df.empty:
            break
        frames.append(df)
        if len(df) < limit:
            break
        page += 1
    cast_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if cast_raw.empty:
        return pd.DataFrame(columns=["adm1_join","cast_state","forecast_date"])
    cast_raw = cast_raw.rename(columns={"admin1":"adm1_name","total_forecast":"cast_raw"})
    cast_raw["cast_raw"] = pd.to_numeric(cast_raw["cast_raw"], errors="coerce").fillna(0.0)
    cast_raw["month_num"] = month_num
    # Use a scalar Timestamp; pandas will broadcast this value to all rows
    cast_raw["forecast_date"] = pd.Timestamp(year=int(year), month=int(month_num), day=1)
    cast_raw["adm1_join"] = normalize_adm1_join_name(cast_raw["adm1_name"])
    # Per-month scaling to 0-1 (winsor 5-95)
    a, b = np.nanquantile(cast_raw["cast_raw"], 0.05), np.nanquantile(cast_raw["cast_raw"], 0.95)
    x = np.clip(cast_raw["cast_raw"].astype(float), a, b)
    cast_raw["cast_state"] = (x - a) / (b - a) if b > a else 0.5
    out = cast_raw[["adm1_join","cast_state","forecast_date"]].drop_duplicates().copy()
    return out

# ----------------- load static assets -----------------
print("=" * 60)
print("BACKFILL HISTORY: 90-day bootstrap")
print("=" * 60)

print(f"Using ADM2 shapefile: {ADM2_SHP}")
assert ADM2_SHP.exists(), f"Missing {ADM2_SHP}; ensure data files are present."
assert FACT_CSV.exists(), f"Missing {FACT_CSV}; run pipeline.py once first."

static_today = pd.read_csv(FACT_CSV)
for k in ["adm2_code","adm1_name","adm2_name"]:
    static_today[k] = static_today[k].astype(str)
# minimal static features reused across all days
static = static_today[["adm2_code","adm1_name","adm2_name","pop_wra","access_A","mvi","cast_state"]].copy()
static = static.sort_values("adm2_code").drop_duplicates("adm2_code", keep="last")
static["pop_wra"] = pd.to_numeric(static["pop_wra"], errors="coerce").fillna(0.0)
static["adm1_join"] = normalize_adm1_join_name(static["adm1_name"])

# Optional: state-level CAST history by day (if pipeline produced it)
cast_daily = None
if CAST_DAILY_CSV.exists():
    _cd = pd.read_csv(CAST_DAILY_CSV)
    # Expect columns: snapshot_date, adm1_name, cast_state
    if "snapshot_date" in _cd.columns:
        _cd["snapshot_date"] = pd.to_datetime(_cd["snapshot_date"], errors="coerce").dt.date
    cast_daily = _cd[["snapshot_date", "adm1_name", "cast_state"]].copy()

# ADM2 + weights matrix
print("Loading ADM2 polygons and building spatial weights...")
adm2 = gpd.read_file(ADM2_SHP, encoding="utf-8")
if adm2.crs is None or adm2.crs.to_epsg() != 4326:
    adm2 = adm2.to_crs(4326)
adm2 = adm2.rename(columns={"ADM2_PCODE":"adm2_code","ADM2_ES":"adm2_name","ADM1_ES":"adm1_name"})[
    ["adm1_name","adm2_name","adm2_code","geometry"]
].copy()
for k in ["adm2_code","adm1_name","adm2_name"]:
    adm2[k] = adm2[k].astype(str)
adm2["adm1_join"] = normalize_adm1_join_name(adm2["adm1_name"])

# align to static ADM2 universe
adm2 = adm2.merge(static[["adm2_code"]].drop_duplicates(), on="adm2_code", how="inner")
wq = Queen.from_dataframe(adm2, ids=adm2["adm2_code"].tolist())
wq.transform = "R"
W = wq.sparse  # scipy csr
adm2_index = adm2.drop(columns="geometry").copy()

# ----------------- fetch 180d of events -----------------
print(f"Fetching 180 days of ACLED events (for rolling windows)...")
token = get_token(ACLED_USER, ACLED_PASS)
# probe allowed end date (fallback to today if missing)
probe = requests.get(ACLED_READ_URL, headers={"Authorization": f"Bearer {token}"}, params={"_format":"json","limit":1}, verify=SSL_VERIFY, timeout=30).json()
cap = (probe.get("data_query_restrictions") or {}).get("date_recency", {})
end_allowed = pd.to_datetime(cap.get("date"), errors="coerce").date() if cap else dt.date.today()

# After computing end_allowed (ACLED recency cap):
print(f"ACLED recency cap: {end_allowed}")

run_today = dt.date.today()
snapshot_dates = [run_today - dt.timedelta(days=i) for i in range(0, 90)]
snapshot_dates = list(reversed(snapshot_dates))
print(f"Will generate 90 snapshots (run dates) from {snapshot_dates[0]} to {snapshot_dates[-1]}")

snap_specs = []
for snap_date in snapshot_dates:
    offset_days = (run_today - snap_date).days
    anchor = end_allowed - dt.timedelta(days=offset_days)
    snap_specs.append({"run_date": snap_date, "anchor": anchor})

needed_months = sorted({ add_month(first_of_month(spec["run_date"])) for spec in snap_specs })
print(f"CAST one-month-ahead months needed: {[str(x) for x in needed_months]}")

earliest_anchor = min(spec["anchor"] for spec in snap_specs)
events_start = earliest_anchor - dt.timedelta(days=89)
print(f"Event window start: {events_start}")

event_params = {
    "country": "Mexico",
    "event_date": f"{events_start.isoformat()}|{end_allowed.isoformat()}",
    "event_date_where": "BETWEEN",
    "fields": ACLED_FIELDS,
}
events_raw = acled_fetch(event_params, token)
if events_raw.empty:
    print("Warning: ACLED returned no violent events for the backfill window.")
    j = pd.DataFrame(columns=["adm2_code","adm1_name","adm2_name","event_date"])
else:
    events_raw = events_raw[events_raw["event_type"].isin(VIOLENT)].copy()
    if "country" in events_raw.columns:
        events_raw = events_raw[events_raw["country"] == "Mexico"]
    for col in ("latitude","longitude"):
        events_raw[col] = pd.to_numeric(events_raw[col], errors="coerce")
    events_raw["event_date"] = pd.to_datetime(events_raw["event_date"], errors="coerce").dt.date
    events_raw = events_raw.dropna(subset=["event_date","latitude","longitude"])
    events_gdf = gpd.GeoDataFrame(
        events_raw,
        geometry=gpd.points_from_xy(events_raw["longitude"], events_raw["latitude"]),
        crs=4326,
    )
    joined = gpd.sjoin(
        events_gdf,
        adm2[["adm2_code","geometry"]],
        how="left",
        predicate="intersects",
    ).drop(columns=["index_right"], errors="ignore")
    joined = joined.merge(adm2_index, on="adm2_code", how="left")
    joined["event_date"] = pd.to_datetime(joined["event_date"], errors="coerce").dt.date
    joined["adm2_code"] = joined["adm2_code"].astype(str)
    j = joined.dropna(subset=["event_date"]).copy()

if not j.empty:
    j.to_csv(EVENT_JOIN_CSV, index=False)

cast_by_month = {}
for month_date in needed_months:
    cast_df = fetch_cast_for_month(token, month_date.year, month_date.month)
    cast_by_month[month_date] = cast_df
    label = month_date.strftime("%Y-%m")
    if cast_df.empty:
        print(f"Warning: no CAST rows returned for {label}; falling back to static cast_state.")
    else:
        print(f"Fetched {len(cast_df)} CAST rows for {label}.")

rows, cast_hist, ev_hist = [], [], []

for spec in snap_specs:
    run_date = spec["run_date"]
    anchor = min(spec["anchor"], end_allowed)

    w30_start    = anchor - dt.timedelta(days=29)
    w90_start    = anchor - dt.timedelta(days=89)
    prev30_start = anchor - dt.timedelta(days=59)
    prev30_end   = anchor - dt.timedelta(days=30)

    mask90 = (j["event_date"] >= w90_start) & (j["event_date"] <= anchor)
    mask30 = (j["event_date"] >= w30_start) & (j["event_date"] <= anchor)
    mask_prev = (j["event_date"] >= prev30_start) & (j["event_date"] <= prev30_end)
    d90  = j[mask90]
    d30  = j[mask30]
    dP30 = j[mask_prev]

    c90  = d90.groupby("adm2_code").size()
    c30  = d30.groupby("adm2_code").size()
    cP30 = dP30.groupby("adm2_code").size()

    df = adm2_index.drop(columns="adm1_join").copy()
    df["events_90v"]   = df["adm2_code"].map(c90).fillna(0).astype(int)
    df["events30"]     = df["adm2_code"].map(c30).fillna(0).astype(int)
    df["events_prevv"] = df["adm2_code"].map(cP30).fillna(0).astype(int)

    df = df.merge(static, on=["adm2_code","adm1_name","adm2_name"], how="left")
    df["adm1_join"] = normalize_adm1_join_name(df["adm1_name"])

    cast_month = add_month(first_of_month(run_date))
    cast_m = cast_by_month.get(cast_month)
    if cast_m is not None and not cast_m.empty:
        df = df.merge(cast_m[["adm1_join","cast_state"]], on="adm1_join", how="left", suffixes=("", "_dyn"))
        df["cast_state_use"] = df["cast_state_dyn"].fillna(df["cast_state"])
    else:
        df["cast_state_use"] = df["cast_state"]
    df = df.drop(columns=["cast_state_dyn"], errors="ignore")

    den = df["pop_wra"].astype(float).clip(lower=0)
    df["v30"]        = np.where(den > 0, 1e5 * df["events30"]    .astype(float) / den, 0.0)
    df["v3m"]        = np.where(den > 0, 1e5 * df["events_90v"]  .astype(float) / den, 0.0)
    df["v30_prev"]   = np.where(den > 0, 1e5 * df["events_prevv"].astype(float) / den, 0.0)
    df["dlt_v30_raw"] = df["v30"] - df["v30_prev"]

    v30_vec = df["v30"].to_numpy(dtype=float)
    df["spillover"] = pd.Series(np.nan_to_num(W.dot(v30_vec), nan=0.0), index=df.index)

    V30 = winsor01(df["v30"])
    V3m = winsor01(df["v3m"])
    d = df["dlt_v30_raw"].astype(float)
    p1, p9 = np.nanpercentile(d, 10), np.nanpercentile(d, 90)
    d_clip = np.clip(d, p1, p9)
    d_unit = -1 + 2 * (d_clip - p1)/(p9 - p1) if p9 > p1 else np.zeros_like(d_clip)
    dV30 = winsor01(pd.Series(d_unit, index=df.index))
    S_norm    = winsor01(df["spillover"])
    CAST_norm = pd.to_numeric(df["cast_state_use"], errors="coerce").fillna(0.0).clip(0, 1)
    A_norm    = pd.to_numeric(df["access_A"],       errors="coerce").fillna(0.0).clip(0, 1)
    MVI_norm  = pd.to_numeric(df["mvi"],            errors="coerce").fillna(0.0).clip(0, 1)

    DCR = V3m*W_DCR["V3m"] + S_norm*W_DCR["S"] + A_norm*W_DCR["A"] + MVI_norm*W_DCR["MVI"]
    if CAST_norm.max() > 0:
        PRS = (V30*W_PRS_CAST["V30"] + dV30*W_PRS_CAST["dV30"] + S_norm*W_PRS_CAST["S"] +
               CAST_norm*W_PRS_CAST["CAST"] + A_norm*W_PRS_CAST["A"] + MVI_norm*W_PRS_CAST["MVI"])
    else:
        PRS = (V30*W_PRS_NC["V30"] + dV30*W_PRS_NC["dV30"] + S_norm*W_PRS_NC["S"] +
               A_norm*W_PRS_NC["A"] + MVI_norm*W_PRS_NC["MVI"])

    rows.append(pd.DataFrame({
        "snapshot_date": run_date,
        "data_as_of": anchor,
        "adm2_code": df["adm2_code"],
        "adm1_name": df["adm1_name"],
        "adm2_name": df["adm2_name"],
        "DCR100": 100*DCR,
        "PRS100": 100*PRS,
        "priority100": 100*(0.6*PRS + 0.4*DCR),
        "v30": df["v30"],
        "v3m": df["v3m"],
        "dlt_v30_raw": df["dlt_v30_raw"],
        "spillover": df["spillover"]
    }))

    cast_snapshot = df[["adm1_join","cast_state_use"]].copy()
    cast_snapshot["cast_state_use"] = pd.to_numeric(cast_snapshot["cast_state_use"], errors="coerce").fillna(0.0)
    cast_snapshot = (
        cast_snapshot.groupby("adm1_join", as_index=False)["cast_state_use"]
        .mean()
        .rename(columns={"cast_state_use": "cast_state"})
    )
    cast_snapshot["snapshot_date"] = run_date
    cast_hist.append(cast_snapshot[["snapshot_date","adm1_join","cast_state"]])

    ev_hist.append(pd.DataFrame({
        "snapshot_date": [run_date],
        "data_as_of": [anchor],
        "events_30d": [int(df["events30"].sum())],
        "events_90d": [int(df["events_90v"].sum())],
        "events_prev30": [int(df["events_prevv"].sum())]
    }))

hist = pd.concat(rows, ignore_index=True)
hist = hist.sort_values(["snapshot_date","adm2_code"]).reset_index(drop=True)
hist.to_csv(FACT_HISTORY_CSV, index=False)
pd.concat(cast_hist, ignore_index=True).to_csv(CAST_HISTORY_CSV, index=False)
pd.concat(ev_hist, ignore_index=True).to_csv(EVENTS_SUMMARY_HISTORY_CSV, index=False)

print("\n" + "=" * 60)
print("BACKFILL COMPLETE")
print("=" * 60)
print(f"Generated {len(hist)} snapshot rows ({len(hist)//len(adm2_index)} days × {len(adm2_index)} municipalities)")
print(f"\nWrote:")
print(f"  {FACT_HISTORY_CSV}")
print(f"    - {len(hist):,} rows")
print(f"  {CAST_HISTORY_CSV}")
print(f"    - {len(pd.concat(cast_hist, ignore_index=True)):,} rows")
print(f"  {EVENTS_SUMMARY_HISTORY_CSV}")
print(f"    - {len(pd.concat(ev_hist, ignore_index=True)):,} rows")
print(f"\nDate range: {hist['snapshot_date'].min()} to {hist['snapshot_date'].max()}")
print(f"\nThe daily pipeline will progressively replace these snapshots")
print(f"with fresh data over the next 90 days.")
