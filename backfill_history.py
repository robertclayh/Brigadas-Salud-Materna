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

import os, pathlib, datetime as dt, json
import pandas as pd, numpy as np, geopandas as gpd, requests
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
adm2 = gpd.read_file(ADM2_SHP)
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

# ----------------- fetch 180d of events -----------------
print(f"Fetching 180 days of ACLED events (for rolling windows)...")
token = get_token(ACLED_USER, ACLED_PASS)
# probe allowed end date (fallback to today if missing)
probe = requests.get(ACLED_READ_URL, headers={"Authorization": f"Bearer {token}"}, params={"_format":"json","limit":1}, verify=SSL_VERIFY, timeout=30).json()
cap = (probe.get("data_query_restrictions") or {}).get("date_recency", {})
end_allowed = pd.to_datetime(cap.get("date"), errors="coerce").date() if cap else dt.date.today()

print(f"ACLED recency cap: {end_allowed}")
print(f"Will generate 90 snapshots from {end_allowed - dt.timedelta(days=89)} to {end_allowed}")

start_180 = end_allowed - dt.timedelta(days=179)
params = {
    "country": "Mexico",
    "event_date": f"{start_180}|{end_allowed}",
    "event_date_where": "BETWEEN",
    "fields": ACLED_FIELDS
}
print(f"Fetching ACLED events from {start_180} to {end_allowed}...")
ev = acled_fetch(params, token)
if ev.empty:
    raise SystemExit("No ACLED data returned for last 180 days; cannot backfill.")

print(f"Fetched {len(ev)} total events")
print(f"Fetched {len(ev)} total events")

# clean + filter violent
print("Filtering for violent events and valid coordinates...")
ev = ev[ev.get("country","")=="Mexico"].copy()
ev = ev.drop_duplicates(subset=["event_id_cnty"])
ev = ev[ev["event_type"].isin(VIOLENT)].copy()
print(f"  After filtering: {len(ev)} violent events")

ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce").dt.date
ev["latitude"] = pd.to_numeric(ev["latitude"], errors="coerce")
ev["longitude"] = pd.to_numeric(ev["longitude"], errors="coerce")
ev = ev.dropna(subset=["latitude","longitude"])
print(f"  With valid coordinates: {len(ev)} events")

g = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev["longitude"], ev["latitude"]), crs=4326)

# spatial join once to ADM2
print("Performing spatial join to ADM2 polygons...")
j = gpd.sjoin(g, adm2[["adm2_code","geometry"]], how="left", predicate="intersects").drop(columns=["index_right"])
j["adm2_code"] = j["adm2_code"].astype(str)
# Persist raw joined events (only minimal columns needed for later audits)
if not EVENT_JOIN_CSV.exists():
    j_out = j[["event_id_cnty","event_date","event_type","adm2_code","latitude","longitude"]].copy()
    j_out.to_csv(EVENT_JOIN_CSV, index=False)
    print(f"  Joined events written: {EVENT_JOIN_CSV}")
else:
    print(f"  Joined events already present: {EVENT_JOIN_CSV}")

# to speed up groupby by date
j["yyyymmdd"] = pd.to_datetime(j["event_date"]).dt.date
print(f"  Events spatially joined: {len(j)}")

# ensure a full ADM2 index order
adm2_index = adm2[["adm2_code","adm1_name","adm2_name"]].copy()
idx_map = {code:i for i,code in enumerate(adm2_index["adm2_code"])}

# ----------------- fetch CAST one-month-ahead for each snapshot -----------------
# Build set of forecast months needed for 90 snapshots (one-month-ahead of each day)
def first_of_month(d: dt.date) -> dt.date:
    return d.replace(day=1)
def add_month(d: dt.date) -> dt.date:
    y, m = (d.year + (d.month // 12), 1) if d.month == 12 else (d.year, d.month + 1)
    return dt.date(y, m, 1)
start_snap = end_allowed - dt.timedelta(days=89)
needed_months = sorted({ add_month(first_of_month(start_snap + dt.timedelta(days=i))) for i in range(90) })
print(f"CAST one-month-ahead months needed: {[str(x) for x in needed_months]}")

# Fetch CAST for those months and index by forecast_date
cast_token = get_token(ACLED_USER, ACLED_PASS)
cast_by_month = {}
for fm in needed_months:
    dfm = fetch_cast_for_month(cast_token, fm.year, fm.month)
    if not dfm.empty:
        cast_by_month[fm] = dfm
    else:
        # will fallback to static CAST if a month is unavailable
        cast_by_month[fm] = pd.DataFrame(columns=["adm1_join","cast_state","forecast_date"])

# ----------------- per-day synthesis -----------------
print(f"\nGenerating 90 daily snapshots...")
rows = []
cast_hist = []
ev_hist = []

for k in range(89, -1, -1):
    t = end_allowed - dt.timedelta(days=k)
    
    # Progress reporting every 10 days
    if k % 10 == 0 or k == 89:
        print(f"  Processing day {90-k}/90: {t} (k={k})")
    
    w30_start = t - dt.timedelta(days=29)
    w90_start = t - dt.timedelta(days=89)
    prev30_start = t - dt.timedelta(days=59)
    prev30_end   = t - dt.timedelta(days=30)

    # slice by date once, then aggregate counts by adm2
    d90  = j[(j["yyyymmdd"]>=w90_start) & (j["yyyymmdd"]<=t)]
    d30  = j[(j["yyyymmdd"]>=w30_start) & (j["yyyymmdd"]<=t)]
    dP30 = j[(j["yyyymmdd"]>=prev30_start) & (j["yyyymmdd"]<=prev30_end)]

    c90  = d90.groupby("adm2_code").size()
    c30  = d30.groupby("adm2_code").size()
    cP30 = dP30.groupby("adm2_code").size()

    df = adm2_index.copy()
    df["events_90v"] = df["adm2_code"].map(c90).fillna(0).astype(int)
    df["events30"]   = df["adm2_code"].map(c30).fillna(0).astype(int)
    df["events_prevv"]= df["adm2_code"].map(cP30).fillna(0).astype(int)

    # attach static features
    df = df.merge(static, on=["adm2_code","adm1_name","adm2_name"], how="left")
    # one-month-ahead CAST for this snapshot
    df["adm1_join"] = normalize_adm1_join_name(df["adm1_name"])
    cast_month = add_month(t.replace(day=1))
    cast_m = cast_by_month.get(cast_month)
    if cast_m is not None and not cast_m.empty:
        df = df.merge(cast_m[["adm1_join","cast_state"]], on="adm1_join", how="left", suffixes=("", "_dyn"))
        df["cast_state_use"] = df["cast_state_dyn"].fillna(df["cast_state"])  # fallback to static if missing
    else:
        df["cast_state_use"] = df["cast_state"]
    den = df["pop_wra"].astype(float).clip(lower=0)

    df["v30"]      = np.where(den>0, 1e5*df["events30"].astype(float)/den, 0.0)
    df["v3m"]      = np.where(den>0, 1e5*df["events_90v"].astype(float)/den, 0.0)
    df["v30_prev"] = np.where(den>0, 1e5*df["events_prevv"].astype(float)/den, 0.0)
    df["dlt_v30_raw"] = df["v30"] - df["v30_prev"]

    # spillover via Queen weights
    v30_vec = df["v30"].to_numpy(dtype=float)
    # W.dot returns a numpy array; convert to a Series aligned to df.index and ensure no NaN
    S = np.asarray(W.dot(v30_vec), dtype=float)
    df["spillover"] = pd.Series(np.nan_to_num(S, nan=0.0), index=df.index)

    # normalize and indices (same logic as pipeline)
    V30 = winsor01(df["v30"])
    V3m = winsor01(df["v3m"])

    # symmetric transform then winsor on dV30
    d = df["dlt_v30_raw"].astype(float)
    p1, p9 = np.nanpercentile(d, 10), np.nanpercentile(d, 90)
    d_clip = np.clip(d, p1, p9)
    d_unit = -1 + 2*(d_clip - p1)/(p9 - p1) if p9 > p1 else np.zeros_like(d_clip)
    dV30 = winsor01(pd.Series(d_unit, index=df.index))

    S_norm = winsor01(df["spillover"])
    # Use dynamic CAST by snapshot_date if available; else fall back to static cast_state.
    # These are already 0–1 in pipeline outputs; avoid double-winsorization.
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

    out = pd.DataFrame({
        "snapshot_date": [t]*len(df),
        "data_as_of": [t]*len(df),
        "adm2_code": df["adm2_code"],
        "adm1_name": df["adm1_name"],
        "adm2_name": df["adm2_name"],
        "DCR100": 100*DCR,
        "PRS100": 100*PRS,
        "priority100": 100*(0.6*PRS + 0.4*DCR),
        "v30": df["v30"], "v3m": df["v3m"], "dlt_v30_raw": df["dlt_v30_raw"], "spillover": df["spillover"]
    })
    rows.append(out)

    # State-level CAST history: one row per state (ADM1) per snapshot date
    # Group by state and compute mean CAST per state
    state_cast = df.groupby("adm1_name", as_index=False).agg({
        "cast_state_use": lambda x: float(pd.to_numeric(x, errors="coerce").fillna(0.0).mean())
    }).rename(columns={"cast_state_use": "cast_state_mean"})
    state_cast["snapshot_date"] = t
    cast_hist.append(state_cast[["snapshot_date", "adm1_name", "cast_state_mean"]])
    
    ev_hist.append(pd.DataFrame({"snapshot_date":[t], "data_as_of":[t],
                                 "events_30d":[int(df["events30"].sum())],
                                 "events_90d":[int(df["events_90v"].sum())],
                                 "events_prev30":[int(df["events_prevv"].sum())]}))

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

# After hist is built (before printing BACKFILL COMPLETE) add anomaly detection:
# Simple anomaly flag: extreme v30 (>400) but events_90v == 0 in recomputed set (should not happen)
anom = hist[(hist["v30"] > 400)]
if not anom.empty:
    anom.to_csv(ANOMALIES_CSV, index=False)
    print(f"\nAnomalies flagged (v30>400): {len(anom)} rows -> {ANOMALIES_CSV}")
else:
    print("\nNo extreme v30 anomalies detected (threshold 400).")
print("=" * 60)