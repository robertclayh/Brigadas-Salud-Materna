# backfill_history.py
# One-time bootstrap of a 90-day history using 180 days of ACLED events.
# Requires: .env (ACLED_USER, ACLED_PASS), data ADM2 shapefile, and a current out/adm2_risk_daily.csv
# Outputs: out/history/adm2_risk_daily_history.csv, cast_state_history.csv, acled_event_counts_history.csv

import os, pathlib, datetime as dt, json
import pandas as pd, numpy as np, geopandas as gpd, requests
from shapely.geometry import Point
from dotenv import load_dotenv
from libpysal.weights import Queen

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

# Optional: state-level CAST history by day (if pipeline produced it)
CAST_DAILY_CSV = OUT_DIR / "cast_state_daily.csv"

ACLED_TOKEN_URL = "https://acleddata.com/oauth/token"
ACLED_READ_URL  = "https://acleddata.com/api/acled/read"
ACLED_FIELDS = "event_id_cnty|event_date|event_type|admin1|admin2|latitude|longitude|country"
VIOLENT = {"Violence against civilians", "Battles", "Explosions/Remote violence"}

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

# Weights (same as pipeline.py)
W_DCR = {"V3m":0.35, "S":0.15, "A":0.30, "MVI":0.20}
W_PRS_CAST = {"V30":0.30, "dV30":0.25, "S":0.10, "CAST":0.18, "A":0.12, "MVI":0.05}
W_PRS_NC   = {"V30":0.40, "dV30":0.30, "S":0.10,          "A":0.12, "MVI":0.08}

# ----------------- load static assets -----------------
assert FACT_CSV.exists(), f"Missing {FACT_CSV}; run pipeline.py once first."
static_today = pd.read_csv(FACT_CSV)
for k in ["adm2_code","adm1_name","adm2_name"]:
    static_today[k] = static_today[k].astype(str)

# minimal static features reused across all days
static = static_today[["adm2_code","adm1_name","adm2_name","pop_wra","access_A","mvi","cast_state"]].copy()
static["pop_wra"] = pd.to_numeric(static["pop_wra"], errors="coerce").fillna(0.0)

# Optional: state-level CAST history by day (if pipeline produced it)
cast_daily = None
if CAST_DAILY_CSV.exists():
    _cd = pd.read_csv(CAST_DAILY_CSV)
    # Expect columns: snapshot_date, adm1_name, cast_state
    if "snapshot_date" in _cd.columns:
        _cd["snapshot_date"] = pd.to_datetime(_cd["snapshot_date"], errors="coerce").dt.date
    cast_daily = _cd[["snapshot_date", "adm1_name", "cast_state"]].copy()

# ADM2 + weights matrix
adm2 = gpd.read_file(ADM2_SHP)
if adm2.crs is None or adm2.crs.to_epsg() != 4326:
    adm2 = adm2.to_crs(4326)
adm2 = adm2.rename(columns={"ADM2_PCODE":"adm2_code","ADM2_ES":"adm2_name","ADM1_ES":"adm1_name"})[
    ["adm1_name","adm2_name","adm2_code","geometry"]
].copy()
for k in ["adm2_code","adm1_name","adm2_name"]:
    adm2[k] = adm2[k].astype(str)

# align to static ADM2 universe
adm2 = adm2.merge(static[["adm2_code"]].drop_duplicates(), on="adm2_code", how="inner")
wq = Queen.from_dataframe(adm2, ids=adm2["adm2_code"].tolist())
wq.transform = "R"
W = wq.sparse  # scipy csr

# ----------------- fetch 180d of events -----------------
token = get_token(ACLED_USER, ACLED_PASS)
# probe allowed end date (fallback to today if missing)
probe = requests.get(ACLED_READ_URL, headers={"Authorization": f"Bearer {token}"}, params={"_format":"json","limit":1}, verify=SSL_VERIFY, timeout=30).json()
cap = (probe.get("data_query_restrictions") or {}).get("date_recency", {})
end_allowed = pd.to_datetime(cap.get("date"), errors="coerce").date() if cap else dt.date.today()

start_180 = end_allowed - dt.timedelta(days=179)
params = {
    "country": "Mexico",
    "event_date": f"{start_180}|{end_allowed}",
    "event_date_where": "BETWEEN",
    "fields": ACLED_FIELDS
}
ev = acled_fetch(params, token)
if ev.empty:
    raise SystemExit("No ACLED data returned for last 180 days; cannot backfill.")

# clean + filter violent
ev = ev[ev.get("country","")=="Mexico"].copy()
ev = ev.drop_duplicates(subset=["event_id_cnty"])
ev = ev[ev["event_type"].isin(VIOLENT)].copy()
ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce").dt.date
ev["latitude"] = pd.to_numeric(ev["latitude"], errors="coerce")
ev["longitude"] = pd.to_numeric(ev["longitude"], errors="coerce")
ev = ev.dropna(subset=["latitude","longitude"])
g = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev["longitude"], ev["latitude"]), crs=4326)

# spatial join once to ADM2
j = gpd.sjoin(g, adm2[["adm2_code","geometry"]], how="left", predicate="intersects").drop(columns=["index_right"])
j["adm2_code"] = j["adm2_code"].astype(str)

# to speed up groupby by date
j["yyyymmdd"] = pd.to_datetime(j["event_date"]).dt.date

# ensure a full ADM2 index order
adm2_index = adm2[["adm2_code","adm1_name","adm2_name"]].copy()
idx_map = {code:i for i,code in enumerate(adm2_index["adm2_code"])}

# ----------------- per-day synthesis -----------------
rows = []
cast_hist = []
ev_hist = []

for k in range(89, -1, -1):
    t = end_allowed - dt.timedelta(days=k)
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
    # Use dynamic CAST by snapshot_date if available; else fall back to static cast_state
    if cast_daily is not None:
        cd_today = cast_daily[cast_daily["snapshot_date"] == t][["adm1_name", "cast_state"]]
        df = df.merge(cd_today, on="adm1_name", how="left", suffixes=("", "_dyn"))
        df["cast_state_use"] = df["cast_state_dyn"].fillna(df["cast_state"])
    else:
        df["cast_state_use"] = df["cast_state"]
    CAST_norm = winsor01(pd.to_numeric(df["cast_state_use"], errors="coerce").fillna(0.0))
    A_norm    = winsor01(df["access_A"])
    MVI_norm  = winsor01(df["mvi"])

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

    # simple histories for CAST (using dynamic if available) and national events
    cast_hist.append(pd.DataFrame({
        "snapshot_date": [t],
        "adm1_join": ["NA"],
        "cast_state_mean": [float(pd.to_numeric(df["cast_state_use"], errors="coerce").fillna(0.0).mean())]
    }))
    ev_hist.append(pd.DataFrame({"snapshot_date":[t], "data_as_of":[t],
                                 "events_30d":[int(df["events30"].sum())],
                                 "events_90d":[int(df["events_90v"].sum())],
                                 "events_prev30":[int(df["events_prevv"].sum())]}))

hist = pd.concat(rows, ignore_index=True)
hist = hist.sort_values(["snapshot_date","adm2_code"]).reset_index(drop=True)
hist.to_csv(FACT_HISTORY_CSV, index=False)
pd.concat(cast_hist, ignore_index=True).to_csv(CAST_HISTORY_CSV, index=False)
pd.concat(ev_hist, ignore_index=True).to_csv(EVENTS_SUMMARY_HISTORY_CSV, index=False)

print(f"Wrote:\n  {FACT_HISTORY_CSV}\n  {CAST_HISTORY_CSV}\n  {EVENTS_SUMMARY_HISTORY_CSV}")