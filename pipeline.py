"""
Pipeline to build ADM2-level risk tables for Mexico. 
Daily step pulls ACLED events and CAST forecasts (subject to ACLED recency caps), 
merges static inputs (population, facilities, poverty), computes indicators, and writes CSVs. 
Caching is enabled by default to avoid redundant API calls. 
Google Sheets upload is disabled by default and can be enabled via environment variables. 
See # %% Configuration & environment for toggles.
"""
# ----------------------------------------
# Usage
# ----------------------------------------
# 1) Create a .env file in the project root with:
# ACLED_USER=your_email@domain
# ACLED_PASS=your_password
# SSL_VERIFY=true
# ACLED_REFRESH=false
# CAST_REFRESH=false
# FORCE_REBUILD_POP=false
# ENABLE_SHEETS=false
# SHEET_NAME=mx_brigadas_dashboard
# GOOGLE_CREDS_JSON=/full/path/to/google-creds-XXXX.json
#
# 2) By default the script uses cached ACLED events and CAST forecasts.
#    Set ACLED_REFRESH=true and CAST_REFRESH=true to force API pulls.
#
# 3) Optional Google Sheets publishing is disabled by default.
#    To enable, set ENABLE_SHEETS=true and ensure GOOGLE_CREDS_JSON points
#    to a valid service account JSON file. Share the target Sheet or
#    parent folder with the service account email.
#
# 4) Run:
#    python pipeline.py
# ----------------------------------------

# %%
# Imports
import os
import pathlib
import datetime as dt
from typing import Dict

import requests
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point as ShpPoint
from dotenv import load_dotenv
from unidecode import unidecode

# Spatial weights (required for spillover)
from libpysal.weights import Queen

# Raster utilities for one-time population aggregation
from rasterstats import zonal_stats
import rasterio

# %%
#
# Configuration & environment
# Loads environment, sets up toggles for API refreshes and Sheets upload
load_dotenv()  # expects .env in project root
ACLED_USER = os.getenv("ACLED_USER")
ACLED_PASS = os.getenv("ACLED_PASS")
assert ACLED_USER and ACLED_PASS, "Set ACLED_USER and ACLED_PASS in your .env"

SSL_VERIFY = os.getenv("SSL_VERIFY", "true").lower() == "true"

# Project folders
ROOT = pathlib.Path.cwd()
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Accessibility blend weight (distance vs. inverse-density), 0..1
ACCESS_BLEND_W = float(os.getenv("ACCESS_BLEND_W", "0.5"))
ACCESS_BLEND_W = 0.0 if ACCESS_BLEND_W < 0 else 1.0 if ACCESS_BLEND_W > 1 else ACCESS_BLEND_W

# Population build control
FORCE_REBUILD_POP = os.getenv("FORCE_REBUILD_POP", "false").lower() == "true"

# Toggle daily API refreshes; default is to use cache unless explicitly refreshed
ACLED_REFRESH = os.getenv("ACLED_REFRESH", "false").lower() == "true"
CAST_REFRESH  = os.getenv("CAST_REFRESH",  "false").lower() == "true"

# Optional: bootstrap history by replaying previous N days (disabled by default)
BOOTSTRAP_HISTORY_DAYS = int(os.getenv("BOOTSTRAP_HISTORY_DAYS", "0"))

# Output publishing controls (disabled by default for reproducibility)
ENABLE_SHEETS = os.getenv("ENABLE_SHEETS", "false").lower() == "true"
ENABLE_HISTORY_SHEETS = os.getenv("ENABLE_HISTORY_SHEETS", "true").lower() == "true"
SHEET_NAME = os.getenv("SHEET_NAME", "mx_brigadas_dashboard")
HISTORY_SHEET_NAME = os.getenv("HISTORY_SHEET_NAME", f"{SHEET_NAME}_history")
GOOGLE_CREDS_JSON = pathlib.Path(os.getenv("GOOGLE_CREDS_JSON", "")).expanduser()

# Administrative shapefile rebuild toggle
FORCE_REBUILD_ADM_SHPS = os.getenv("FORCE_REBUILD_ADM_SHPS", "false").lower() == "true"

# Meta/cache files
META_JSON = OUT_DIR / "acled_meta.json"

# Administrative boundaries (with ADM1 attributes)
MX_ADM2_SHP = DATA_DIR / "mex_admbnda_govmex_20210618_SHP" / "mex_admbnda_adm2_govmex_20210618.shp"
assert MX_ADM2_SHP.exists(), f"Missing shapefile: {MX_ADM2_SHP}"
MX_ADM2_DIR = MX_ADM2_SHP.parent
MX_ADM1_DIR = DATA_DIR / "mex_admbnda_adm1_govmex_20210618_SHP"
MX_ADM0_DIR = DATA_DIR / "mex_admbnda_adm0_govmex_20210618_SHP"
MX_ADM1_SHP = MX_ADM1_DIR / "mex_admbnda_adm1_govmex_20210618.shp"
MX_ADM0_SHP = MX_ADM0_DIR / "mex_admbnda_adm0_govmex_20210618.shp"

def read_shapefile(path: pathlib.Path, default_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Read shapefile and ensure CRS is set (fallback to default_crs if missing)."""
    gdf = gpd.read_file(path)
    if gdf.crs is None and default_crs:
        gdf = gdf.set_crs(default_crs)
    return gdf

def ensure_admin_shapefiles(force=False):
    """Create ADM1 and ADM0 shapefiles by dissolving ADM2 geometries when missing."""
    need_adm1 = force or not MX_ADM1_SHP.exists()
    need_adm0 = force or not MX_ADM0_SHP.exists()
    if not (need_adm1 or need_adm0):
        return

    print("Building derived ADM shapefiles (this runs only when outputs are missing or forced).")
    adm2 = read_shapefile(MX_ADM2_SHP)

    if need_adm1:
        adm1 = adm2.dissolve(by="ADM1_PCODE", as_index=False)
        if adm1.crs is None and adm2.crs is not None:
            adm1 = adm1.set_crs(adm2.crs)
        adm1["adm1_code"] = adm1["ADM1_PCODE"]
        adm1["adm1_name"] = adm1["ADM1_ES"]
        adm1["adm0_code"] = adm1["ADM0_PCODE"]
        adm1["adm0_name"] = adm1["ADM0_ES"]
        adm1 = adm1[["adm1_code", "adm1_name", "adm0_code", "adm0_name", "geometry"]].copy()
        MX_ADM1_DIR.mkdir(parents=True, exist_ok=True)
        adm1.to_file(MX_ADM1_SHP)
        print(f"Wrote ADM1 shapefile: {MX_ADM1_SHP} ({len(adm1)} features)")

    if need_adm0:
        adm0 = adm2.dissolve(by="ADM0_PCODE", as_index=False)
        if adm0.crs is None and adm2.crs is not None:
            adm0 = adm0.set_crs(adm2.crs)
        adm0["adm0_code"] = adm0["ADM0_PCODE"]
        adm0["adm0_name"] = adm0["ADM0_ES"]
        adm0 = adm0[["adm0_code", "adm0_name", "geometry"]].copy()
        MX_ADM0_DIR.mkdir(parents=True, exist_ok=True)
        adm0.to_file(MX_ADM0_SHP)
        print(f"Wrote ADM0 shapefile: {MX_ADM0_SHP} ({len(adm0)} feature{'s' if len(adm0) != 1 else ''})")

ensure_admin_shapefiles(force=FORCE_REBUILD_ADM_SHPS)

# ACLED constants
ACLED_TOKEN_URL = "https://acleddata.com/oauth/token"
ACLED_READ_URL = "https://acleddata.com/api/acled/read"
ACLED_CAST_URL = "https://acleddata.com/api/cast/read"
ACLED_FIELDS = "event_id_cnty|event_date|year|disorder_type|event_type|sub_event_type|country|region|iso|admin1|admin2|location|latitude|longitude|source|fatalities|notes"
VIOLENT_TYPES = {"Violence against civilians", "Battles", "Explosions/Remote violence"}

# Static inputs (created once, then reused)
POP_RASTER = DATA_DIR / "worldpop" / "mex_pop_2025_CN_100m_R2025A_v1.tif"
TEMP_RASTER = DATA_DIR / "worldpop" / "mex_pop_2025_clean.tif"
POP_CSV = DATA_DIR / "pop_adm2.csv"

CLUES_XLSX = DATA_DIR / "CLUES" / "ESTABLECIMIENTO_SALUD_202509.xlsx"
FAC_CSV = DATA_DIR / "clues_facility_counts_adm2.csv"
FAC_POINTS_CSV = OUT_DIR / "clues_facilities_filtered.csv"

CONEVAL_XLSX = DATA_DIR / "CONEVAL" / "Concentrado_indicadores_de_pobreza_2020.xlsx"
MVI_CSV = DATA_DIR / "coneval_muni.csv"

CAST_STATE_CSV = DATA_DIR / "cast_state.csv"

# Outputs
FACT_CSV = OUT_DIR / "adm2_risk_daily.csv"
GEOM_CSV = OUT_DIR / "adm2_geometry.csv"
EVENTS_CSV = OUT_DIR / "acled_events_violent_90d.csv"
EVENTS_PREV_CSV = OUT_DIR / "acled_events_prev30.csv"

# %%
import json

# Utility helpers
def winsor01_series(s: pd.Series, lo=0.05, hi=0.95) -> pd.Series:
    """Winsorize a numeric series to [lo, hi] and scale to 0-1; returns 0s if empty."""
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    a, b = np.nanquantile(s, lo), np.nanquantile(s, hi)
    s2 = np.clip(s, a, b)
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s2 - a) / (b - a)

def normalize_admin_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize admin1 names for Mexico to ensure join compatibility across sources."""
    rep = {
        "Michoacan de Ocampo": "Michoacán",
        "Queretaro": "Querétaro",
        "Nuevo Leon": "Nuevo León",
        "San Luis Potosi": "San Luis Potosí",
        "Yucatan": "Yucatán",
        "Mexico": "México",
    }
    d = df.copy()
    if "admin1" in d.columns:
        d["admin1"] = d["admin1"].replace(rep)
    return d

def normalize_adm1_join_name(s: pd.Series) -> pd.Series:
    """Return canonical ADM1 key for joining across sources (CAST - ADM2). Handles accents and aliases."""
    def canon(x: str) -> str:
        if pd.isna(x):
            return ""
        y = unidecode(str(x)).strip()
        # Title-case for consistent tokens
        y = " ".join(w.capitalize() for w in y.split())
        # Mexico-specific aliasing (CAST <-> INEGI)
        rep = {
            "Distrito Federal": "Ciudad De Mexico",
            "Ciudad De Mexico": "Ciudad De Mexico",
            "Estado De Mexico": "Mexico",
            "Mexico": "Mexico",
            "Queretaro De Arteaga": "Queretaro",
            "Queretaro": "Queretaro",
            "Veracruz": "Veracruz De Ignacio De La Llave",
            "Yucatan": "Yucatan",
            "Nuevo leon": "Nuevo Leon",
            "San Luis Potosi": "San Luis Potosi",
            "Michoacan De Ocampo": "Michoacan De Ocampo",
        }
        return rep.get(y, y)
    return s.apply(canon)

def to_points(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert DataFrame rows with latitude/longitude to GeoDataFrame points (EPSG:4326)."""
    if df.empty:
        return gpd.GeoDataFrame(df.copy(), geometry=[], crs="EPSG:4326")
    d = df.copy()
    for col in ("latitude", "longitude"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["latitude", "longitude"]).copy()
    if d.empty:
        return gpd.GeoDataFrame(d, geometry=[], crs="EPSG:4326")
    geom = [ShpPoint(lon, lat) for lon, lat in zip(d["longitude"], d["latitude"])]
    return gpd.GeoDataFrame(d, geometry=geom, crs="EPSG:4326")

def get_acled_token(username: str, password: str, verify: bool = True) -> str:
    """Obtain an OAuth2 access token for ACLED API."""
    resp = requests.post(
        ACLED_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        data={"username": username, "password": password, "grant_type": "password", "client_id": "acled"},
        timeout=60,
        verify=verify,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

def acled_quick_probe(params: Dict, token: str, verify: bool = True) -> Dict:
    """Probe ACLED API for metadata or small sample (limit 1 row) given params and token."""
    q = {"_format": "json", "page": 1, "limit": 1}
    q.update(params)
    r = requests.get(
        ACLED_READ_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params=q,
        timeout=60,
        verify=verify,
    )
    r.raise_for_status()
    return r.json()

def acled_fetch(params: Dict, token: str, *, limit: int = 5000, max_pages: int = 200, verify: bool = True) -> pd.DataFrame:
    """Fetch ACLED API data with paging, returning DataFrame of all rows for given params."""
    frames = []
    for page in range(1, max_pages + 1):
        q = {"_format": "json", "page": page, "limit": limit}
        q.update(params)
        r = requests.get(
            ACLED_READ_URL,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            params=q,
            timeout=120,
            verify=verify,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def resolve_recency_cap(token: str, verify: bool = True) -> dt.date:
    """Return the latest permissible event date from ACLED recency restrictions."""
    js = acled_quick_probe({}, token, verify=verify)
    allowed_end_str = (js.get("data_query_restrictions") or {}).get("date_recency", {}).get("date")
    if not allowed_end_str:
        return dt.date.today()
    try:
        return dt.date.fromisoformat(allowed_end_str)
    except ValueError:
        return dt.date.today()

def anchors_from_end(end_date: dt.date) -> Dict[str, dt.date]:
    """Return anchor dates for 90-day and 30-day windows ending at end_date."""
    return dict(
        TODAY=end_date,
        FROM_30=end_date - dt.timedelta(days=30),
        FROM_90=end_date - dt.timedelta(days=90),
        FROM_60PREV=end_date - dt.timedelta(days=60),
        TO_30PREV=end_date - dt.timedelta(days=30),
    )

def build_params(country: str, iso: int, start: dt.date, end: dt.date) -> Dict[str, Dict]:
    """Return parameter dicts for ACLED API, both by ISO code and country name."""
    iso_params = {"iso": iso, "event_date": f"{start}|{end}", "event_date_where": "BETWEEN", "fields": ACLED_FIELDS}
    ctry_params = {"country": country, "event_date": f"{start}|{end}", "event_date_where": "BETWEEN", "fields": ACLED_FIELDS}
    return {"iso": iso_params, "country": ctry_params}

def choose_live_params(params_pair: Dict[str, Dict], token: str, verify: bool = True) -> Dict:
    """Select the ACLED API params (iso or country) that returns data for this query."""
    p_iso = acled_quick_probe(params_pair["iso"], token, verify=verify)
    if (p_iso.get("total_count") or 0) > 0:
        return params_pair["iso"]
    p_ctry = acled_quick_probe(params_pair["country"], token, verify=verify)
    if (p_ctry.get("total_count") or 0) > 0:
        return params_pair["country"]
    return params_pair["iso"]

def _read_meta(path: pathlib.Path) -> dict:
    """Read pipeline meta/cache file as JSON dict, or return empty if missing or error."""
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _write_meta(path: pathlib.Path, data: dict):
    """Write meta/cache file as JSON; ignore errors."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, default=str, indent=2)
    except Exception:
        pass

# %%
#
# Build population per ADM2 once; reuse cached CSV unless FORCE_REBUILD_POP=true
def build_population_if_needed():
    """Aggregate WorldPop raster to ADM2 polygons; estimate WRA as 25 percent of total; cache to CSV."""
    if POP_CSV.exists() and not FORCE_REBUILD_POP:
        pop = pd.read_csv(POP_CSV)
        for c in ["adm2_code","adm1_name","adm2_name"]:
            if c in pop.columns:
                pop[c] = pop[c].astype(str)
        return pop

    assert POP_RASTER.exists(), f"Missing raster: {POP_RASTER}"
    print("Running population zonal statistics; this may take several minutes.")

    with rasterio.open(POP_RASTER) as src:
        profile = src.profile
        data = src.read(1).astype("float32")
        data[data < 0] = 0
        profile.update(dtype="float32")
        TEMP_RASTER.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(TEMP_RASTER, "w", **profile) as dst:
            dst.write(data, 1)

adm2 = read_shapefile(MX_ADM2_SHP).to_crs(4326)
    adm2 = adm2.rename(columns={"ADM1_ES": "adm1_name", "ADM2_ES": "adm2_name", "ADM2_PCODE": "adm2_code"})
    adm2 = adm2[["adm1_name", "adm2_name", "adm2_code", "geometry"]]

    # Fix invalid geometries and dissolve to single part per ADM2
    adm2["geometry"] = adm2.geometry.buffer(0)
    adm2 = adm2.dissolve(by=["adm1_name","adm2_name","adm2_code"], as_index=False)

    # Compute zonal stats; do NOT treat 0 as nodata (0 is a valid population value)
    zs = zonal_stats(
        adm2,
        TEMP_RASTER,
        stats=["sum", "count"],
        nodata=None,
        geojson_out=False,
        all_touched=True
    )
    adm2["pop_total"] = [float(z.get("sum", 0) or 0) for z in zs]
    adm2["pix_count"] = [int(z.get("count", 0) or 0) for z in zs]

    # Diagnostic: flag ADM2 with zero population but non-trivial land area
    adm2_proj = adm2.to_crs(3857)
    adm2["area_km2"] = adm2_proj.geometry.area.values / 1e6
    zero_pop_flags = adm2[(adm2["pop_total"] <= 0) & (adm2["area_km2"] > 5)]
    if not zero_pop_flags.empty:
        chk_dir = OUT_DIR / "checks"
        chk_dir.mkdir(parents=True, exist_ok=True)
        zero_pop_flags[["adm1_name","adm2_name","adm2_code","area_km2","pix_count"]].to_csv(chk_dir / "pop_zero_area_flags.csv", index=False)

    adm2["pop_total"] = np.where(adm2["pop_total"] < 0, 0, adm2["pop_total"])
    adm2["pop_wra"] = (adm2["pop_total"] * 0.25)
    adm2["pop_total"] = adm2["pop_total"].round(0).astype(int)
    adm2["pop_wra"] = adm2["pop_wra"].round(0).astype(int)

    # Ensure string keys when writing/returning
    adm2["adm2_code"] = adm2["adm2_code"].astype(str)
    adm2["adm1_name"] = adm2["adm1_name"].astype(str)
    adm2["adm2_name"] = adm2["adm2_name"].astype(str)

    pop_out = adm2.drop(columns=["geometry","pix_count","area_km2"], errors="ignore").copy()
    pop_out.to_csv(POP_CSV, index=False)
    print(f"Saved ADM2 population summary -> {POP_CSV}")

    # Optional: print probe rows for specific ADM2 codes if present
    probe_codes = {"MX02001","MX23003"}
    probe = pop_out[pop_out["adm2_code"].isin(probe_codes)]
    if not probe.empty:
        print("Probe population rows:")
        print(probe.to_string(index=False))

    return pop_out

#
# Build facility counts by ADM2 via spatial join; reuse cached CSV if present
def build_clues_if_needed():
    """Read CLUES Excel, filter for public, active, non-mobile, valid locations, spatially join to ADM2, output facility counts per ADM2."""
    if FAC_CSV.exists():
        fac = pd.read_csv(FAC_CSV)
        for c in ["adm2_code","entidad","municipio"]:
            if c in fac.columns:
                fac[c] = fac[c].astype(str)
        # Guard: enforce uniqueness per adm2_code in cached file (defensive)
        if "adm2_code" in fac.columns:
            fac = fac.sort_values("adm2_code").drop_duplicates("adm2_code", keep="first")
        # Rebuild facility points file if missing (for distance metric)
        if not FAC_POINTS_CSV.exists():
            df = pd.read_excel(CLUES_XLSX, sheet_name="CLUES_202509")
            df.columns = (
                df.columns.str.strip().str.lower().str.replace(" ", "_")
                .str.replace("á","a").str.replace("é","e").str.replace("í","i")
                .str.replace("ó","o").str.replace("ú","u").str.replace("ñ","n")
            )
            public_institutions = {"SSA", "IMB", "IMS", "IST", "SDN", "SMP"}
            def s(x): return "" if pd.isna(x) else str(x).upper()
            df["inst"]   = df["clave_de_la_institucion"].map(s)
            df["status"] = df["clave_estatus_de_operacion"].map(s)
            df["nivel"]  = df["clave_nivel_atencion"].map(s)
            for col in ("latitud","longitud"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["latitud","longitud"]).copy()
            filtered = df[
                df["inst"].isin(public_institutions) & (df["status"] == "1") & (df["nivel"] != "6")
            ].copy()
            filtered.rename(columns={"latitud":"lat","longitud":"lon"})[["lon","lat"]].to_csv(FAC_POINTS_CSV, index=False)
        return fac
    # Load CLUES raw
    df = pd.read_excel(CLUES_XLSX, sheet_name="CLUES_202509")
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_")
        .str.replace("á","a").str.replace("é","e").str.replace("í","i")
        .str.replace("ó","o").str.replace("ú","u").str.replace("ñ","n")
    )

    # Filters: public networks, active, non-mobile, valid lat/lon
    public_institutions = {"SSA", "IMB", "IMS", "IST", "SDN", "SMP"}
    def s(x): return "" if pd.isna(x) else str(x).upper()

    df["inst"]   = df["clave_de_la_institucion"].map(s)
    df["status"] = df["clave_estatus_de_operacion"].map(s)   # '1' = active
    df["nivel"]  = df["clave_nivel_atencion"].map(s)         # '6' = mobile

    for col in ("latitud","longitud"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["latitud","longitud"]).copy()

    filtered = df[
        df["inst"].isin(public_institutions) &
        (df["status"] == "1") &
        (df["nivel"] != "6")
    ].copy()
    # Persist filtered facility points for reuse (ensures same filter for both measures)
    filtered.rename(columns={"latitud":"lat","longitud":"lon"})[["lon","lat"]].to_csv(FAC_POINTS_CSV, index=False)
    # Spatially assign ADM2 via point-in-polygon
adm2 = read_shapefile(MX_ADM2_SHP)
    if adm2.crs is None or adm2.crs.to_epsg() != 4326:
        adm2 = adm2.to_crs(4326)
    adm2 = adm2.rename(columns={
        "ADM2_PCODE": "adm2_code",
        "ADM2_ES": "adm2_name",
        "ADM1_ES": "entidad"
    })[["adm2_code","adm2_name","entidad","geometry"]].copy()
    adm2["adm2_code"] = adm2["adm2_code"].astype(str)
    # Dissolve to ensure one row per ADM2 (avoid multi-part duplication)
    adm2["geometry"] = adm2.geometry.buffer(0)
    adm2 = adm2.dissolve(by=["adm2_code","adm2_name","entidad"], as_index=False)

    g = gpd.GeoDataFrame(
        filtered,
        geometry=gpd.points_from_xy(filtered["longitud"], filtered["latitud"]),
        crs=4326
    )

    joined = gpd.sjoin(g, adm2, how="left", predicate="within").drop(columns=["index_right"])
    # Standardize expected column names after sjoin (defensive for any suffixing)
    if "adm2_code_right" in joined.columns and "adm2_code" not in joined.columns:
        joined = joined.rename(columns={"adm2_code_right": "adm2_code"})
    if "adm2_name_right" in joined.columns and "adm2_name" not in joined.columns:
        joined = joined.rename(columns={"adm2_name_right": "adm2_name"})
    if "entidad_right" in joined.columns and "entidad" not in joined.columns:
        joined = joined.rename(columns={"entidad_right": "entidad"})

    # Some points may fall outside polygons; keep them but they won't count into any ADM2
    # Aggregate counts by ADM2 code (robust even if 'entidad' is missing)
    fac_counts = (
        joined.groupby(["adm2_code"], dropna=False)
        .size()
        .reset_index(name="facilities")
    )

    # Attach human-readable names from ADM2 lookup
    fac_counts = (
        fac_counts.merge(adm2.drop(columns="geometry"), on="adm2_code", how="left")
        [["adm2_code", "adm2_name", "entidad", "facilities"]]
    )
    # Enforce uniqueness post merge
    fac_counts = fac_counts.sort_values("adm2_code").drop_duplicates("adm2_code", keep="first")

    # Ensure types
    fac_counts["adm2_code"] = fac_counts["adm2_code"].astype(str)
    fac_counts["adm2_name"] = fac_counts["adm2_name"].astype(str)
    fac_counts["entidad"]   = fac_counts["entidad"].astype(str)

    fac_counts.to_csv(FAC_CSV, index=False)
    print(f"Saved aggregated facility counts (spatially joined) -> {FAC_CSV}")
    return fac_counts

#
# Extract municipal poverty and map to ADM2; reuse cached CSV if present
def build_coneval_if_needed():
    """Extract municipal poverty rates from CONEVAL Excel, map to ADM2, output to CSV."""
    if MVI_CSV.exists():
        mvi = pd.read_csv(MVI_CSV)
        for c in ["adm2_code","entidad","municipio"]:
            if c in mvi.columns:
                mvi[c] = mvi[c].astype(str)
        # ensure MVI_raw exists
        if "MVI_raw" not in mvi.columns and "poverty_rate" in mvi.columns:
            mvi["MVI_raw"] = pd.to_numeric(mvi["poverty_rate"], errors="coerce")
        return mvi

    raw = pd.read_excel(CONEVAL_XLSX, sheet_name="Concentrado municipal", header=None, engine="openpyxl")
    hdr_idx = None
    for i in range(min(50, len(raw))):
        val = str(raw.iloc[i, 1]).strip().lower() if not pd.isna(raw.iloc[i, 1]) else ""
        if val.startswith("clave de entidad"):
            hdr_idx = i
            break
    if hdr_idx is None:
        raise RuntimeError("Could not locate header row in CONEVAL sheet.")

    df = raw.iloc[hdr_idx + 1 :, 1:11].copy()
    df.columns = [
        "clave_ent","entidad","clave_mun_5d","municipio",
        "pobreza_personas_2010","pobreza_pct_2010",
        "pobreza_personas_2015","pobreza_pct_2015",
        "pobreza_personas_2020","pobreza_pct_2020",
    ]
    df = df[~(df["clave_ent"].isna() & df["clave_mun_5d"].isna())].copy()
    def to_str(x): 
        return "" if pd.isna(x) else str(x).strip()
    df["clave_ent"] = df["clave_ent"].apply(to_str).str.zfill(2)
    df["clave_mun_5d"] = df["clave_mun_5d"].apply(to_str).str.zfill(5)
    df = df[df["clave_ent"].str.match(r"^\d{2}$")]
    df = df[df["clave_mun_5d"].str.match(r"^\d{5}$")]
    df["clave_mun_3d"] = df["clave_mun_5d"].str[-3:]
    df["adm2_code"] = "MX" + df["clave_ent"] + df["clave_mun_3d"]

    def parse_pct(x):
        if pd.isna(x): 
            return np.nan
        s = str(x).strip().lower()
        if s in {"n.d.", "nd", "n.d", "na", "n/a", ""}:
            return np.nan
        s = s.replace(",", ".")
        try:
            v = float(s)
        except Exception:
            return np.nan
        return max(0.0, min(100.0, v))

    df["poverty_rate"] = df["pobreza_pct_2020"].apply(parse_pct)
    out = df[["adm2_code", "entidad", "municipio", "poverty_rate"]].drop_duplicates().reset_index(drop=True)
    out["adm2_code"] = out["adm2_code"].astype(str)
    out["entidad"] = out["entidad"].astype(str)
    out["municipio"] = out["municipio"].astype(str)
    out["MVI_raw"] = pd.to_numeric(out["poverty_rate"], errors="coerce")
    out.to_csv(MVI_CSV, index=False, encoding="utf-8")
    print(f"Saved municipal poverty records -> {MVI_CSV}")
    return out

# %%
#
# Daily refresh: ACLED events and ACLED CAST
def cast_fetch_mexico(token: str, *, verify: bool = True, future_only: bool = True) -> pd.DataFrame:
    """
    Download ACLED CAST forecast for Mexico and return one-month-ahead per state.
    Returns DataFrame with 'adm1_name','cast_raw' (for next month), used to derive cast_state.
    """
    headers = {"Authorization": f"Bearer {token}"}
    dfs = []
    page, limit = 1, 5000
    while True:
        params = {"_format": "csv", "country": "Mexico", "limit": limit, "page": page}
        r = requests.get(ACLED_CAST_URL, headers=headers, params=params, timeout=60, verify=verify)
        r.raise_for_status()
        chunk = pd.read_csv(pd.io.common.StringIO(r.text))
        if chunk.empty:
            break
        dfs.append(chunk)
        if len(chunk) < limit:
            break
        page += 1

    cast_raw = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if cast_raw.empty:
        return cast_raw

    cast_raw = cast_raw.rename(columns={"admin1": "adm1_name", "total_forecast": "cast_raw"})
    month_map = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
                 "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}
    cast_raw["month_num"] = cast_raw["month"].map(month_map)
    cast_raw["forecast_date"] = pd.to_datetime(
        dict(year=cast_raw["year"], month=cast_raw["month_num"], day=1), errors="coerce"
    )

    # One-month-ahead selection: choose next month relative to run date anchor
    anchor = dt.date.today().replace(day=1)
    next_month = (anchor.replace(year=anchor.year + (1 if anchor.month == 12 else 0),
                                 month=(1 if anchor.month == 12 else anchor.month + 1)))
    cast_next = cast_raw[cast_raw["forecast_date"] == pd.Timestamp(next_month)]
    if cast_next.empty:
        # fallback: earliest forecast_date >= anchor
        future = cast_raw[cast_raw["forecast_date"] >= pd.Timestamp(anchor)]
        cast_next = future[future["forecast_date"] == future["forecast_date"].min()] if not future.empty else cast_raw

    # Return next-month forecast per state
    cast_state = cast_next.groupby("adm1_name", as_index=False)["cast_raw"].sum()
    cast_state["cast_raw"] = cast_state["cast_raw"].fillna(0).astype(float)
    return cast_state

# %%
#
# Load static inputs (build if missing)
pop = build_population_if_needed()
fac = build_clues_if_needed()
mvi = build_coneval_if_needed()

#
# Geometry once (dissolve to ensure one row per ADM2)
adm2_raw = read_shapefile(MX_ADM2_SHP)
if adm2_raw.crs is None or adm2_raw.crs.to_epsg() != 4326:
    adm2_raw = adm2_raw.to_crs(4326)
ADM2 = adm2_raw.rename(columns={
    "ADM2_PCODE": "adm2_code",
    "ADM2_ES": "adm2_name",
    "ADM1_PCODE": "adm1_code",
    "ADM1_ES": "adm1_name",
}).loc[:, ["adm1_name","adm1_code","adm2_name","adm2_code","geometry"]].copy()
# Fix invalid geometries, dissolve to single multipart per ADM2
ADM2["geometry"] = ADM2.geometry.buffer(0)
ADM2 = ADM2.dissolve(by=["adm2_code","adm2_name","adm1_code","adm1_name"], as_index=False)
# Harmonize key types (avoid object/float mismatches)
ADM2["adm2_code"] = ADM2["adm2_code"].astype(str)
ADM2["adm1_code"] = ADM2["adm1_code"].astype(str)
ADM2["adm1_name"] = ADM2["adm1_name"].astype(str)
ADM2["adm2_name"] = ADM2["adm2_name"].astype(str)

# Canonical ADM1 join key on ADM2 and lookup without geometry
ADM2["adm1_join"] = normalize_adm1_join_name(ADM2["adm1_name"])
ADM2_LU = ADM2.drop(columns="geometry").copy()

# %%
# %%
#
# Determine ACLED recency cap and date windows; prefer cache unless ACLED_REFRESH=true
# Authenticate and compute event windows (with caching to avoid unnecessary API calls)
token = get_acled_token(ACLED_USER, ACLED_PASS, verify=SSL_VERIFY)
end_allowed = resolve_recency_cap(token, verify=SSL_VERIFY)
if end_allowed < dt.date.today():
    print(f"ACLED recency cap in effect; ending at {end_allowed}")
A = anchors_from_end(end_allowed)

# If BOOTSTRAP_HISTORY_DAYS &gt; 0, a separate one-off backfill script can iterate
# anchors_from_end(end_allowed - k) to materialize historical snapshots. Default is 0.

#
# Fetch events for last 90 days and prior 30 days; write cached copies for reuse
# Determine whether to refresh ACLED calls
meta = _read_meta(META_JSON)
last_end_allowed = dt.date.fromisoformat(meta["end_allowed"]) if meta.get("end_allowed") else None
can_use_cache = (
    (EVENTS_CSV.exists() and EVENTS_PREV_CSV.exists()) and
    (last_end_allowed == end_allowed)
)
do_refresh_acled = ACLED_REFRESH and not can_use_cache

if do_refresh_acled:
    # Build parameter sets and fetch fresh data
    p_90   = build_params("Mexico", 484, A["FROM_90"],   A["TODAY"])
    p_prev = build_params("Mexico", 484, A["FROM_60PREV"], A["TO_30PREV"])
    active_90   = choose_live_params(p_90,   token, verify=SSL_VERIFY)
    active_prev = choose_live_params(p_prev, token, verify=SSL_VERIFY)

    acled_90 = acled_fetch(active_90, token, verify=SSL_VERIFY)
    acled_prev = acled_fetch(active_prev, token, verify=SSL_VERIFY)

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        d = df.copy()
        if "country" in d.columns:
            d = d[d["country"] == "Mexico"]
        if "event_id_cnty" in d.columns:
            d = d.drop_duplicates(subset=["event_id_cnty"]).reset_index(drop=True)
        return d

    acled_90 = _clean(acled_90)
    acled_prev = _clean(acled_prev)

    acled_90v = (
        normalize_admin_names(acled_90[acled_90["event_type"].isin(VIOLENT_TYPES)].copy())
        if not acled_90.empty else pd.DataFrame(columns=acled_90.columns)
    )
    acled_prevv = (
        normalize_admin_names(acled_prev[acled_prev["event_type"].isin(VIOLENT_TYPES)].copy())
        if not acled_prev.empty else pd.DataFrame(columns=acled_prev.columns)
    )

    # Points & spatial joins to ADM2
    acled_90v_g = to_points(acled_90v)
    acled_prevv_g = to_points(acled_prevv)

    if not acled_90v_g.empty:
        j_90 = gpd.sjoin(acled_90v_g, ADM2[["adm2_code","adm2_name","geometry"]], how="left", predicate="intersects")
        events_out = j_90.drop(columns=["geometry","index_right"], errors="ignore").copy()
        events_out["data_as_of"] = end_allowed
        events_out["run_date"] = dt.date.today()
        if "adm2_code" in events_out.columns:
            events_out["adm2_code"] = events_out["adm2_code"].astype(str)
        events_out.to_csv(EVENTS_CSV, index=False)
        print(f"Wrote violent events (last 90d) -> {EVENTS_CSV}")
    else:
        events_out = pd.DataFrame()

    if not acled_prevv_g.empty:
        j_prev = gpd.sjoin(acled_prevv_g, ADM2[["adm2_code","adm2_name","geometry"]], how="left", predicate="intersects")
        events_prev_out = j_prev.drop(columns=["geometry","index_right"], errors="ignore").copy()
        events_prev_out["data_as_of"] = end_allowed
        events_prev_out["run_date"] = dt.date.today()
        if "adm2_code" in events_prev_out.columns:
            events_prev_out["adm2_code"] = events_prev_out["adm2_code"].astype(str)
        events_prev_out.to_csv(EVENTS_PREV_CSV, index=False)
    else:
        events_prev_out = pd.DataFrame()

    # Update meta
    meta.update({
        "end_allowed": str(end_allowed),
        "last_run": str(dt.datetime.now())
    })
    _write_meta(META_JSON, meta)

# Use cached ACLED outputs if not refreshing
else:
    print("ACLED refresh skipped (using cached EVENTS CSVs).")
    events_out = pd.read_csv(EVENTS_CSV) if EVENTS_CSV.exists() else pd.DataFrame()
    events_prev_out = pd.read_csv(EVENTS_PREV_CSV) if EVENTS_PREV_CSV.exists() else pd.DataFrame()
    for _df in (events_out, events_prev_out):
        if isinstance(_df, pd.DataFrame) and not _df.empty:
            if "adm2_code" in _df.columns:
                _df["adm2_code"] = _df["adm2_code"].astype(str)
            if "adm2_name" in _df.columns:
                _df["adm2_name"] = _df["adm2_name"].astype(str)
# Build muni_counts from cached or freshly computed outputs
# Aggregate event counts by ADM2 from the ACLED data
def build_muni_counts(events_90v, events_prevv):
    """Build municipality event counts from violent events DataFrames."""
    muni_90 = pd.DataFrame(columns=["adm2_code", "adm2_name", "events_90v"])
    muni_prev = pd.DataFrame(columns=["adm2_code", "adm2_name", "events_prevv"])
    
    if isinstance(events_90v, pd.DataFrame) and not events_90v.empty:
        muni_90 = (
            events_90v.groupby(["adm2_code", "adm2_name"], dropna=False)
            .size()
            .reset_index(name="events_90v")
        )
        for c in ["adm2_code", "adm2_name"]:
            if c in muni_90.columns:
                muni_90[c] = muni_90[c].astype(str)
    
    if isinstance(events_prevv, pd.DataFrame) and not events_prevv.empty:
        muni_prev = (
            events_prevv.groupby(["adm2_code", "adm2_name"], dropna=False)
            .size()
            .reset_index(name="events_prevv")
        )
        for c in ["adm2_code", "adm2_name"]:
            if c in muni_prev.columns:
                muni_prev[c] = muni_prev[c].astype(str)
    
    # Merge the two count DataFrames
    if not muni_90.empty and not muni_prev.empty:
        muni_counts = muni_90.merge(muni_prev, on=["adm2_code", "adm2_name"], how="outer")
    elif not muni_90.empty:
        muni_counts = muni_90.copy()
        muni_counts["events_prevv"] = 0
    elif not muni_prev.empty:
        muni_counts = muni_prev.copy()
        muni_counts["events_90v"] = 0
    else:
        muni_counts = pd.DataFrame(columns=["adm2_code", "adm2_name", "events_90v", "events_prevv"])
    
    # Fill NaN values with 0
    muni_counts = muni_counts.fillna({"events_90v": 0, "events_prevv": 0})
    return muni_counts

# Build muni_counts from the processed events data
muni_counts = build_muni_counts(events_out, events_prev_out)
if isinstance(muni_counts, pd.DataFrame) and not muni_counts.empty:
    # Keep unique per adm2_code
    if "adm2_code" in muni_counts.columns:
        muni_counts = muni_counts.sort_values("adm2_code").drop_duplicates("adm2_code", keep="last")

# ---------------- CAST one‑month‑ahead fetch & cache ----------------
if CAST_STATE_CSV.exists() and not CAST_REFRESH:
    cast_raw = pd.read_csv(CAST_STATE_CSV)
else:
    cast_raw = cast_fetch_mexico(token, verify=SSL_VERIFY)
    cast_raw.to_csv(CAST_STATE_CSV, index=False)

if cast_raw.empty:
    cast = pd.DataFrame(columns=["adm1_join","cast_state"])
else:
    cast_raw["adm1_name"] = cast_raw["adm1_name"].astype(str)
    cast_raw["adm1_join"] = normalize_adm1_join_name(cast_raw["adm1_name"])
    cast_raw["cast_raw"] = pd.to_numeric(cast_raw["cast_raw"], errors="coerce").fillna(0.0)
    cast_raw["cast_state"] = winsor01_series(cast_raw["cast_raw"])
    cast = cast_raw[["adm1_join","cast_state"]].drop_duplicates().copy()
# --------------------------------------------------------------------

# %%
#
# Attach population and compute rates
def _ensure_str(df, cols):
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = d[c].astype(str)
    return d

pop = _ensure_str(pop, ["adm2_code","adm1_name","adm2_name"])
fac = _ensure_str(fac, ["adm2_code"])
mvi = _ensure_str(mvi, ["adm2_code"])
mvi = mvi.sort_values("adm2_code").drop_duplicates("adm2_code", keep="first")

adm2_lite = ADM2_LU.copy()
acled_metrics = (
    adm2_lite
    .merge(pop[["adm2_code","pop_wra"]], on="adm2_code", how="left")
    .merge(muni_counts[["adm2_code","events_90v","events_prevv"]], on="adm2_code", how="left")
    .fillna({"pop_wra": 0, "events_90v": 0, "events_prevv": 0})
)

# 30-day window counts (events30) derived from events_out (works for fresh or cached paths)
def _parse_event_dates(df_events: pd.DataFrame, col: str = "event_date") -> pd.DataFrame:
    if df_events is None or len(df_events) == 0:
        return pd.DataFrame()
    d = df_events.copy()
    if col in d.columns:
        d[col] = pd.to_datetime(d[col], errors="coerce").dt.date
    for k in ["adm2_code", "adm2_name"]:
        if k in d.columns:
            d[k] = d[k].astype(str)
    return d

events_out_parsed = _parse_event_dates(events_out, "event_date") if isinstance(events_out, pd.DataFrame) else pd.DataFrame()
if not events_out_parsed.empty:
    events30_df = (
        events_out_parsed[events_out_parsed["event_date"] > A["FROM_30"]]
        .groupby(["adm2_code"], dropna=False)
        .size()
        .reset_index(name="events30")
    )
    events30_df["adm2_code"] = events30_df["adm2_code"].astype(str)
else:
    events30_df = pd.DataFrame(columns=["adm2_code", "events30"])

acled_metrics = acled_metrics.merge(
    events30_df, on=["adm2_code"], how="left"
)
acled_metrics["events30"] = pd.to_numeric(acled_metrics["events30"], errors="coerce").fillna(0).astype(int)

#
# Defensive denominators: if pop_wra == 0, force rates to 0 instead of dividing by 1
den = acled_metrics["pop_wra"].astype(float)

acled_metrics["v30"] = np.where(
    den > 0, 1e5 * acled_metrics["events30"].astype(float) / den, 0.0
)
acled_metrics["v3m"] = np.where(
    den > 0, 1e5 * acled_metrics["events_90v"].astype(float) / den, 0.0
)
acled_metrics["v30_prev"] = np.where(
    den > 0, 1e5 * acled_metrics["events_prevv"].astype(float) / den, 0.0
)
acled_metrics["dlt_v30_raw"] = acled_metrics["v30"] - acled_metrics["v30_prev"]

# Flag ADM2 with zero population but non-zero event counts (can skew spillovers)
zero_pop_anom = acled_metrics[
    (den <= 0) & (
        (acled_metrics["events30"] > 0) |
        (acled_metrics["events_90v"] > 0) |
        (acled_metrics["events_prevv"] > 0)
    )
][["adm1_name","adm2_name","adm2_code","events30","events_90v","events_prevv"]].copy()

if not zero_pop_anom.empty:
    anom_dir = OUT_DIR / "checks"
    anom_dir.mkdir(parents=True, exist_ok=True)
    zero_pop_anom.to_csv(anom_dir / "zero_population_anomalies.csv", index=False)

acled_metrics = acled_metrics[["adm1_name","adm2_name","adm2_code","pop_wra","v30","v3m","dlt_v30_raw"]].copy()
acled_metrics = acled_metrics.sort_values("adm2_code").drop_duplicates("adm2_code", keep="last")

# %%
#
# Compute spatial spillover as Queen-contiguity average of v30
# Spillover (queen contiguity on ADM2 polygons)
g = ADM2.merge(acled_metrics[["adm2_code","v30"]], on="adm2_code", how="left")
g["v30"] = g["v30"].fillna(0)
# Build Queen contiguity weights; use ADM2 codes as IDs to avoid index warnings
wq = Queen.from_dataframe(g, ids=g["adm2_code"].tolist())
# Row-standardize so the matrix multiplication yields neighbor averages
wq.transform = "R"
S = wq.sparse.dot(g["v30"].to_numpy())
spill = pd.DataFrame({"adm2_code": g["adm2_code"].astype(str), "spillover": S})

# %%
#
# Normalize inputs and compute indices (DCR, PRS, priority) with defined weights
# Access (A), Strain (H), Vulnerability (MVI), CAST (state-level)
adm2_base = (
    ADM2_LU
    .merge(pop[["adm2_code","pop_total","pop_wra"]], on="adm2_code", how="left")
    .merge(fac[["adm2_code","facilities"]], on="adm2_code", how="left")
)

# Ensure required join keys exist; backfill from ADM2 lookup if needed
adm2_lu = ADM2.drop(columns="geometry").copy()
for c in ["adm1_name", "adm2_name", "adm2_code"]:
    if c in adm2_lu.columns:
        adm2_lu[c] = adm2_lu[c].astype(str)

# If adm2_name/adm1_name are missing or contain nulls, reattach from lookup by code
if ("adm2_name" not in adm2_base.columns) or adm2_base["adm2_name"].isna().any() or ("adm1_name" not in adm2_base.columns) or adm2_base["adm1_name"].isna().any():
    cols_to_drop = [c for c in ["adm1_name", "adm2_name"] if c in adm2_base.columns]
    adm2_base = adm2_base.drop(columns=cols_to_drop, errors="ignore").merge(
        adm2_lu[["adm2_code", "adm1_name", "adm2_name"]], on="adm2_code", how="left"
    )

# Normalize types on keys to avoid object/float mismatches
for c in ["adm1_name", "adm2_name", "adm2_code"]:
    if c in adm2_base.columns:
        adm2_base[c] = adm2_base[c].astype(str)

adm2_base = adm2_base.merge(ADM2_LU[["adm2_code","adm1_join"]], on="adm2_code", how="left")

adm2_base["facilities"] = adm2_base.get("facilities", pd.Series(dtype=float)).fillna(0)
adm2_base["pop_wra"]    = adm2_base.get("pop_wra", pd.Series(dtype=float)).fillna(0)
adm2_base["fac_per_100k"] = (1e5 * adm2_base["facilities"] / adm2_base["pop_wra"].clip(lower=1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
inv_fac_density = pd.Series(np.where(adm2_base["fac_per_100k"] > 0, 1.0 / adm2_base["fac_per_100k"], np.nan), index=adm2_base.index)

# Compute distance-to-nearest facility (same CLUES filter), in km, winsorized 5–95
try:
    fac_pts_df = pd.read_csv(FAC_POINTS_CSV)
    fac_pts_df["lon"] = pd.to_numeric(fac_pts_df["lon"], errors="coerce")
    fac_pts_df["lat"] = pd.to_numeric(fac_pts_df["lat"], errors="coerce")
    fac_pts_df = fac_pts_df.dropna(subset=["lon","lat"])
    if not fac_pts_df.empty:
        fac_pts_gdf = gpd.GeoDataFrame(
            fac_pts_df, geometry=gpd.points_from_xy(fac_pts_df["lon"], fac_pts_df["lat"]), crs=4326
        ).to_crs(3857)
        adm2_proj = ADM2.to_crs(3857)
        nearest = gpd.sjoin_nearest(adm2_proj, fac_pts_gdf[["geometry"]], how="left", distance_col="dist_m")
        dist_km = (nearest["dist_m"].astype(float) / 1000.0).replace([np.inf, -np.inf], np.nan)
        A_distance_norm = winsor01_series(dist_km)
        adm2_base = adm2_base.merge(
            pd.DataFrame({"adm2_code": ADM2["adm2_code"].astype(str), "A_distance": A_distance_norm}),
            on="adm2_code", how="left"
        )
    else:
        adm2_base["A_distance"] = 0.0
except Exception:
    # Fallback if facilities or geopandas nearest not available
    adm2_base["A_distance"] = 0.0

# Current density-based access (inverse facilities-per-100k), winsorized
adm2_base["A_density"] = winsor01_series(inv_fac_density)
# Blend distance vs. density with env-configurable weight (default 0.5)
adm2_base["A_blend"] = (ACCESS_BLEND_W * pd.to_numeric(adm2_base["A_distance"], errors="coerce").fillna(0.0) +
                        (1.0 - ACCESS_BLEND_W) * pd.to_numeric(adm2_base["A_density"], errors="coerce").fillna(0.0))

# ...existing code building final...
# Ensure adm1_join exists for CAST merge
if "adm1_join" not in adm2_base.columns:
    adm2_base = adm2_base.merge(ADM2_LU[["adm2_code","adm1_join"]], on="adm2_code", how="left")
final = (
    adm2_base
    .merge(acled_metrics[["adm2_code","v30","v3m","dlt_v30_raw","pop_wra"]], on="adm2_code", how="left")
    .merge(spill, on="adm2_code", how="left")
    .merge(cast, on="adm1_join", how="left")
    .merge(mvi[["adm2_code","MVI_raw"]], on="adm2_code", how="left")
)
final = final.sort_values("adm2_code").drop_duplicates("adm2_code", keep="last")

# Ensure population columns present (guard against upstream loss)
if "pop_wra" not in final.columns or "pop_total" not in final.columns:
    final = final.merge(
        pop[["adm2_code","pop_total","pop_wra"]],
        on="adm2_code",
        how="left"
    )

# If still missing, create zeros to avoid KeyError
if "pop_wra" not in final.columns:
    final["pop_wra"] = 0
if "pop_total" not in final.columns:
    final["pop_total"] = 0

# (Optional debug)
# if "pop_wra" not in final.columns: print("DEBUG: pop_wra missing after merge; filled with 0s. Columns:", final.columns.tolist())

for col in ["v30","v3m","dlt_v30_raw","spillover","cast_state","fac_per_100k","MVI_raw","pop_wra","pop_total"]:
    final[col] = pd.to_numeric(final.get(col, 0), errors="coerce").fillna(0.0)

final["V30"] = winsor01_series(final["v30"])
final["V3m"] = winsor01_series(final["v3m"])

d = final["dlt_v30_raw"].astype(float)
p1, p9 = np.nanpercentile(d, 10), np.nanpercentile(d, 90)
d_clip = np.clip(d, p1, p9)
d_unit = -1 + 2 * (d_clip - p1) / (p9 - p1) if p9 > p1 else np.zeros_like(d_clip)
final["dV30"] = winsor01_series(pd.Series(d_unit, index=final.index))

final["S"]    = winsor01_series(final["spillover"])
final["CAST"] = final["cast_state"]
# final["A"] = winsor01_series(inv_fac_density)
# Use blended accessibility (distance - inverse-density)
final["A"] = pd.to_numeric(final.get("A_blend", 0.0), errors="coerce").fillna(
                 pd.to_numeric(final.get("A_density", 0.0), errors="coerce")
             ).fillna(0.0)
final["MVI"] = winsor01_series(final["MVI_raw"])

med_wra = final["pop_wra"].replace(0, np.nan).median()
w = np.sqrt(final["pop_wra"].replace(0, 1) / (med_wra if med_wra and med_wra > 0 else 1))
final["w_exposure"] = np.clip(w, 0.5, 2.0)
final["w_exposure"] = final["w_exposure"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

# Note: Dropped strain_H (system load) to avoid redundancy with access_A (facility coverage),
# which showed very high correlation. Access_A remains the single proxy for health-system fragility.
# Revised weights (removed H/strain to avoid redundancy with access_A)
w_dcr = {"V3m":0.35, "S":0.15, "A":0.30, "MVI":0.20}
w_prs_cast = {"V30":0.30, "dV30":0.25, "S":0.10, "CAST":0.18, "A":0.12, "MVI":0.05}
w_prs_nocast = {"V30":0.40, "dV30":0.30, "S":0.10, "A":0.12, "MVI":0.08}

def compute_prs_row(r):
    if r.get("CAST", 0) > 0:
        w = w_prs_cast
        return (r["V30"]*w["V30"] + r["dV30"]*w["dV30"] + r["S"]*w["S"] +
                r["CAST"]*w["CAST"] + r["A"]*w["A"] + r["MVI"]*w["MVI"])
    else:
        w = w_prs_nocast
        return (r["V30"]*w["V30"] + r["dV30"]*w["dV30"] + r["S"]*w["S"] +
                r["A"]*w["A"] + r["MVI"]*w["MVI"])

final["DCR"] = (final["V3m"]*w_dcr["V3m"] + final["S"]*w_dcr["S"] +
                final["A"]*w_dcr["A"] + final["MVI"]*w_dcr["MVI"])
final["PRS"] = final.apply(compute_prs_row, axis=1)

final["DCR100"] = 100*final["DCR"]
final["PRS100"] = 100*final["PRS"]
final["priority100"] = 100*(0.6*final["PRS"] + 0.4*final["DCR"])
run_date_today = dt.date.today()
final["run_date"] = run_date_today

for c in ["DCR","PRS","DCR100","PRS100","priority100"]:
    final[c] = pd.to_numeric(final[c], errors="coerce").fillna(0.0)

fact = final[[
    "run_date","adm1_name","adm2_name","adm2_code","pop_total","pop_wra","w_exposure",
    "v30","v3m","dlt_v30_raw","spillover","CAST","A","MVI",
    "DCR100","PRS100","priority100"
]].copy().rename(columns={"CAST":"cast_state","A":"access_A","MVI":"mvi"})

# Final guard: enforce one row per adm2_code in fact
fact = fact.sort_values("adm2_code").drop_duplicates("adm2_code", keep="last")

#
# Add data_as_of column to capture last ACLED date available (end_allowed)
# Note: data_as_of is written to the fact CSV for transparency of ACLED recency.
fact["data_as_of"] = end_allowed

numeric_cols = ["pop_total","pop_wra","w_exposure","v30","v3m","dlt_v30_raw","spillover","cast_state","access_A","mvi","DCR100","PRS100","priority100"]
for c in numeric_cols:
    fact[c] = pd.to_numeric(fact[c], errors="coerce").fillna(0.0)

print("Fact rows:", len(fact))
n_zero_pop = int((final["pop_wra"] <= 0).sum())
print(f"ADM2 with zero pop_wra: {n_zero_pop}")

# %%
#
# Derive ADM2 centroids in projected CRS to avoid distortion; export as lookup
# Geometry lookup (centroids) for BI
# Compute centroids in a projected CRS to avoid distortion, then convert back to lon/lat
adm2_proj = ADM2.to_crs(3857).copy()
centroids_proj = adm2_proj.geometry.centroid
centroids_ll = gpd.GeoSeries(centroids_proj, crs=adm2_proj.crs).to_crs(4326)

geom_lu = pd.DataFrame({
    "adm1_name": ADM2["adm1_name"].astype(str),
    "adm2_name": ADM2["adm2_name"].astype(str),
    "adm2_code": ADM2["adm2_code"].astype(str),
    "lon": centroids_ll.x,
    "lat": centroids_ll.y,
})

# %%
#
# Outputs — CSVs (Tableau-ready)
# Ensure data_as_of is included in the CSV export
fact.to_csv(FACT_CSV, index=False)
geom_lu.to_csv(GEOM_CSV, index=False)
print(f"Written:\n  {FACT_CSV}\n  {GEOM_CSV}")

# -------------------------------
# Rolling 90-day history snapshots
# -------------------------------
HISTORY_DIR = OUT_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

FACT_HISTORY_CSV = HISTORY_DIR / "adm2_risk_daily_history.csv"
CAST_HISTORY_CSV = HISTORY_DIR / "cast_state_history.csv"
EVENTS_SUMMARY_HISTORY_CSV = HISTORY_DIR / "acled_event_counts_history.csv"

def _as_date(x):
    return pd.to_datetime(x, errors="coerce").dt.date

# 1) Append today's ADM2 snapshot and keep only last 90 days
today = run_date_today
min_keep = today - dt.timedelta(days=90)

fact_snap = fact.copy()
fact_snap["snapshot_date"] = today
# keep lean schema for dashboard time series
keep_cols = [
    "snapshot_date", "data_as_of", "adm2_code", "adm1_name", "adm2_name",
    "DCR100", "PRS100", "priority100", "v30", "v3m", "dlt_v30_raw", "spillover"
]
fact_snap = fact_snap[keep_cols]

if FACT_HISTORY_CSV.exists():
    hist = pd.read_csv(FACT_HISTORY_CSV)
    hist["snapshot_date"] = _as_date(hist["snapshot_date"])
    hist = pd.concat([hist, fact_snap], ignore_index=True)
    # de-dup per ADM2 per day (keep last)
    hist = hist.sort_values(["snapshot_date"]).drop_duplicates(["snapshot_date", "adm2_code"], keep="last")
    # keep rolling 90 days
    hist = hist[hist["snapshot_date"] >= min_keep]
else:
    hist = fact_snap

hist.to_csv(FACT_HISTORY_CSV, index=False)

# 2) Append today's CAST state snapshot (for state trend lines)
cast_snap = cast.copy() if isinstance(cast, pd.DataFrame) else pd.DataFrame(columns=["adm1_join","cast_state"])
cast_hist = None
if not cast_snap.empty:
    cast_snap = cast_snap.copy()
    cast_snap["snapshot_date"] = today
    cast_keep = ["snapshot_date", "adm1_join", "cast_state"]
    cast_snap = cast_snap[cast_keep]

    if CAST_HISTORY_CSV.exists():
        cast_hist = pd.read_csv(CAST_HISTORY_CSV)
        cast_hist["snapshot_date"] = _as_date(cast_hist["snapshot_date"])
        cast_hist = pd.concat([cast_hist, cast_snap], ignore_index=True)
        cast_hist = cast_hist.sort_values(["snapshot_date"]).drop_duplicates(["snapshot_date", "adm1_join"], keep="last")
        cast_hist = cast_hist[cast_hist["snapshot_date"] >= min_keep]
    else:
        cast_hist = cast_snap

    cast_hist.to_csv(CAST_HISTORY_CSV, index=False)

# 3) Append a national events summary (for simple time series marks)
#    Uses counts derived earlier; if missing, computes zeroes.
total_events_90 = int(events_out.shape[0]) if isinstance(events_out, pd.DataFrame) else 0
total_events_prev30 = int(events_prev_out.shape[0]) if isinstance(events_prev_out, pd.DataFrame) else 0
total_events_30 = int(events30_df["events30"].sum()) if isinstance(events30_df, pd.DataFrame) and not events30_df.empty else 0

ev_row = pd.DataFrame([{
    "snapshot_date": today,
    "data_as_of": end_allowed,
    "events_30d": total_events_30,
    "events_90d": total_events_90,
    "events_prev30": total_events_prev30
}])

if EVENTS_SUMMARY_HISTORY_CSV.exists():
    ev_hist = pd.read_csv(EVENTS_SUMMARY_HISTORY_CSV)
    ev_hist["snapshot_date"] = _as_date(ev_hist["snapshot_date"])
    ev_hist = pd.concat([ev_hist, ev_row], ignore_index=True)
    ev_hist = ev_hist.sort_values(["snapshot_date"]).drop_duplicates(["snapshot_date"], keep="last")
    ev_hist = ev_hist[ev_hist["snapshot_date"] >= min_keep]
else:
    ev_hist = ev_row

ev_hist.to_csv(EVENTS_SUMMARY_HISTORY_CSV, index=False)
print(f"Updated 90-day history in: {HISTORY_DIR}")

# %%
# Optional Google Sheets export. Requires ENABLE_SHEETS=true and GOOGLE_CREDS_JSON path in environment.
if ENABLE_SHEETS and GOOGLE_CREDS_JSON and GOOGLE_CREDS_JSON.exists():
    print(f"ACLED_REFRESH={ACLED_REFRESH}, CAST_REFRESH={CAST_REFRESH}")
    import gspread
    from gspread_dataframe import set_with_dataframe
    from oauth2client.service_account import ServiceAccountCredentials

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(str(GOOGLE_CREDS_JSON), scope)
    gc = gspread.authorize(creds)

    # Open or create spreadsheet
    try:
        sh = gc.open(SHEET_NAME)
    except gspread.SpreadsheetNotFound:
        sh = gc.create(SHEET_NAME)

    def write_or_replace(sheet, df: pd.DataFrame, ws_name: str):
        try:
            ws = sheet.worksheet(ws_name)
        except gspread.WorksheetNotFound:
            ws = sheet.add_worksheet(title=ws_name, rows="2", cols="2")
        ws.clear()
        set_with_dataframe(ws, df)
        return ws

    # 1) Main fact table
    ws_risk = write_or_replace(sh, fact, "adm2_risk_daily")

    # 2) ACLED violent events (last 90d) — load from CSV if present
    if EVENTS_CSV.exists():
        events_df = pd.read_csv(EVENTS_CSV)
        ws_events = write_or_replace(sh, events_df, "acled_events_90d")
    else:
        ws_events = None

    # 3) Geometry lookup (centroids)
    ws_geom = write_or_replace(sh, geom_lu, "adm2_geometry")

    # Optional: Provenance / sources log
    prov = pd.DataFrame({
        "run_date": [dt.date.today()] * 3,
        "source": ["ACLED", "CONEVAL", "CLUES/Population"],
        "details": ["MX violent events last 90d/prev30", "Municipal poverty", "Facility counts + WRA"],
        "url": ["https://acleddata.com", "https://www.coneval.org.mx", "http://www.dgis.salud.gob.mx"]
    })
    ws_sources = write_or_replace(sh, prov, "sources_log")

    # Reorder worksheets so risk_daily is first, then events, then geometry, then sources
    # Only include worksheets that actually exist
    desired_titles = ["adm2_risk_daily", "acled_events_90d", "adm2_geometry", "sources_log"]
    current = {ws.title: ws for ws in sh.worksheets()}
    ordered = [current[t] for t in desired_titles if t in current]
    if ordered:
        sh.reorder_worksheets(ordered)

    # Summary
    counts = {
        "adm2_risk_daily": len(fact),
        "acled_events_90d": (len(events_df) if EVENTS_CSV.exists() else 0),
        "adm2_geometry": len(geom_lu),
        "sources_log": len(prov),
    }
    print(f"Wrote to Google Sheet: {SHEET_NAME}")
    print("Tabs and row counts:", counts)

    history_tables = [
        ("adm2_risk_daily_history", hist if isinstance(hist, pd.DataFrame) else None),
        ("cast_state_history", cast_hist if isinstance(cast_hist, pd.DataFrame) else None),
        ("acled_event_counts_history", ev_hist if isinstance(ev_hist, pd.DataFrame) else None),
    ]
    history_tables = [(title, df) for title, df in history_tables if df is not None]

    if ENABLE_HISTORY_SHEETS and history_tables:
        sh_history = None
        try:
            sh_history = gc.open(HISTORY_SHEET_NAME)
        except gspread.SpreadsheetNotFound:
            try:
                sh_history = gc.create(HISTORY_SHEET_NAME)
            except gspread.exceptions.APIError as exc:
                print(f"Warning: failed to create history sheet '{HISTORY_SHEET_NAME}': {exc}")
                print("Skipping history sheet upload (likely Drive quota exceeded).")
        except gspread.exceptions.APIError as exc:
            print(f"Warning: failed to open history sheet '{HISTORY_SHEET_NAME}': {exc}")
            print("Skipping history sheet upload.")

        if sh_history is not None:
            hist_counts = {}
            for title, df in history_tables:
                ws = write_or_replace(sh_history, df, title)
                hist_counts[title] = len(df)

            desired_history = [title for title, _ in history_tables]
            current_hist = {ws.title: ws for ws in sh_history.worksheets()}
            ordered_hist = [current_hist[t] for t in desired_history if t in current_hist]
            if ordered_hist:
                sh_history.reorder_worksheets(ordered_hist)

            print(f"Wrote history tables to Google Sheet: {HISTORY_SHEET_NAME}")
            print("History tabs and row counts:", hist_counts)
    elif not ENABLE_HISTORY_SHEETS and history_tables:
        print("History sheet upload disabled via ENABLE_HISTORY_SHEETS=false; skipping.")
elif ENABLE_SHEETS:
    print("Google Sheets export requested but GOOGLE_CREDS_JSON was not found; skipping upload.")
# %%
