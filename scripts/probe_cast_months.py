"""
Quick CAST probe script.

Fetches ACLED CAST forecasts for Mexico (ADM1) over a configurable range
of months so we can inspect how the API publishes one-month-ahead predictions.

Usage:
    python scripts/probe_cast_months.py [startYYYY-MM] [endYYYY-MM]

Environment:
    ACLED_USER, ACLED_PASS, optional SSL_VERIFY flag (defaults true).
Outputs:
    Writes summarized rows to out/checks/cast_month_probe.csv and prints
    a small preview per month.
"""

from __future__ import annotations

import os
import sys
import pathlib
import datetime as dt
import pandas as pd
import requests
from dotenv import load_dotenv
from unidecode import unidecode

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "out"
CHECK_DIR = OUT_DIR / "checks"
CHECK_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()
ACLED_USER = os.getenv("ACLED_USER")
ACLED_PASS = os.getenv("ACLED_PASS")
SSL_VERIFY = os.getenv("SSL_VERIFY", "true").lower() == "true"

assert ACLED_USER and ACLED_PASS, "Set ACLED_USER and ACLED_PASS in your environment or .env file."

ACLED_TOKEN_URL = "https://acleddata.com/oauth/token"
ACLED_CAST_URL = "https://acleddata.com/api/cast/read"

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


def month_name(num: int) -> str:
    return MONTH_NAMES[num - 1]


def normalize_adm1_join(name: str) -> str:
    if not name:
        return ""
    norm = unidecode(str(name)).strip()
    norm = " ".join(w.capitalize() for w in norm.split())
    aliases = {
        "Distrito Federal": "Ciudad De Mexico",
        "Mexico City": "Ciudad De Mexico",
        "Federal District": "Ciudad De Mexico",
        "Coahuila De Zaragoza": "Coahuila",
        "Michoacan De Ocampo": "Michoacan",
        "Veracruz De Ignacio De La Llave": "Veracruz",
    }
    return aliases.get(norm, norm)


def get_token(user: str, password: str) -> str:
    resp = requests.post(
        ACLED_TOKEN_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        data={
            "username": user,
            "password": password,
            "grant_type": "password",
            "client_id": "acled",
        },
        timeout=60,
        verify=SSL_VERIFY,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_cast_month(token: str, year: int, month_num: int) -> pd.DataFrame:
    """Return raw CAST rows for Mexico for the requested forecast month (all API fields)."""
    frames = []
    page, limit = 1, 5000
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    while True:
        params = {
            "_format": "json",
            "country": "Mexico",
            "year": year,
            "month": month_name(month_num),
            "limit": limit,
            "page": page,
        }
        resp = requests.get(ACLED_CAST_URL, headers=headers, params=params, timeout=60, verify=SSL_VERIFY)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        frames.append(pd.DataFrame(data))
        if len(data) < limit:
            break
        page += 1
    cast_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if cast_raw.empty:
        return cast_raw
    if "admin1" in cast_raw.columns and "adm1_name" not in cast_raw.columns:
        cast_raw = cast_raw.rename(columns={"admin1": "adm1_name"})
    cast_raw["forecast_date"] = pd.Timestamp(year=int(year), month=int(month_num), day=1)
    cast_raw["adm1_join"] = cast_raw.get("adm1_name", pd.Series(dtype=str)).apply(normalize_adm1_join)
    cast_raw["requested_month"] = cast_raw["forecast_date"].dt.date
    return cast_raw


def iter_months(start: dt.date, end: dt.date):
    cur = start.replace(day=1)
    stop = end.replace(day=1)
    while cur <= stop:
        yield cur
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)


def parse_month(arg: str) -> dt.date:
    return dt.datetime.strptime(arg, "%Y-%m").date()


def main():
    today = dt.date.today().replace(day=1)
    default_start = today - dt.timedelta(days=365)
    default_end = today + dt.timedelta(days=120)

    if len(sys.argv) >= 2:
        start = parse_month(sys.argv[1])
    else:
        start = default_start
    if len(sys.argv) >= 3:
        end = parse_month(sys.argv[2])
    else:
        end = default_end

    print(f"Probing CAST months from {start:%Y-%m} to {end:%Y-%m}")
    token = get_token(ACLED_USER, ACLED_PASS)

    rows = []
    raw_frames = []
    for month_date in iter_months(start, end):
        df = fetch_cast_month(token, month_date.year, month_date.month)
        if df.empty:
            print(f"  {month_date:%Y-%m}: no rows returned")
            continue
        raw_frames.append(df)
        lead_col = None
        for candidate in ("lead", "Lead"):
            if candidate in df.columns:
                lead_col = candidate
                break
        if lead_col is None:
            df["lead"] = "NA"
        else:
            df["lead"] = df[lead_col].astype(str)

        month_col = "month" if "month" in df.columns else None
        year_col = "year" if "year" in df.columns else None
        if month_col and year_col:
            df["forecast_label"] = df[month_col].astype(str) + " " + df[year_col].astype(str)
        else:
            df["forecast_label"] = month_date.strftime("%B %Y")

        agg_dict = {"casts": ("total_forecast", "first")}
        if "lead" in df.columns:
            agg_dict["min_lead"] = ("lead", "min")
            agg_dict["max_lead"] = ("lead", "max")

        month_summary = df.groupby("adm1_join").agg(**agg_dict).reset_index()
        month_summary["forecast_date"] = month_date
        rows.append(month_summary)
        sample_cols = [c for c in ["adm1_name", "lead", month_col, year_col, "total_forecast"] if c and c in df.columns]
        sample = df.head(3)[sample_cols] if sample_cols else df.head(3)
        print(f"  {month_date:%Y-%m}: {len(df)} rows")
        print(sample.to_string(index=False))

    if rows:
        out_df = pd.concat(rows, ignore_index=True)
        out_csv = CHECK_DIR / "cast_month_probe_summary.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"\nWrote summary table -> {out_csv}")
    else:
        print("No CAST data retrieved for the requested range.")

    if raw_frames:
        raw_df = pd.concat(raw_frames, ignore_index=True)
        raw_csv = CHECK_DIR / "cast_month_probe_full.csv"
        raw_df.to_csv(raw_csv, index=False)
        print(f"Wrote full raw export ({len(raw_df):,} rows) -> {raw_csv}")


if __name__ == "__main__":
    main()
