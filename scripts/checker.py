#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

FILES = {
    "risk_daily": Path("out/adm2_risk_daily.csv"),
    "risk_daily_history": Path("out/history/adm2_risk_daily_history.csv"),
}
TOL = 1e-6  # tighten/loosen if needed


def check_file(label, path):
    if not path.exists():
        print(f"[{label}] missing file: {path}")
        return False

    df = pd.read_csv(path)
    required = {"DCR100", "PRS100", "priority100"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        print(f"[{label}] missing columns: {missing}")
        return False

    expected = 0.6 * df["PRS100"] + 0.4 * df["DCR100"]
    diff = (df["priority100"] - expected).abs()

    # Pull any context columns that exist so reporting works for both daily and history files
    context_cols = [c for c in ("run_date", "snapshot_date", "data_as_of", "adm1_name", "adm2_name", "adm2_code") if c in df.columns]
    cols_to_show = context_cols + ["priority100"]
    bad_mask = diff > TOL
    bad = df.loc[bad_mask, cols_to_show].copy()
    bad["expected"] = expected[bad_mask].values

    if bad.empty:
        print(f"[{label}] priority100 matches 0.6*PRS100 + 0.4*DCR100 (tol={TOL}).")
        return True

    print(f"[{label}] {len(bad)} mismatches (tol={TOL}):")
    print(bad.to_string(index=False, max_rows=20))
    return False


if __name__ == "__main__":
    all_ok = True
    for name, file_path in FILES.items():
        all_ok &= check_file(name, file_path)
    if not all_ok:
        raise SystemExit(1)
