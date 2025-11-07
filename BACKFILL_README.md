# Backfilling 90-Day History

## Purpose

The `backfill_history.py` script creates an initial 90-day history of ADM2-level risk indicators. This provides historical context for trend analysis and dashboards while the daily pipeline progressively replaces these synthetic snapshots with fresh data over 90 days.

## How It Works

1. **Fetches 180 days** of ACLED violent events (to have enough data for rolling windows)
2. **Generates 90 daily snapshots** (from 89 days ago through today), each with:
   - Properly computed 30-day, 90-day, and previous-30-day event windows
   - DCR, PRS, and priority100 scores using the same methodology as `pipeline.py`
   - Spillover calculations via Queen contiguity weights
   - State-level CAST forecasts (if available)
3. **Writes three history files** to `out/history/`:
   - `adm2_risk_daily_history.csv` - Main risk table (90 days × ~2500 municipalities = ~225,000 rows)
   - `cast_state_history.csv` - State-level CAST over time
   - `acled_event_counts_history.csv` - National event counts for trend lines

## Prerequisites

### Must Run First
```bash
# 1. Run pipeline once to generate static inputs and download INEGI data
python pipeline.py
```

This ensures:
- INEGI 2024 municipality shapefile is downloaded to `data/inegi_2024/extracted/`
- Static features (population, facilities, poverty) are cached
- `out/adm2_risk_daily.csv` exists for reference

### Environment
Requires `.env` with:
```
ACLED_USER=your_email@domain
ACLED_PASS=your_password
SSL_VERIFY=true
```

## Usage

```bash
python backfill_history.py
```

### Expected Output
```
============================================================
BACKFILL HISTORY: 90-day bootstrap
============================================================
Using ADM2 shapefile: data\inegi_2024\extracted\...\AGEM_2024.shp
Loading ADM2 polygons and building spatial weights...
Fetching 180 days of ACLED events (for rolling windows)...
ACLED recency cap: 2024-10-22
Will generate 90 snapshots from 2024-07-24 to 2024-10-22
Fetching ACLED events from 2024-04-25 to 2024-10-22...
Fetched 8234 total events
Filtering for violent events and valid coordinates...
  After filtering: 4521 violent events
  With valid coordinates: 4521 events
Performing spatial join to ADM2 polygons...
  Events spatially joined: 4521

Generating 90 daily snapshots...
  Processing day 1/90: 2024-07-24 (k=89)
  Processing day 11/90: 2024-08-03 (k=79)
  ...
  Processing day 90/90: 2024-10-22 (k=0)

============================================================
BACKFILL COMPLETE
============================================================
Generated 225,000 snapshot rows (90 days × 2500 municipalities)

Wrote:
  out\history\adm2_risk_daily_history.csv
    - 225,000 rows
  out\history\cast_state_history.csv
    - 90 rows
  out\history\acled_event_counts_history.csv
    - 90 rows

Date range: 2024-07-24 to 2024-10-22

The daily pipeline will progressively replace these snapshots
with fresh data over the next 90 days.
============================================================
```

## Runtime

- **Duration**: 5-15 minutes depending on network speed and machine
- **Bottlenecks**:
  - ACLED API fetch (180 days of events, paginated)
  - Spatial join (~4500 events × 2500 municipalities)
  - 90 iterations of index calculations

## What Happens Next

Once backfilled:

1. **Daily pipeline** (`pipeline.py`) appends today's snapshot to history files
2. **History files maintain rolling 90-day window**:
   - Older snapshots (>90 days) are automatically dropped
   - New snapshots replace synthetic backfill data
3. **After 90 days**, all synthetic data is replaced with real daily runs

## Validation

Check that outputs look reasonable:

```python
import pandas as pd

# Load history
hist = pd.read_csv("out/history/adm2_risk_daily_history.csv")

# Check date range
print(f"Dates: {hist['snapshot_date'].min()} to {hist['snapshot_date'].max()}")
print(f"Days: {hist['snapshot_date'].nunique()}")

# Check municipalities per day
print(f"Municipalities per day: {len(hist) / hist['snapshot_date'].nunique():.0f}")

# Check score distributions
print("\nScore summaries:")
print(hist[['DCR100', 'PRS100', 'priority100']].describe())

# Check for nulls
print(f"\nNull counts:\n{hist.isnull().sum()}")
```

Expected:
- 90 unique dates
- ~2500 municipalities per day
- Scores in 0-100 range
- Minimal nulls (only for municipalities with zero population or missing data)

## Troubleshooting

### "Missing INEGI municipality shapefile"
**Solution**: Run `pipeline.py` or `pipeline.ipynb` once first to download INEGI 2024 data.

### "No ACLED data returned"
**Cause**: ACLED recency cap limits access (e.g., 12 months for non-commercial accounts).  
**Solution**: Check your account's `data_query_restrictions` in ACLED API response.

### "Missing out/adm2_risk_daily.csv"
**Solution**: Run `pipeline.py` once to generate static feature CSVs.

### Memory errors
**Solution**: Process is memory-intensive (~2GB). Close other applications or run on a machine with more RAM.

## Files Modified

This script **replaces** (not appends) the following files:
- `out/history/adm2_risk_daily_history.csv`
- `out/history/cast_state_history.csv`
- `out/history/acled_event_counts_history.csv`

If you need to re-run backfill, simply execute the script again. Previous history will be overwritten.
