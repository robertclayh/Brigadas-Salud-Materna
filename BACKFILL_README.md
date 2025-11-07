# Backfilling 90-Day History

## Purpose

The `backfill_history.py` script creates an initial 90-day history of ADM2-level risk indicators. This provides historical context for trend analysis and dashboards while the daily pipeline progressively replaces these synthetic snapshots with fresh data over 90 days.

## Quick Start

```bash
# 1. Ensure pipeline has run at least once to generate static inputs
python pipeline.py

# 2. Run backfill to generate 90 days of synthetic history
python backfill_history.py
```

## How It Works

1. **Fetches 180 days** of ACLED violent events (to have enough data for rolling windows)
2. **Fetches one-month-ahead CAST forecasts** for each snapshot date (forecast month = snapshot month + 1)
3. **Performs one-time spatial join** of events to ADM2 polygons (reused for all 90 snapshots)
4. **Generates 90 daily snapshots** (from 89 days ago through today), each with:
   - Properly computed 30-day, 90-day, and previous-30-day event windows
   - DCR, PRS, and priority100 scores using the same methodology and weights as `pipeline.py`
   - Spillover calculations via Queen contiguity weights
   - Snapshot-specific CAST forecasts (one-month-ahead) merged by state
5. **Writes three history files** to `out/history/`:
   - `adm2_risk_daily_history.csv` - Main risk table (90 days × ~2500 municipalities = ~225,000 rows)
   - `cast_state_history.csv` - State-level CAST over time (90 days × ~32 states = ~2,880 rows)
   - `acled_event_counts_history.csv` - National event counts for trend lines (90 rows)

## Prerequisites

### Must Run First
```bash
# 1. Run pipeline once to generate static inputs
python pipeline.py
```

This ensures:
- HDX ADM2 shapefile exists at `data/mex_admbnda_govmex_20210618_SHP/`
- Static features (population, facilities, poverty, baseline CAST) are cached
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
Using ADM2 shapefile: C:\...\data\mex_admbnda_govmex_20210618_SHP\mex_admbnda_adm2_govmex_20210618.shp
Loading ADM2 polygons and building spatial weights...
Fetching 180 days of ACLED events (for rolling windows)...
ACLED recency cap: 2024-11-07
Will generate 90 snapshots from 2024-08-10 to 2024-11-07
Fetching ACLED events from 2024-05-12 to 2024-11-07...
Fetched 7857 total events
Filtering for violent events and valid coordinates...
  After filtering: 3841 violent events
  With valid coordinates: 3841 events
Performing spatial join to ADM2 polygons...
  Events spatially joined: 3841
CAST one-month-ahead months needed: ['2024-09-01', '2024-10-01', '2024-11-01', '2024-12-01']
[Fetches CAST for each forecast month...]

Generating 90 daily snapshots...
  Processing day 1/90: 2024-08-10 (k=89)
  Processing day 11/90: 2024-08-20 (k=79)
  ...
  Processing day 90/90: 2024-11-07 (k=0)

============================================================
BACKFILL COMPLETE
============================================================
Generated 225,000 snapshot rows (90 days × 2500 municipalities)

Wrote:
  out\history\adm2_risk_daily_history.csv
    - 225,000 rows
  out\history\cast_state_history.csv
    - 2,880 rows
  out\history\acled_event_counts_history.csv
    - 90 rows

Date range: 2024-08-10 to 2024-11-07

The daily pipeline will progressively replace these snapshots
with fresh data over the next 90 days.
============================================================
```

## Runtime

- **Duration**: 5-15 minutes depending on network speed and machine
- **Bottlenecks**:
  - ACLED events API fetch (180 days of events, paginated)
  - ACLED CAST API fetch (4 forecast months for one-month-ahead)
  - Spatial join (~3800 events × 2500 municipalities, one-time operation)
  - 90 iterations of indicator calculations and Queen weights spillover

## What Gets Created

### Before Backfill
```
out/
├── adm2_risk_daily.csv          # Today's snapshot (from pipeline.py)
├── adm2_geometry.csv            # Municipality centroids
└── acled_events_violent_90d.csv # Recent events
```

### After Backfill
```
out/
├── adm2_risk_daily.csv          # Today's snapshot (unchanged)
├── adm2_geometry.csv            # Municipality centroids (unchanged)
├── acled_events_violent_90d.csv # Recent events (unchanged)
└── history/
    ├── adm2_risk_daily_history.csv        # 90 days × ~2500 munis = ~225K rows
    ├── cast_state_history.csv             # 90 days × 32 states = ~2,880 rows
    └── acled_event_counts_history.csv     # 90 days of national event counts
```

## What Happens Next

Once backfilled:

1. **Daily pipeline** (`pipeline.py`) appends today's snapshot to history files
2. **History files maintain rolling 90-day window**:
   - Older snapshots (>90 days) are automatically dropped
   - New snapshots replace synthetic backfill data
3. **After 90 days**, all synthetic data is replaced with real daily runs

### Timeline Example

Assume you run backfill on **2024-11-07** (today):

| Day | Date       | Source           | Notes                                    |
|-----|------------|------------------|------------------------------------------|
| 1   | 2024-08-09 | Backfill         | Synthetic (uses ACLED up to recency cap) |
| 2   | 2024-08-10 | Backfill         | Synthetic                                |
| ... | ...        | ...              | ...                                      |
| 89  | 2024-11-06 | Backfill         | Synthetic (yesterday)                    |
| 90  | 2024-11-07 | Backfill         | Synthetic (today, will be replaced)      |
| 91  | 2024-11-08 | Daily pipeline   | Real (replaces 2024-08-09)               |
| 92  | 2024-11-09 | Daily pipeline   | Real (replaces 2024-08-10)               |
| ... | ...        | ...              | ...                                      |
| 179 | 2025-02-04 | Daily pipeline   | Real (all synthetic data now replaced)   |

## Data Quality Notes

### Synthetic Backfill Characteristics
- **Uses actual ACLED events** from the time period (subject to recency cap)
- **Same methodology** as daily pipeline (DCR, PRS, spillover, winsorization, weights)
- **One-month-ahead CAST** for each snapshot date (forecast valid for that snapshot's next month)
- **Accurate for trend analysis** but may differ slightly from what would have been computed on those actual dates due to:
  - ACLED event revisions/corrections made after original publication
  - CAST forecasts fetched retroactively (not the forecasts that were available on those dates)

### When to Re-run Backfill
- If you change the risk model weights
- If you update static inputs (population, facilities, poverty)
- If you want to reset the history window

### When NOT to Re-run Backfill
- After the daily pipeline has started running (you'll lose real daily snapshots)
- If you just want to add one missing day (use pipeline.py for that specific date)

## Validation

You can manually validate the backfilled data using Python:

```python
import pandas as pd

# Load history files
hist = pd.read_csv("out/history/adm2_risk_daily_history.csv")
cast_hist = pd.read_csv("out/history/cast_state_history.csv")
events_hist = pd.read_csv("out/history/acled_event_counts_history.csv")

# Check ADM2 risk history
print(f"ADM2 history: {len(hist):,} rows")
print(f"Date range: {hist['snapshot_date'].min()} to {hist['snapshot_date'].max()}")
print(f"Unique dates: {hist['snapshot_date'].nunique()}")
print(f"Municipalities per day: {len(hist) / hist['snapshot_date'].nunique():.0f}")
print(f"\nScore ranges:")
print(hist[['DCR100', 'PRS100', 'priority100']].describe())

# Check CAST history
print(f"\n\nCAST history: {len(cast_hist):,} rows")
print(f"Unique states: {cast_hist['adm1_name'].nunique()}")
print(f"Unique dates: {cast_hist['snapshot_date'].nunique()}")

# Check events summary
print(f"\n\nEvents history: {len(events_hist):,} rows")
print(events_hist[['events_30d', 'events_90d']].describe())
```

Expected results:
- **ADM2 history**: ~225,000 rows (90 days × ~2,500 municipalities)
- **CAST history**: ~2,880 rows (90 days × 32 states)
- **Events history**: 90 rows (one per day)
- **Score ranges**: DCR100, PRS100, priority100 between 0-100
- **Minimal nulls**: Only in municipalities with zero population

## Troubleshooting

### "Missing ADM2 shapefile"
**Solution**: Ensure HDX shapefile exists at `data/mex_admbnda_govmex_20210618_SHP/mex_admbnda_adm2_govmex_20210618.shp`. Download from [HDX Mexico COD-AB](https://data.humdata.org/dataset/cod-ab-mex) if missing.

### "No ACLED data returned"
**Cause**: ACLED recency cap limits access (e.g., 12 months for non-commercial accounts).  
**Solution**: Check your account's `data_query_restrictions` in ACLED API response.

### "Missing out/adm2_risk_daily.csv"
**Solution**: Run `pipeline.py` once to generate static feature CSVs.

### Memory errors
**Solution**: Process is memory-intensive (~2GB). Close other applications or run on a machine with more RAM.

### Scores look unusual
- Run validation code (see Validation section above) to check distributions
- Compare with today's pipeline.py output (should be similar)
- Check for nulls in key columns (pop_wra, cast_state, etc.)

## Integration with GitHub Actions

If you're using GitHub Actions for daily pipeline runs, backfill should be:

1. **Run manually once** on your local machine or a VM (not in Actions)
2. **Committed** to the repository (the `out/history/` files)
3. **Used by Actions** going forward (Actions will append to existing history)

Do NOT run backfill in GitHub Actions daily—it will overwrite real data.

## Files Modified

This script **replaces** (not appends) the following files:
- `out/history/adm2_risk_daily_history.csv`
- `out/history/cast_state_history.csv`
- `out/history/acled_event_counts_history.csv`

If you need to re-run backfill, simply execute the script again. Previous history will be overwritten.
