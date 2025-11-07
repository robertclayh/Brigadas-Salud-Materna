# 90-Day History Backfill Workflow

## Quick Start

```bash
# 1. Ensure pipeline has run at least once (to download INEGI data and build static inputs)
python pipeline.py

# 2. Run backfill to generate 90 days of synthetic history
python backfill_history.py

# 3. Validate the backfilled data
python validate_backfill.py
```

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
    ├── cast_state_history.csv             # 90 days of state CAST
    └── acled_event_counts_history.csv     # 90 days of national event counts
```

## How Daily Pipeline Uses History

The `pipeline.py` script (when run daily) will:

1. **Append today's snapshot** to `adm2_risk_daily_history.csv`
2. **Keep only last 90 days** (drops snapshots older than 90 days)
3. **Gradually replace** synthetic backfill data with real daily runs

After 90 days of daily pipeline runs, all synthetic backfill data will be replaced.

## Timeline Example

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
- **Same methodology** as daily pipeline (DCR, PRS, spillover, etc.)
- **Static CAST** (state-level forecasts don't change retroactively)
- **Accurate for trend analysis** but may differ slightly from what would have been computed on those actual dates

### When to Re-run Backfill
- If you change the risk model weights
- If you update static inputs (population, facilities, poverty)
- If you want to reset the history window

### When NOT to Re-run Backfill
- After the daily pipeline has started running (you'll lose real daily snapshots)
- If you just want to add one missing day (use pipeline.py for that specific date)

## Validation Checklist

After running `python validate_backfill.py`, expect:

✓ 90 unique dates  
✓ ~2500 municipalities per day  
✓ DCR100, PRS100, priority100 in 0-100 range  
✓ Minimal nulls (<10% in any column)  
✓ 90 rows in CAST history  
✓ 90 rows in events summary  

## Troubleshooting

### "No ACLED data returned"
- Check your ACLED account's date_recency restriction
- Your account may only allow access to recent 12 months
- Backfill will use whatever data is available within that window

### "Memory error"
- Backfill is memory-intensive (~2-3 GB)
- Close other applications
- Consider running on a machine with more RAM

### Scores look unusual
- Run validation script to check distributions
- Compare with today's pipeline.py output (should be similar)
- Check for nulls in key columns (pop_wra, cast_state, etc.)

## Integration with GitHub Actions

If you're using GitHub Actions for daily pipeline runs, backfill should be:

1. **Run manually once** on your local machine or a VM (not in Actions)
2. **Committed** to the repository (the `out/history/` files)
3. **Used by Actions** going forward (Actions will append to existing history)

Do NOT run backfill in GitHub Actions daily—it will overwrite real data.

## See Also

- `BACKFILL_README.md` - Detailed backfill documentation
- `pipeline.py` - Daily pipeline script
- `README.md` - Main project documentation
