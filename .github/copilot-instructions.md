# Brigadas-Salud-Materna Copilot Instructions

## Project Overview

This repository contains a Python data pipeline that generates ADM2-level (municipality) risk tables to support maternal health brigades in Mexico. The pipeline processes multiple data sources to create actionable risk scores that help humanitarian programs allocate resources to municipalities with the greatest maternal health challenges.

## Purpose

The pipeline operationalizes an evidence-based risk prioritization model by:
- Integrating violence data (ACLED), healthcare facility density (CLUES), poverty metrics (CONEVAL), and population data (WorldPop)
- Computing composite risk scores combining structural vulnerabilities and dynamic indicators
- Producing visualization-ready tables for programmatic decision-making

## Project Structure

### Core Files
- `pipeline.py` - Main data pipeline script (single monolithic file)
- `requirements.txt` - Python package dependencies
- `.env` - Environment variables and API credentials (not committed to repo)

### Key Directories
- `data/` - Input data files (COD-AB boundaries, CLUES facilities, CONEVAL poverty, WorldPop rasters)
- `out/` - Output CSV files committed to repo:
  - `adm2_risk_daily.csv` - Main risk table (one row per municipality)
  - `acled_events_violent_90d.csv` - Recent violent events for mapping
  - `adm2_geometry.csv` - Municipality centroids
  - `acled_meta.json` - Metadata for caching logic

### GitHub Configuration
- `.github/workflows/daily-pipeline.yml` - Automated daily execution via GitHub Actions
- `.github/workflows/README.md` - Workflow documentation and secrets setup

## Technology Stack

### Core Dependencies
- **Data Processing**: pandas, numpy
- **Geospatial**: geopandas, shapely, pyproj, fiona, rtree, libpysal
- **Raster Processing**: rasterio, rasterstats (for WorldPop population aggregation)
- **API/External**: requests, python-dotenv
- **Optional**: gspread, gspread-dataframe, oauth2client (for Google Sheets publishing)

### Python Version
- Python 3.x (compatible with pandas 2.2+, numpy 1.26+, geopandas 0.14+)

## Build and Test Procedures

### Setup
```bash
pip install -r requirements.txt
```

### Running the Pipeline
```bash
python pipeline.py
```

**Note**: There are currently no automated tests. The pipeline is validated by:
1. Successful execution producing expected output CSVs
2. Visual inspection of risk scores and distributions
3. Validation that output CSVs match expected schema

### Environment Variables
Create a `.env` file with:
- `ACLED_USER` - ACLED API username (required)
- `ACLED_PASS` - ACLED API password (required)
- `SSL_VERIFY=true` - SSL verification toggle
- `ACLED_REFRESH=false` - Force ACLED events API refresh
- `CAST_REFRESH=false` - Force CAST forecast API refresh
- `FORCE_REBUILD_POP=false` - Rebuild population aggregations
- `ENABLE_SHEETS=false` - Enable Google Sheets publishing
- `SHEET_NAME=mx_brigadas_dashboard` - Target Google Sheet name
- `GOOGLE_CREDS_JSON=/path/to/creds.json` - Service account credentials path

## Data Pipeline Workflow

### Caching Strategy
- ACLED data is cached in `out/` directory with metadata tracking
- Pipeline checks if ACLED recency cap has advanced before making API calls
- Static inputs (population, facilities, poverty) are built once unless forced rebuild
- This minimizes API calls and speeds up daily runs

### Daily Automation
- GitHub Actions runs pipeline daily at 2:00 AM UTC
- Forces fresh ACLED data pulls (`ACLED_REFRESH=true`, `CAST_REFRESH=true`)
- Commits updated output files back to repository
- Requires `ACLED_USER` and `ACLED_PASS` secrets in repository settings

### Output Schema
The main risk table (`adm2_risk_daily.csv`) includes:
- Administrative identifiers (`adm1_name`, `adm2_name`, `adm2_code`)
- Population metrics (`pop_total`, `pop_wra`)
- Violence indicators (`v30`, `v3m`, `dlt_v30_raw`, `spillover`)
- Static factors (`cast_state`, `access_A`, `mvi`)
- Composite scores (`DCR100`, `PRS100`, `priority100`)

## Coding Standards

### Code Style
- The pipeline is currently a single monolithic Python script
- Uses NumPy-style docstrings where present
- Follows PEP 8 conventions generally
- Geospatial operations leverage geopandas/shapely idioms

### Key Patterns
- **Spatial joins**: Uses `gpd.sjoin()` for point-in-polygon operations
- **Queen contiguity**: Uses `libpysal.weights.Queen.from_dataframe()` for spillover calculation
- **Winsorization**: Risk indicators are winsorized (5th-95th percentile) and scaled 0-1
- **Weighted composites**: Final scores combine indicators with evidence-based weights

### Dependencies Management
- Pin major/minor versions in `requirements.txt` (e.g., `pandas~=2.2`)
- Geospatial stack requires system libraries (GDAL, PROJ, GEOS) - handled by GitHub Actions workflow
- Avoid adding new dependencies unless necessary for core functionality

## Important Constraints

### Data Access
- ACLED has a 12-month limit on disaggregated event data
- Pipeline assumes access to recent ACLED data via valid API credentials
- Recency caps may delay availability of most recent events

### Performance
- Population aggregation from WorldPop raster is expensive (minutes)
- Cached outputs speed up subsequent runs
- GitHub Actions runner has memory/time limits

### Security
- Never commit `.env` file or API credentials
- Google service account JSON should be stored as GitHub secret
- ACLED credentials are sensitive and should be protected

## Working with This Repository

### Making Changes
- **pipeline.py**: Main script - changes should preserve caching logic and output schema
- **requirements.txt**: Only add dependencies if absolutely necessary
- **Output files**: Committed to repo for versioning and easy access, but are auto-generated
- **Documentation**: Keep README.md synchronized with any pipeline changes

### Common Tasks
- **Update risk model weights**: Modify weight constants in composite score calculations
- **Add new indicator**: Integrate new data source, compute scaled indicator, update composite formulas
- **Change output schema**: Update column names/additions in final CSV writes
- **Adjust caching**: Modify metadata checks and cache invalidation logic

### Things to Avoid
- Don't break the daily GitHub Actions workflow
- Don't modify output file schemas without updating documentation
- Don't introduce dependencies that require complex system libraries beyond the geo stack
- Don't remove or alter caching logic without understanding performance implications

## References

Key methodological resources cited in the project:
- ACLED API documentation
- PySAL spatial weights tutorial (Queen contiguity)
- CONEVAL municipal poverty data
- WorldPop gridded population datasets
- COD-AB administrative boundaries (HDX)
