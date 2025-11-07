# Brigadas-Salud-Materna — Data Pipeline

## Overview

This repository hosts a reproducible Python data pipeline designed to generate ADM2-level risk tables that support maternal health brigades operating in Mexico under humanitarian programming. The pipeline operationalizes an evidence-based risk prioritization model, integrating multiple data sources to provide actionable insights for targeting interventions where they are most needed. By transforming complex datasets into concise, interpretable risk scores, the pipeline empowers program managers and field teams with timely, data-driven decision support.

The risk model combines structural vulnerabilities and dynamic indicators to capture both current and near-term maternal health risks. This approach aligns with evidence-based programming, ensuring resources are allocated efficiently to municipalities facing the greatest challenges.

### Input Variables and Conceptual Roles

- **Violence (ACLED event counts):** Violence adversely affects maternal health by disrupting access to care and increasing stress and insecurity. Including recent violent event counts captures acute risk factors impacting communities.

- **Access (blended proximity and density):** Accessibility is measured as a blend of two components, both scaled 0–1 (higher = worse): (a) distance to the nearest filtered CLUES facility (proximity), and (b) inverse facilities-per-100k WRA within the ADM2 (per‑capita supply). These are combined as A = w·distance_norm + (1−w)·inverse_density_norm with w set by ACCESS_BLEND_W (default 0.5). This reduces boundary artifacts for small ADM2s while retaining capacity signal.

- **Poverty (CONEVAL municipal poverty rates):** Socioeconomic deprivation is a key determinant of health outcomes. Poverty rates contextualize structural vulnerabilities influencing maternal health risks.

- **Spillover (Neighboring violence rates):** Violence and instability can spread geographically. The spillover metric captures the influence of violence in adjacent municipalities, acknowledging spatial contagion effects.

- **CAST Forecast (ACLED state-level forecast):** The CAST forecast provides predictive insight into likely near-term violence trends, enabling anticipatory program adjustments.

### Weighting Logic

The risk scores are constructed by weighting these inputs to reflect their relative importance based on prior validation and expert consensus. Structural factors like violence trends and access receive higher weights due to their direct impact on maternal health outcomes, while spillover and poverty contribute complementary context. The weighting scheme balances stability (structural risk) and responsiveness (predictive risk) to generate actionable priority scores.

### Where to find the tables

- **Google Sheet:** [Google Sheet](https://docs.google.com/spreadsheets/d/13s_SpJSEw9CbPCmktJBCKbrsy2HiIKVpHZk4yeA8MfU/edit?gid=1483145298#gid=1483145298) (tabs: `adm2_risk_daily`, `acled_events_90d`, `adm2_geometry`, `sources_log`)  
- **CSVs in repo:**  
  - [`out/adm2_risk_daily.csv`](https://github.com/robertclayh/Brigadas-SaludMaterna/blob/main/out/adm2_risk_daily.csv)  
  - [`out/acled_events_violent_90d.csv`](https://github.com/robertclayh/Brigadas-SaludMaterna/blob/main/out/acled_events_violent_90d.csv)  
  - [`out/adm2_geometry.csv`](https://github.com/robertclayh/Brigadas-SaludMaterna/blob/main/out/adm2_geometry.csv)

---

## What the Pipeline Produces (Vizualization-Ready Tables)

### 1) `adm2_risk_daily.csv` — main fact table (one row per ADM2)

| Column        | Meaning                                                                                           |
|:--------------|:-------------------------------------------------------------------------------------------------|
| `run_date`    | Date the pipeline ran.                                                                            |
| `data_as_of`  | Last ACLED date available (recency cap).                                                        |
| `adm1_name`, `adm2_name`, `adm2_code` | State, municipality, INEGI/COD-AB code (e.g., MX25008).                                    |
| `pop_total`, `pop_wra` | Total population & women 15–49 (WRA, proxied @25%)[^2].                                          |
| `v30`, `v3m`  | Events per 100k WRA in last 30/90 days (violent types).                                          |
| `dlt_v30_raw` | v30 minus previous 30-day rate.                                                                  |
| `spillover`   | Queen-contiguity neighbor average of v30[^1].                                                       |
| `cast_state`  | State-level CAST forecast (0–1, winsorized 5–95%).                                              |
| `access_A`    | Blended accessibility (0–1): w·distance-to-nearest + (1−w)·inverse facility density (both winsorized 5–95%). w = ACCESS_BLEND_W (default 0.5). |
| `mvi`         | Municipal poverty (CONEVAL 2020 % pobreza), scaled 0–1.                                         |
| `DCR100`      | 100 × [0.35·V3m + 0.15·S + 0.30·A + 0.20·MVI].                                                 |
| `PRS100`      | 100 × [with CAST: 0.30·V30 + 0.25·dV30 + 0.10·S + 0.18·CAST + 0.12·A + 0.05·MVI; without CAST: 0.40·V30 + 0.30·dV30 + 0.10·S + 0.12·A + 0.08·MVI]. |
| `priority100` | 100 × [0.6·PRS + 0.4·DCR].                                                                      |

**Notes:** V30/V3m/dV30/S/CAST/A/MVI are all winsorized & scaled so higher = worse.  

### Mathematical Formulations

The **Descriptive Composite Risk (DCR100)** is calculated as:

$$
\text{DCR100} = 100 \times \left( 0.35 \times V_{3m} + 0.15 \times S + 0.30 \times A + 0.20 \times MVI \right)
$$

where:  
- $$V_{3m}$$ = 90-day violent event rate per 100k WRA  
- $$S$$ = Spillover (neighbor average violence rate)  
- $$A$$ = Inverse facility density (access)  
- $$MVI$$ = Municipal poverty index

The **Predictive Risk Score (PRS100)** incorporates recent trends and forecasts:

$$
\text{PRS100} = 100 \times \left( 0.30 \times V_{30} + 0.25 \times \Delta V_{30} + 0.10 \times S + 0.18 \times CAST + 0.12 \times A + 0.05 \times MVI \right)
$$

(with CAST forecast available), or:

$$
\text{PRS100} = 100 \times \left( 0.40 \times V_{30} + 0.30 \times \Delta V_{30} + 0.10 \times S + 0.12 \times A + 0.08 \times MVI \right)
$$

(without CAST forecast), where:  
- $$V_{30}$$ = 30-day violent event rate per 100k WRA  
- $$\Delta V_{30}$$ = Change in 30-day violent event rate  
- $$CAST$$ = State-level forecast of violence risk

The final **priority score** balances predictive and descriptive risk:

$$
\text{priority100} = 100 \times (0.6 \times PRS + 0.4 \times DCR)
$$

### Accessibility blend definition

Let A_distance be the winsorized (5–95%) and scaled 0–1 distance from the ADM2 centroid to the nearest filtered CLUES facility (higher = farther/worse), and A_density be the winsorized (5–95%) inverse of facilities per 100k WRA within the ADM2 (higher = fewer per‑capita/worse). The final accessibility term used in the model is:

A = w * A_distance + (1 − w) * A_density, where w = `ACCESS_BLEND_W` (0–1, default 0.5).

The same CLUES filters (public networks, active, non‑mobile, valid coordinates) are applied for both components to ensure consistency.

---

### 2) `acled_events_90d.csv` — event points for map layers

ACLED violent events (last 90 days), with original ACLED fields (`event_id_cnty`, `event_date`, `event_type`, `latitude`, `longitude`, `source`, etc.) plus the joined `adm2_code`/`adm2_name`, and `run_date`/`data_as_of`. Ready for direct plotting.

---

### 3) `adm2_geometry.csv` — centroids helper

Minimal lookup (`adm1_name`, `adm2_name`, `adm2_code`, `lon`, `lat`) for lightweight map layers or tooltips.

---

## How It Refreshes (Automation & Caching)

- **Daily API refresh:**
  - ACLED events (last 90 days and prior 30 days)
  - ACLED CAST (state-level forecasts, scaled)
- **Caching:** Stores meta file (`out/acled_meta.json`) and last pulled CSVs.  
  If the ACLED recency cap hasn’t advanced, the API call is skipped.
- **Static builds:** Population (WorldPop raster), CLUES facilities, CONEVAL poverty (one-time unless missing).

**Important:** ACLED’s public “recency” restriction means recent events may be unavailable.  
The pipeline records `data_as_of` to reflect this and uses cached data if the cap is unchanged.

### Automated Daily Execution (GitHub Actions)

A GitHub Actions workflow is configured to run the pipeline automatically every day at 2:00 AM UTC. The workflow:
- Installs all required dependencies (Python packages and geospatial libraries)
- Runs `pipeline.py` with fresh ACLED data pulls
- Commits updated output files back to the repository

**To enable automated runs:**
1. Configure required repository secrets (see [`.github/workflows/README.md`](.github/workflows/README.md))
   - `ACLED_USER`: Your ACLED API username
   - `ACLED_PASS`: Your ACLED API password
2. Optionally configure Google Sheets integration secrets if needed

The workflow can also be triggered manually from the Actions tab in the repository.

---

## Data Access

ACLED has a 12-month limit on disaggregated event-level data.

The current ETL design, indicator weighting, and modeling framework assume access to recent disaggregated ACLED data to capture acute and near-term violence trends critical for accurate risk prioritization at the ADM2 level.

The pipeline and model could be recalibrated to rely more heavily on longer-term violence trends and aggregated indicators, such as state-level forecasts and structural risk factors. This approach would aim to preserve the model’s predictive integrity and utility for programmatic decision-making.

---

## Running the Pipeline

### Environment

```bash
pip install -r requirements.txt
```

or ensure the following are installed:

```bash
pandas geopandas shapely pyproj rtree libpysal rasterio rasterstats
requests python-dotenv gspread gspread-dataframe oauth2client unidecode
```

### Secrets

Create a `.env` file in the project root with:

```env
ACLED_USER=your_email@domain
ACLED_PASS=your_password
SSL_VERIFY=true
ACLED_REFRESH=false
CAST_REFRESH=false
ACCESS_BLEND_W=0.5
FORCE_REBUILD_POP=false
ENABLE_SHEETS=false
SHEET_NAME=mx_brigadas_dashboard
GOOGLE_CREDS_JSON=/full/path/to/google-creds-XXXX.json
```

**Notes:**
- By default, the pipeline uses cached ACLED events and CAST forecasts.  
  To force API pulls, set `ACLED_REFRESH=true` and `CAST_REFRESH=true`.  
- Google Sheets publishing is disabled by default.  
  To enable it, set `ENABLE_SHEETS=true` and ensure `GOOGLE_CREDS_JSON` points to a valid service account JSON file.  
  Share the target Sheet or folder with the service account email.

### Execute

```bash
python pipeline.py
```

Outputs:
- `out/adm2_risk_daily.csv`
- `out/acled_events_violent_90d.csv`
- `out/adm2_geometry.csv`

Uploaded to Google Sheets tabs (in order):  
`adm2_risk_daily`, `acled_events_90d`, `adm2_geometry`, `sources_log`.

---

## Citations

**Primary Data Sources**

- Armed Conflict Location & Event Data Project (ACLED). (2025). *ACLED dataset*. https://acleddata.com  
  — Events and CAST forecasts accessed via API for Mexico, filtered for violent event types (rolling 30-day and 90-day windows).

- Secretaría de Salud, Dirección General de Información en Salud (DGIS). (n.d.). *Catálogo de Clave Única de Establecimientos de Salud (CLUES)*. http://www.dgis.salud.gob.mx/contenidos/sinais/s_clues.html  
  — Facility locations and attributes used to calculate access indicators through spatial joins.

- Consejo Nacional de Evaluación de la Política de Desarrollo Social (CONEVAL). (n.d.). *Pobreza a nivel municipio 2010–2020*. https://www.coneval.org.mx/Medicion/Paginas/Pobreza-municipio-2010-2020.aspx  
  — Municipal poverty data integrated as a structural vulnerability indicator.

- Instituto Nacional de Estadística y Geografía (INEGI). (n.d.). *Censo de Población y Vivienda*. https://www.inegi.org.mx/programas/ccpv  
  — Population counts used to estimate total population and women of reproductive age (WRA), approximated at 25% of the total.

- WorldPop. (2025). *WorldPop Mexico 2025 (R2025A) population dataset*. https://hub.worldpop.org/geodata/summary?id=74383  
  — 100m resolution raster used to spatially aggregate population by ADM2.

- Humanitarian Data Exchange (HDX). (n.d.). *COD-AB Mexico Administrative Boundaries (ADM2)*. https://data.humdata.org/dataset/cod-ab-mex  
  — Used as the spatial reference for joins, spillover computation, and centroid extraction.

**Methodological Reference**

- Arribas-Bel, D. (n.d.). *A course on Geographic Data Science — Lab E: Spatial Weights (Queen/Rook contiguity and transformations)*. https://darribas.org/gds_course/content/bE/lab_E.html  
  — Reference for constructing Queen-contiguity spatial weight matrices using PySAL.

**Tools and Assistance**

- GitHub Copilot. (n.d.). *GitHub Copilot for Visual Studio Code*. https://github.com/features/copilot  
  — Used for code autocompletion and function scaffolding during pipeline development.

- OpenAI. (2025). *ChatGPT (VS Code Integration)*. https://help.openai.com/en/articles/10128592-how-to-install-the-work-with-apps-visual-studio-code-extension  
  — Used for debugging assistance and formatting within the development environment.

---

## Quick Links

- [`adm2_risk_daily.csv`](https://github.com/robertclayh/Brigadas-SaludMaterna/blob/main/out/adm2_risk_daily.csv)
- [`acled_events_violent_90d.csv`](https://github.com/robertclayh/Brigadas-SaludMaterna/blob/main/out/acled_events_violent_90d.csv)
- [`adm2_geometry.csv`](https://github.com/robertclayh/Brigadas-SaludMaterna/blob/main/out/adm2_geometry.csv)
- Google Sheet: `mx_brigadas_dashboard` [Google Sheet](https://docs.google.com/spreadsheets/d/13s_SpJSEw9CbPCmktJBCKbrsy2HiIKVpHZk4yeA8MfU/edit?gid=1483145298#gid=1483145298) (tabs auto-created by pipeline)

[^1]: *Queen-contiguity* defines neighboring spatial units as those sharing either a border or a vertex. See Arribas-Bel, D. **A course on Geographic Data Science** — *Lab E: Spatial Weights*. Available at: https://darribas.org/gds_course/content/bE/lab_E.html.
[^2]: Based on national population statistics from INEGI (Instituto Nacional de Estadística y Geografía), women of reproductive age (15–49) comprise approximately 25% of Mexico’s total population. This proportion is used here as a proxy for WRA where disaggregated data are unavailable. The pipeline design allows this assumption to be refined later if more spatially discrete WRA data become available to improve model accuracy.

## AI Usage Disclosure

This deliverable was developed with assistance from generative AI tools consistent with SDS program guidelines. GitHub Copilot and the ChatGPT VS Code integration were used to assist with code autocompletion, debugging, and formatting within the development environment. No generative AI tools were used to write descriptive text or interpret readings. All conceptual reasoning, model design, and analytical decisions were made by the author.
