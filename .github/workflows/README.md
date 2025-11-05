# GitHub Actions Workflows

## Daily Data Pipeline

This workflow automatically runs the `pipeline.py` script daily at 2:00 AM UTC to update the maternal health risk data.

### Required Repository Secrets

To enable the daily pipeline, you must configure the following secrets in your GitHub repository (Settings → Secrets and variables → Actions):

#### Required Secrets:
- **`ACLED_USER`**: Your ACLED API username/email
- **`ACLED_PASS`**: Your ACLED API password

#### Optional Secrets (for Google Sheets integration):
- **`ENABLE_SHEETS`**: Set to `true` to enable Google Sheets export (default: `false`)
- **`SHEET_NAME`**: Name of the Google Sheet (default: `mx_brigadas_dashboard`)
- **`GOOGLE_CREDS_JSON`**: Full path to Google service account credentials JSON file (if enabling Sheets)

### How to Add Secrets

1. Go to your repository on GitHub
2. Click on **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret with its name and value
5. Click **Add secret**

### Manual Trigger

You can manually trigger the workflow at any time:
1. Go to the **Actions** tab in your repository
2. Select **Daily Data Pipeline** from the left sidebar
3. Click **Run workflow** button
4. Select the branch and click **Run workflow**

### Workflow Behavior

- **Schedule**: Runs automatically every day at 2:00 AM UTC
- **Data refresh**: Forces fresh ACLED data pulls (`ACLED_REFRESH=true`, `CAST_REFRESH=true`)
- **Outputs**: Commits updated CSV files in the `out/` directory back to the repository
- **Google Sheets**: If enabled via secrets, uploads data to the configured Google Sheet

### Troubleshooting

If the workflow fails:
1. Check the **Actions** tab for error logs
2. Verify all required secrets are properly configured
3. Ensure ACLED credentials are valid and have API access
4. Check that the ACLED API is accessible and not rate-limited
