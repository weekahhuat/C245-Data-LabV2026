# C245 Data Lab Streamlit Cloud Version

This package has been converted for GitHub and Streamlit Community Cloud deployment.

## What was changed

- Streamlit app files were moved to the repository root.
- Docker-only files were removed from this deployment package.
- The data folder path was changed from `/data/common` to `common_data`.
- The sample Excel file is included in `common_data/VitaCore_Health.xlsx`.

## GitHub repository structure

Your GitHub repository should contain:

```text
Home.py
requirements.txt
rp_logo.png
utils_shared.py
_sidebar.py
.streamlit/config.toml
common_data/VitaCore_Health.xlsx
pages/
  1_Central_Tendencies.py
  2_Regression.py
  3_Probability_Distributions.py
  4_CI_and_Hypothesis_Testing.py
  5_TTest_ANOVA_Chisquare.py
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload all files and folders from this package into the repository root.
3. Go to Streamlit Community Cloud.
4. Create a new app.
5. Select the repository and branch.
6. Set the main file path to:

```text
Home.py
```

7. Click Deploy.

## Important limitation

Streamlit Community Cloud does not run Docker Compose. It will not start Ollama, n8n, MySQL, phpMyAdmin, or Open WebUI. This deployment is for the Streamlit statistics/data lab app only.

If Ollama is not available, the app may show the AI stack as offline. The statistics tools can still run.
