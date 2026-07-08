import os, sys, math
from io import StringIO
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import (
    binom, chi2_contingency, f_oneway, norm, poisson, t,
    ttest_1samp, ttest_ind, ttest_rel, gaussian_kde, linregress,
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_shared import (
    round_dataframe_for_display, get_excel_sheet_names,
    save_uploaded_data_file, delete_data_file, list_data_files,
    load_data_file, parse_number_list, parse_anova_groups,
    parse_contingency_table, z_critical, decision_text,
    p_value_from_test_stat_z, p_value_from_test_stat_t,
    format_probability_result, plot_discrete_distribution,
    plot_normal_distribution, kde_mode_value, get_hist_edges,
)
from _sidebar import _sidebar

st.set_page_config(page_title="Regression", layout="wide")
_sidebar("Regression")

DATA_DIR = os.getenv("COMMON_DATA_DIR", "common_data")
os.makedirs(DATA_DIR, exist_ok=True)

st.markdown("""
<style>
/* Compact Streamlit Cloud layout */
section[data-testid="stSidebar"] {
    width: 220px !important;
    min-width: 220px !important;
}
section[data-testid="stSidebar"] > div:first-child {
    width: 220px !important;
    min-width: 220px !important;
    padding-left: 0.6rem !important;
    padding-right: 0.6rem !important;
}
[data-testid="stSidebarNav"] {display:none !important;}
section[data-testid="stSidebar"] {background:#f8f9fa;}
section[data-testid="stSidebar"] hr {margin: 8px 0 !important;}

/* Main content gets more usable width */
.block-container {
    max-width: 100% !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    padding-top: 1.2rem !important;
}

/* Keep tab labels compact and on one line */
.stTabs [data-baseweb="tab-list"] {
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    white-space: nowrap !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 4px 5px !important;
    min-width: auto !important;
}
.stTabs [data-baseweb="tab"] p {
    white-space: nowrap !important;
    font-size: 12px !important;
    line-height: 1.15 !important;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
    height: 3px;
}

/* Reduce general content font size without breaking widgets */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] ol {
    font-size: 14px !important;
    white-space: normal;
}
label, .stSelectbox label, .stNumberInput label,
.stTextInput label, .stRadio label, .stCheckbox label {
    font-size: 14px !important;
    font-weight: 600 !important;
}
.stSelectbox div[data-baseweb="select"] span,
input[type="number"], input[type="text"], textarea {
    font-size: 14px !important;
}
pre  { white-space: pre-wrap !important; font-size: 13px !important; }
code { white-space: pre-wrap !important; font-size: 13px !important; }
.stDataFrame td { white-space: normal !important; font-size: 13px !important; }
.stDataFrame th { font-size: 13px !important; font-weight: 700 !important; }
h1 { font-size: 1.9rem !important; }
h2 { font-size: 1.6rem !important; }
h3 { font-size: 1.35rem !important; }
h4 { font-size: 1.15rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("Regression")
st.subheader("Regression")
st.caption("Linear regression only")

reg_left, reg_right = st.columns([1, 2])

with reg_left:
    reg_files = list_data_files(DATA_DIR)

    uploaded_reg_file = st.file_uploader(
        "Upload CSV / Excel file for Regression",
        type=["csv", "xlsx", "xls"],
        key="reg_uploaded_data_file"
    )
    st.caption("Tip: Upload a file, preview it, and delete it here when you no longer need it in the shared common-data folder.")

    if uploaded_reg_file is not None:
        try:
            saved_path = save_uploaded_data_file(uploaded_reg_file, DATA_DIR)
            st.success(f"Saved to shared folder: {saved_path}")
            st.cache_data.clear()
            reg_files = list_data_files(DATA_DIR)
        except Exception as e:
            st.error(f"Unable to save uploaded data file: {e}")

    if reg_files:
        reg_selected_name = st.selectbox("Choose a data file", reg_files, key="reg_file")
        reg_selected_path = os.path.join(DATA_DIR, reg_selected_name)
        reg_selected_sheet_name = None
        st.write(f"**Selected file path:** `{reg_selected_path}`")

        reg_action_col1, reg_action_col2 = st.columns([1, 1])

        if reg_action_col1.button("Delete Selected File", use_container_width=True, key="reg_delete_selected_data_file"):
            try:
                deleted_path = delete_data_file(reg_selected_name, DATA_DIR)
                st.success(f"Deleted from shared folder: {deleted_path}")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Unable to delete selected file: {e}")

        if reg_action_col2.button("Refresh File List", use_container_width=True, key="reg_refresh_data_files"):
            st.cache_data.clear()
            st.rerun()

        ext = os.path.splitext(reg_selected_path.lower())[1]
        try:
            if ext in {".xlsx", ".xls"}:
                reg_sheet_names = get_excel_sheet_names(reg_selected_path)
                reg_selected_sheet_name = st.selectbox("Choose worksheet", reg_sheet_names, key="reg_excel_sheet")
                reg_df = load_data_file(reg_selected_path, reg_selected_sheet_name)
                st.write(f"**Worksheet:** `{reg_selected_sheet_name}`")
            else:
                reg_df = load_data_file(reg_selected_path)
                st.write("**Worksheet:** CSV file")

            st.write(f"**Shape:** {reg_df.shape[0]} rows × {reg_df.shape[1]} columns")
            st.dataframe(round_dataframe_for_display(reg_df.head(20)), use_container_width=True)
        except Exception as e:
            reg_df = None
            st.error(f"Unable to read file preview: {e}")
    else:
        reg_df = None
        st.info("Upload a CSV or Excel file, or place one in the shared data folder.")

with reg_right:
    if reg_df is not None:
        numeric_cols = list(reg_df.select_dtypes(include=[np.number]).columns)

        if len(numeric_cols) < 2:
            st.warning("At least two numeric columns are required for linear regression.")
        else:
            x_col = st.selectbox("Select X Axis (Predictor)", numeric_cols, key="reg_x")
            y_options = [c for c in numeric_cols if c != x_col]
            y_col = st.selectbox("Select Y Axis (Response)", y_options, key="reg_y")

            reg_working_df = reg_df[[x_col, y_col]].dropna().copy()

            if reg_working_df.shape[0] < 3:
                st.warning("At least 3 complete observations are required for linear regression.")
            elif reg_working_df[x_col].nunique() < 2:
                st.warning("X must have at least two distinct values.")
            else:
                x = reg_working_df[x_col].astype(float).to_numpy()
                y = reg_working_df[y_col].astype(float).to_numpy()
                n = len(x)

                result = linregress(x, y)
                slope = float(result.slope)
                intercept = float(result.intercept)
                r_value = float(result.rvalue)
                p_value = float(result.pvalue)
                slope_stderr = float(result.stderr)

                x_mean = float(np.mean(x))
                ssx = float(np.sum((x - x_mean) ** 2))
                y_hat = intercept + slope * x
                residuals = y - y_hat
                sse = float(np.sum(residuals ** 2))
                see = math.sqrt(sse / (n - 2)) if n > 2 else float("nan")
                r_squared = float(r_value ** 2)
                intercept_stderr = see * math.sqrt((1 / n) + (x_mean ** 2 / ssx)) if ssx > 0 and n > 2 else float("nan")

                st.markdown("### Linear Regression Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=reg_working_df, x=x_col, y=y_col, ax=ax)
                x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
                y_line = intercept + slope * x_line
                ax.plot(x_line, y_line, color="red")
                ax.set_title(f"Linear Regression: {y_col} vs {x_col}")
                st.pyplot(fig)
                plt.close(fig)

                sign = "+" if intercept >= 0 else "-"
                equation = f"{y_col} = {slope:.2f} × {x_col} {sign} {abs(intercept):.2f}"

                st.markdown("### Regression Results")
                metrics_df = pd.DataFrame([
                    {"Metric": "Regression Equation", "Value": equation},
                    {"Metric": "Goodness of Fit (R²)", "Value": f"{r_squared:.2f}"},
                    {"Metric": "Correlation (r)", "Value": f"{r_value:.2f}"},
                    {"Metric": "Significance of Predictor p-value", "Value": f"{p_value:.2f}"},
                    {"Metric": "Standard Error of Estimate", "Value": f"{see:.2f}"},
                ])
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                st.markdown("### Significance of Predictor")
                note_df = pd.DataFrame([
                    {"p-value": "< 0.05", "Meaning": "X significantly affects Y"},
                    {"p-value": "> 0.05", "Meaning": "No significant relationship"},
                ])
                st.dataframe(note_df, use_container_width=True, hide_index=True)
                sig_note = "X significantly affects Y" if p_value < 0.05 else "No significant relationship"
                st.info(f"Interpretation for this model: {sig_note}")

                st.markdown("### Confidence Interval for Coefficients")
                ci_rows = []
                for conf_level in [0.90, 0.95, 0.99]:
                    alpha = 1 - conf_level
                    tcrit = t.ppf(1 - alpha / 2, df=n - 2)

                    slope_lower = slope - tcrit * slope_stderr
                    slope_upper = slope + tcrit * slope_stderr

                    intercept_lower = intercept - tcrit * intercept_stderr
                    intercept_upper = intercept + tcrit * intercept_stderr

                    ci_rows.append({
                        "Confidence Level": f"{int(conf_level*100)}%",
                        "Slope Lower": slope_lower,
                        "Slope Upper": slope_upper,
                        "Intercept Lower": intercept_lower,
                        "Intercept Upper": intercept_upper,
                    })

                ci_df = pd.DataFrame(ci_rows)
                st.dataframe(round_dataframe_for_display(ci_df, decimals=4), use_container_width=True)


