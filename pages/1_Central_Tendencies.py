import os, sys, math
from typing import Any, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_shared import (
    round_dataframe_for_display, get_excel_sheet_names,
    save_uploaded_data_file, delete_data_file, list_data_files,
    load_data_file, kde_mode_value, get_hist_edges,
)
from _sidebar import _sidebar

st.set_page_config(page_title="Central Tendencies", layout="wide")
_sidebar("Central Tendencies")

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

st.title("Central Tendencies")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt2(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def classical_mode_text(series: pd.Series) -> str:
    counts = series.value_counts(dropna=True)
    if counts.empty:
        return "N/A"
    max_count = counts.max()
    if max_count <= 1:
        return "No clear mode"
    modes = counts[counts == max_count].index.tolist()
    return ", ".join(fmt2(v) for v in modes[:5])


def classical_mode_values(series: pd.Series):
    counts = series.value_counts(dropna=True)
    if counts.empty:
        return []
    max_count = counts.max()
    if max_count <= 1:
        return []
    return counts[counts == max_count].index.tolist()


def summary_for_series(series: pd.Series) -> Dict[str, Any]:
    s = series.dropna().astype(float)
    q1       = s.quantile(0.25)
    q3       = s.quantile(0.75)
    iqr      = q3 - q1
    lower    = q1 - 1.5 * iqr
    upper    = q3 + 1.5 * iqr
    outliers = int(((s < lower) | (s > upper)).sum())
    variance = float(s.var(ddof=1)) if s.shape[0] > 1 else 0.0
    return {
        "n":               int(s.shape[0]),
        "Min":             float(s.min()),
        "Max":             float(s.max()),
        "Range":           float(s.max() - s.min()),
        "Mean":            float(s.mean()),
        "Median":          float(s.median()),
        "Classical Mode":  classical_mode_text(s),
        "Q1":              float(q1),
        "Q3":              float(q3),
        "IQR":             float(iqr),
        "Lower Bound":     float(lower),
        "Upper Bound":     float(upper),
        "Std Dev":         float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        "Variance":        variance,
        "KDE Mode":        kde_mode_value(s),
        "Outliers":        outliers,
    }


def _chart_controls(key_prefix: str):
    chart_cols   = st.columns(3)
    show_boxplot = chart_cols[0].checkbox("Box Plot",           value=True, key=f"{key_prefix}_boxplot")
    show_hist    = chart_cols[1].checkbox("Histogram",          value=True, key=f"{key_prefix}_hist")
    show_kde     = chart_cols[2].checkbox("Distribution (KDE)", value=True, key=f"{key_prefix}_kde")
    bin_mode = st.radio(
        "Bin Setting",
        ["JASP-like Auto (Freedman-Diaconis)", "Manual Bin Width", "Manual Number of Bins"],
        horizontal=True,
        key=f"{key_prefix}_bin_mode",
    )
    return show_boxplot, show_hist, show_kde, bin_mode


def _draw_charts(series: pd.Series, label: str,
                 show_boxplot: bool, show_hist: bool, show_kde: bool,
                 bin_mode: str, manual_bin_width, manual_bin_count):
    """Render Box Plot, Histogram and KDE for a single numeric series."""
    s          = series.dropna().astype(float)
    mean_val   = float(s.mean())
    median_val = float(s.median())
    modes      = classical_mode_values(s)
    kde_mode   = kde_mode_value(s)

    # ── Box Plot ──────────────────────────────────────────────────
    if show_boxplot:
        st.markdown("### Box Plot")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(x=s, ax=ax, color="#4C8BF5", width=0.4,
                    flierprops=dict(marker="o", markerfacecolor="#e63946",
                                    markeredgecolor="#e63946", markersize=6))
        ax.axvline(mean_val,   color="red",   linestyle="--", lw=1.8,
                   label=f"Mean = {mean_val:.2f}")
        ax.axvline(median_val, color="green", linestyle="--", lw=1.8,
                   label=f"Median = {median_val:.2f}")
        for i, m in enumerate(modes[:3]):
            ax.axvline(float(m), color="purple", linestyle=":", lw=1.5,
                       label=f"Mode = {fmt2(m)}")
        ax.set_xlabel(label)
        ax.set_title(f"Box Plot of {label}")
        ax.legend(fontsize=9)
        sns.despine(fig)
        st.pyplot(fig)
        plt.close(fig)

        q1  = float(s.quantile(0.25))
        q3  = float(s.quantile(0.75))
        iqr = q3 - q1
        lb  = q1 - 1.5 * iqr
        ub  = q3 + 1.5 * iqr
        n_out = int(((s < lb) | (s > ub)).sum())
        st.caption(
            f"Box spans Q1 = {q1:.2f} to Q3 = {q3:.2f}  |  "
            f"IQR = {iqr:.2f}  |  "
            f"Whiskers: [{lb:.2f}, {ub:.2f}]  |  "
            f"Outliers (IQR rule): {n_out}"
        )

    # ── Histogram ─────────────────────────────────────────────────
    if show_hist:
        st.markdown("### Histogram")
        edges, bin_width, n_bins = get_hist_edges(
            s, manual_width=manual_bin_width, manual_bins=manual_bin_count
        )
        label_prefix = (
            "Auto FD" if bin_mode == "JASP-like Auto (Freedman-Diaconis)"
            else ("Manual Width" if bin_mode == "Manual Bin Width" else "Manual Bins")
        )
        st.caption(
            f"{label_prefix}: bin width = {bin_width:.2f}, number of bins = {n_bins}.  "
            f"Automatic binning uses the Freedman–Diaconis rule."
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(s, bins=edges, kde=False, ax=ax,
                     color="#4C8BF5", edgecolor="white")
        ax.axvline(mean_val,   color="red",   linestyle="--", lw=1.8,
                   label=f"Mean = {mean_val:.2f}")
        ax.axvline(median_val, color="green", linestyle="--", lw=1.8,
                   label=f"Median = {median_val:.2f}")
        for i, m in enumerate(modes[:3]):
            ax.axvline(float(m), color="purple", linestyle=":", lw=1.5,
                       label=f"Mode = {fmt2(m)}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Histogram of {label}")
        ax.legend(fontsize=9)
        sns.despine(fig)
        st.pyplot(fig)
        plt.close(fig)

    # ── KDE ───────────────────────────────────────────────────────
    if show_kde:
        st.markdown("### Distribution (KDE)")
        if s.nunique() > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.kdeplot(s, fill=True, ax=ax, color="#4C8BF5")
            ax.axvline(mean_val,   color="red",    linestyle="--", lw=1.8,
                       label=f"Mean = {mean_val:.2f}")
            ax.axvline(median_val, color="green",  linestyle="--", lw=1.8,
                       label=f"Median = {median_val:.2f}")
            if kde_mode is not None and pd.notna(kde_mode):
                ax.axvline(kde_mode, color="purple", linestyle=":", lw=1.5,
                           label=f"KDE Mode ≈ {kde_mode:.2f}")
            ax.set_xlabel(label)
            ax.set_ylabel("Density")
            ax.set_title(f"KDE Distribution of {label}")
            ax.legend(fontsize=9)
            sns.despine(fig)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "KDE (Kernel Density Estimate) shows the smoothed shape of the distribution.  "
                "The KDE Mode is the peak of the curve — it may differ from the Classical Mode "
                "when data is continuous."
            )
        else:
            st.info("KDE requires at least two distinct numeric values.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TAB STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
ct_main_tab1, ct_main_tab2 = st.tabs([
    "Central Tendency Computation",
    "IQR Calculator",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — CENTRAL TENDENCY COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════
with ct_main_tab1:

    mode_file, mode_manual = st.tabs([
        "📁  File Upload",
        "⌨️  Manual Key-In",
    ])

    # ─────────────────────────────────────────────────────────────
    # MODE 1 — FILE UPLOAD  (existing logic, unchanged)
    # ─────────────────────────────────────────────────────────────
    with mode_file:
        ct_left, ct_right = st.columns([1, 2])

        with ct_left:
            ct_files = list_data_files(DATA_DIR)

            uploaded_ct_file = st.file_uploader(
                "Upload CSV / Excel file for Central Tendencies",
                type=["csv", "xlsx", "xls"],
                key="ct_uploaded_data_file",
            )
            st.caption(
                "Tip: Upload a file, preview it, and delete it here "
                "when you no longer need it in the shared common-data folder."
            )

            if uploaded_ct_file is not None:
                upload_key = f"ct_saved_{uploaded_ct_file.name}_{uploaded_ct_file.size}"
                if upload_key not in st.session_state:
                    try:
                        saved_path = save_uploaded_data_file(uploaded_ct_file, DATA_DIR)
                        st.success(f"Saved to shared folder: {saved_path}")
                        st.cache_data.clear()
                        st.session_state[upload_key] = True
                        ct_files = list_data_files(DATA_DIR)
                    except Exception as e:
                        st.error(f"Unable to save uploaded data file: {e}")

            if ct_files:
                ct_selected_name = st.selectbox(
                    "Choose a data file", ct_files, key="ct_file"
                )
                ct_selected_path = os.path.join(DATA_DIR, ct_selected_name)
                ct_selected_sheet_name = None
                st.write(f"**Selected file path:** `{ct_selected_path}`")

                ct_action_col1, ct_action_col2 = st.columns([1, 1])

                if ct_action_col1.button(
                    "Delete Selected File",
                    use_container_width=True,
                    key="ct_delete_selected_data_file",
                ):
                    try:
                        delete_data_file(ct_selected_name, DATA_DIR)
                        st.cache_data.clear()
                        for k in ["ct_file", "ct_excel_sheet", "ct_uploaded_data_file"]:
                            st.session_state.pop(k, None)
                        for k in [k for k in st.session_state if k.startswith("ct_saved_")]:
                            del st.session_state[k]
                        st.rerun()
                    except Exception as e:
                        st.error(f"Unable to delete selected file: {e}")

                if ct_action_col2.button(
                    "Refresh File List",
                    use_container_width=True,
                    key="ct_refresh_data_files",
                ):
                    st.cache_data.clear()
                    st.rerun()

                ext = os.path.splitext(ct_selected_path.lower())[1]
                try:
                    if ext in {".xlsx", ".xls"}:
                        ct_sheet_names = get_excel_sheet_names(ct_selected_path)
                        ct_selected_sheet_name = st.selectbox(
                            "Choose worksheet", ct_sheet_names, key="ct_excel_sheet"
                        )
                        ct_df = load_data_file(ct_selected_path, ct_selected_sheet_name)
                        st.write(f"**Worksheet:** `{ct_selected_sheet_name}`")
                    else:
                        ct_df = load_data_file(ct_selected_path)
                        st.write("**Worksheet:** CSV file")

                    st.write(f"**Shape:** {ct_df.shape[0]} rows × {ct_df.shape[1]} columns")
                    st.dataframe(
                        round_dataframe_for_display(ct_df.head(20)),
                        use_container_width=True,
                    )
                except Exception as e:
                    ct_df = None
                    st.error(f"Unable to read file preview: {e}")
            else:
                ct_df = None
                st.info("Upload a CSV or Excel file, or place one in the shared data folder.")

        with ct_right:
            if ct_df is not None:
                all_cols     = list(ct_df.columns)
                numeric_cols = list(ct_df.select_dtypes(include=[np.number]).columns)

                if not numeric_cols:
                    st.warning("No numeric columns found in this dataset.")
                else:
                    dimension_options = ["(None - overall only)"] + all_cols
                    dimension = st.selectbox(
                        "Dimension (Group By)", dimension_options, key="ct_dimension"
                    )
                    measure = st.selectbox(
                        "Measure (Numeric)", numeric_cols, key="ct_measure"
                    )

                    show_boxplot, show_hist, show_kde, bin_mode = _chart_controls("ct_file")

                    manual_bin_width = None
                    manual_bin_count = None
                    if bin_mode == "Manual Bin Width":
                        s_hint     = ct_df[measure].dropna().astype(float)
                        data_range = float(s_hint.max() - s_hint.min()) if not s_hint.empty else 1.0
                        auto_hint  = data_range / 10 if data_range > 0 else 1.0
                        manual_bin_width = st.number_input(
                            "Bin Width",
                            min_value=0.0001,
                            value=float(round(auto_hint, 2)),
                            step=float(max(round(auto_hint / 10, 2), 0.1)),
                            key="ct_file_manual_bin_width",
                        )
                    elif bin_mode == "Manual Number of Bins":
                        manual_bin_count = st.number_input(
                            "Number of Bins", min_value=1, value=20, step=1,
                            key="ct_file_manual_bin_count",
                        )

                    selected_dimension = (
                        None if dimension == "(None - overall only)" else dimension
                    )
                    working_df = ct_df[
                        [measure] + ([selected_dimension] if selected_dimension else [])
                    ].copy()
                    working_df = working_df.dropna(
                        subset=[measure] + ([selected_dimension] if selected_dimension else [])
                    )

                    # ── Statistics table ──────────────────────────
                    if selected_dimension:
                        summary_rows = []
                        for group_name, group_df in working_df.groupby(
                            selected_dimension, dropna=False
                        ):
                            row = summary_for_series(group_df[measure])
                            row[selected_dimension] = str(group_name)
                            summary_rows.append(row)
                        summary_df = pd.DataFrame(summary_rows)
                        cols = [selected_dimension, "n", "Min", "Max", "Range",
                                "Mean", "Median", "Classical Mode",
                                "Q1", "Q3", "IQR", "Lower Bound", "Upper Bound",
                                "Std Dev", "Variance", "KDE Mode", "Outliers"]
                        summary_df = summary_df[cols]
                    else:
                        overall    = summary_for_series(working_df[measure])
                        summary_df = pd.DataFrame([{"Group": "Overall", **overall}])
                        cols = ["Group", "n", "Min", "Max", "Range",
                                "Mean", "Median", "Classical Mode",
                                "Q1", "Q3", "IQR", "Lower Bound", "Upper Bound",
                                "Std Dev", "Variance", "KDE Mode", "Outliers"]
                        summary_df = summary_df[cols]

                    st.markdown("### Central Tendency Statistics")
                    st.dataframe(
                        round_dataframe_for_display(summary_df),
                        use_container_width=True,
                    )

                    # ── Outlier records ───────────────────────────
                    show_outliers = st.checkbox(
                        "Show Outlier Records (IQR Rule)",
                        value=False,
                        key="ct_show_outliers",
                    )
                    if show_outliers:
                        st.markdown("### Outlier Records")
                        if selected_dimension:
                            outlier_frames = []
                            for _, grp in working_df.groupby(selected_dimension, dropna=False):
                                sv    = grp[measure].astype(float)
                                q1_g  = sv.quantile(0.25)
                                q3_g  = sv.quantile(0.75)
                                iqr_g = q3_g - q1_g
                                lb_g  = q1_g - 1.5 * iqr_g
                                ub_g  = q3_g + 1.5 * iqr_g
                                out_g = grp[(sv < lb_g) | (sv > ub_g)].copy()
                                if not out_g.empty:
                                    out_g["Lower Bound"] = lb_g
                                    out_g["Upper Bound"] = ub_g
                                    outlier_frames.append(out_g)
                            if outlier_frames:
                                st.dataframe(
                                    round_dataframe_for_display(
                                        pd.concat(outlier_frames, ignore_index=True)
                                    ),
                                    use_container_width=True,
                                )
                            else:
                                st.info("No outliers found using the IQR rule.")
                        else:
                            sv    = working_df[measure].astype(float)
                            q1_g  = sv.quantile(0.25)
                            q3_g  = sv.quantile(0.75)
                            iqr_g = q3_g - q1_g
                            lb_g  = q1_g - 1.5 * iqr_g
                            ub_g  = q3_g + 1.5 * iqr_g
                            out_df = working_df[(sv < lb_g) | (sv > ub_g)].copy()
                            if not out_df.empty:
                                out_df["Lower Bound"] = lb_g
                                out_df["Upper Bound"] = ub_g
                                st.dataframe(
                                    round_dataframe_for_display(out_df),
                                    use_container_width=True,
                                )
                            else:
                                st.info("No outliers found using the IQR rule.")

                    # ── Charts ────────────────────────────────────
                    if selected_dimension:
                        if show_boxplot:
                            st.markdown("### Box Plot")
                            fig, ax = plt.subplots(figsize=(10, 5))
                            sns.boxplot(
                                data=working_df, x=selected_dimension,
                                y=measure, ax=ax,
                            )
                            ax.set_xlabel(selected_dimension)
                            ax.set_ylabel(measure)
                            ax.set_title(f"Box Plot of {measure} by {selected_dimension}")
                            plt.xticks(rotation=45, ha="right")
                            sns.despine(fig)
                            st.pyplot(fig)
                            plt.close(fig)

                        if show_hist:
                            st.markdown("### Histogram")
                            hist_mode = st.radio(
                                "Histogram Style",
                                ["Stacked by Dimension",
                                 "Separate Histogram per Dimension"],
                                horizontal=True,
                                key="ct_hist_mode",
                            )
                            if hist_mode == "Stacked by Dimension":
                                edges, bw, nb = get_hist_edges(
                                    working_df[measure],
                                    manual_width=manual_bin_width,
                                    manual_bins=manual_bin_count,
                                )
                                st.caption(f"bin width = {bw:.2f}, bins = {nb}")
                                fig, ax = plt.subplots(figsize=(10, 5))
                                sns.histplot(
                                    data=working_df, x=measure,
                                    hue=selected_dimension, bins=edges,
                                    multiple="stack", ax=ax,
                                )
                                ax.set_title(
                                    f"Stacked Histogram of {measure} by {selected_dimension}"
                                )
                                sns.despine(fig)
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                for grp_name, grp_df in working_df.groupby(selected_dimension):
                                    edges, bw, nb = get_hist_edges(
                                        grp_df[measure],
                                        manual_width=manual_bin_width,
                                        manual_bins=manual_bin_count,
                                    )
                                    st.caption(f"{grp_name}: bin width = {bw:.2f}, bins = {nb}")
                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    sns.histplot(grp_df[measure], bins=edges, kde=False, ax=ax)
                                    ax.set_title(
                                        f"Histogram of {measure} "
                                        f"({selected_dimension} = {grp_name})"
                                    )
                                    sns.despine(fig)
                                    st.pyplot(fig)
                                    plt.close(fig)

                        if show_kde:
                            st.markdown("### Distribution (KDE)")
                            if working_df[measure].dropna().nunique() > 1:
                                fig, ax = plt.subplots(figsize=(10, 5))
                                sns.kdeplot(
                                    data=working_df, x=measure,
                                    hue=selected_dimension, fill=True, ax=ax,
                                )
                                ax.set_title(
                                    f"KDE Distribution of {measure} by {selected_dimension}"
                                )
                                sns.despine(fig)
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.info("KDE requires at least two distinct numeric values.")
                    else:
                        _draw_charts(
                            working_df[measure], measure,
                            show_boxplot, show_hist, show_kde,
                            bin_mode, manual_bin_width, manual_bin_count,
                        )

    # ─────────────────────────────────────────────────────────────
    # MODE 2 — MANUAL KEY-IN
    # ─────────────────────────────────────────────────────────────
    with mode_manual:
        mk_left, mk_right = st.columns([1, 2])

        with mk_left:
            st.markdown("#### Enter Your Data")
            st.caption(
                "Type or paste values separated by commas, spaces, or one per line."
            )
            raw_text = st.text_area(
                "Data values",
                value="2, 4, 4, 4, 5, 5, 7, 9",
                height=180,
                key="mk_raw_text",
                placeholder="e.g.  12, 15, 14, 16, 18, 13, 17",
            )
            data_label = st.text_input(
                "Variable name (for chart labels)",
                value="My Variable",
                key="mk_label",
            )
            run = st.button(
                "▶  Compute Statistics & Charts",
                key="mk_run",
                type="primary",
                use_container_width=True,
            )

            if run:
                try:
                    cleaned = (
                        raw_text.strip()
                        .replace("\n", ",")
                        .replace(";", ",")
                        .replace(" ", ",")
                    )
                    parts  = [p.strip() for p in cleaned.split(",") if p.strip()]
                    values = [float(p) for p in parts]
                    if len(values) < 2:
                        st.error("Please enter at least 2 values.")
                        st.stop()
                    st.session_state["mk_series"]      = values
                    st.session_state["mk_label_store"] = data_label
                except ValueError:
                    st.error(
                        "Some values could not be parsed. "
                        "Please check your input — only numbers are accepted."
                    )
                    st.stop()

        with mk_right:
            if "mk_series" in st.session_state:
                values = st.session_state["mk_series"]
                lbl    = st.session_state.get("mk_label_store", "My Variable")
                series = pd.Series(values, dtype=float)
                stats  = summary_for_series(series)

                # ── Statistics table — same rows as Excel Helper ──
                st.markdown("### Central Tendency Statistics")
                stat_rows = [
                    ("Count (n)",                    stats["n"]),
                    ("Minimum",                      round(stats["Min"],    4)),
                    ("Maximum",                      round(stats["Max"],    4)),
                    ("Range",                        round(stats["Range"],  4)),
                    ("Mean",                         round(stats["Mean"],   4)),
                    ("Median",                       round(stats["Median"], 4)),
                    ("Mode (Classical)",             stats["Classical Mode"]),
                    ("Q1 (25th Percentile)",         round(stats["Q1"],     4)),
                    ("Q3 (75th Percentile)",         round(stats["Q3"],     4)),
                    ("IQR  (Q3 − Q1)",               round(stats["IQR"],   4)),
                    ("Lower Bound  (Q1 − 1.5×IQR)", round(stats["Lower Bound"], 4)),
                    ("Upper Bound  (Q3 + 1.5×IQR)", round(stats["Upper Bound"], 4)),
                    ("Std Dev (Sample)",             round(stats["Std Dev"],   4)),
                    ("Variance (Sample)",            round(stats["Variance"],  4)),
                    ("KDE Mode (continuous est.)",
                     round(stats["KDE Mode"], 4)
                     if stats["KDE Mode"] is not None and pd.notna(stats["KDE Mode"])
                     else "N/A"),
                    ("Outliers (IQR rule)",          stats["Outliers"]),
                ]
                stat_df = pd.DataFrame(stat_rows, columns=["Statistic", "Value"])
                st.dataframe(stat_df, use_container_width=True, hide_index=True)

                st.markdown("---")

                # ── Chart controls — same style as File Upload ────
                st.markdown("#### Chart Options")
                show_boxplot, show_hist, show_kde, bin_mode = _chart_controls("mk")

                manual_bin_width = None
                manual_bin_count = None
                if bin_mode == "Manual Bin Width":
                    data_range = float(series.max() - series.min()) if len(series) > 1 else 1.0
                    auto_hint  = data_range / 10 if data_range > 0 else 1.0
                    manual_bin_width = st.number_input(
                        "Bin Width",
                        min_value=0.0001,
                        value=float(round(auto_hint, 2)),
                        step=float(max(round(auto_hint / 10, 2), 0.1)),
                        key="mk_manual_bin_width",
                    )
                elif bin_mode == "Manual Number of Bins":
                    manual_bin_count = st.number_input(
                        "Number of Bins", min_value=1, value=10, step=1,
                        key="mk_manual_bin_count",
                    )

                _draw_charts(
                    series, lbl,
                    show_boxplot, show_hist, show_kde,
                    bin_mode, manual_bin_width, manual_bin_count,
                )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — IQR CALCULATOR  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════
with ct_main_tab2:
    st.subheader("IQR Calculator")
    st.caption(
        "Enter Q1 and Q3. The app computes IQR, Lower Bound and Upper Bound."
    )

    q1_col, q3_col = st.columns(2)
    q1_value = q1_col.number_input("Q1", value=0.0, step=0.1, key="iqr_q1")
    q3_value = q3_col.number_input("Q3", value=0.0, step=0.1, key="iqr_q3")

    iqr_value   = q3_value - q1_value
    lower_bound = q1_value - 1.5 * iqr_value
    upper_bound = q3_value + 1.5 * iqr_value

    result_df = pd.DataFrame([{
        "Q1":          q1_value,
        "Q3":          q3_value,
        "IQR":         iqr_value,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
    }])

    st.markdown("### Result")
    st.dataframe(round_dataframe_for_display(result_df), use_container_width=True)

    st.markdown("### Formula")
    st.code("""
IQR         = Q3 − Q1
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
""")
    st.caption(
        "Lower and Upper Bounds define the whisker range for box-and-whisker "
        "plots using the Tukey method. Values outside these bounds are flagged as outliers."
    )
