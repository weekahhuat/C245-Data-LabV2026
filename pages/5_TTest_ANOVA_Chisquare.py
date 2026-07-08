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

st.set_page_config(page_title="T-Test / ANOVA / Chi-square", layout="wide")
_sidebar("T-Test / ANOVA / Chi-square")

DATA_DIR = os.getenv("COMMON_DATA_DIR", "common_data")
os.makedirs(DATA_DIR, exist_ok=True)

st.markdown("""
<style>
/* ── Global Font Enlargement ── */
html, body, [class*="css"] {
    font-size: 17px !important;
}
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] ol {
    font-size: 17px !important;
    white-space: normal;
}
label, .stSelectbox label, .stNumberInput label,
.stTextInput label, .stRadio label, .stCheckbox label {
    font-size: 17px !important;
    font-weight: 600 !important;
}
.stSelectbox div[data-baseweb="select"] span,
input[type="number"] {
    font-size: 17px !important;
}
pre  { white-space: pre-wrap !important; font-size: 15px !important; }
code { white-space: pre-wrap !important; font-size: 15px !important; }
.stDataFrame td { white-space: normal !important; font-size: 15px !important; }
.stDataFrame th { font-size: 15px !important; font-weight: 700 !important; }
h1 { font-size: 2.2rem !important; }
h2 { font-size: 1.9rem !important; }
h3 { font-size: 1.6rem !important; }
h4 { font-size: 1.35rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("T-Test / ANOVA / Chi-square")
st.subheader("T-Test, ANOVA, and Chi-square")
ttest_tab, anova_tab, chisq_tab = st.tabs(["T-Test", "ANOVA", "Chi-square"])

with ttest_tab:
    ttest_type = st.selectbox(
        "Choose T-Test Type",
        ["One-Sample T-Test", "Independent Two-Sample T-Test", "Paired T-Test"],
        key="ttest_type",
    )

    _conf_t = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, key="ttest_alpha")
    alpha_t = round(1 - _conf_t, 2)

    if ttest_type == "One-Sample T-Test":
        sample_text = st.text_area("Sample values (comma-separated)", "12, 15, 14, 16, 18, 13, 17", key="ttest_one_sample_text")
        mu0 = st.number_input("Null mean (μ₀)", value=14.0, step=0.1, key="ttest_one_mu0")

        if st.button("Run One-Sample T-Test", key="btn_run_one_sample_t"):
            try:
                sample = np.array(parse_number_list(sample_text), dtype=float)
                stat, p_val = ttest_1samp(sample, popmean=mu0)
                df = len(sample) - 1
                decision = decision_text(p_val, alpha_t)

                st.markdown("### Hypotheses")
                st.write(f"H0: μ = {mu0}")
                st.write(f"H1: μ ≠ {mu0}")

                st.write(f"**Sample mean:** {sample.mean():.2f}")
                st.write(f"**t statistic:** {stat:.2f}")
                st.write(f"**Degrees of freedom:** {df}")
                st.write(f"**p-value:** {p_val:.2f}")
                st.success(decision)
            except Exception as e:
                st.error(f"Unable to run one-sample t-test: {e}")

    elif ttest_type == "Independent Two-Sample T-Test":
        col1, col2 = st.columns(2)
        sample1_text = col1.text_area("Group 1 values", "12, 15, 14, 16, 18", key="ttest_ind_group1")
        sample2_text = col2.text_area("Group 2 values", "10, 11, 13, 12, 9", key="ttest_ind_group2")
        equal_var = st.checkbox("Assume equal variances", value=False, key="ttest_ind_equal_var")

        if st.button("Run Independent Two-Sample T-Test", key="btn_run_ind_t"):
            try:
                s1 = np.array(parse_number_list(sample1_text), dtype=float)
                s2 = np.array(parse_number_list(sample2_text), dtype=float)
                stat, p_val = ttest_ind(s1, s2, equal_var=equal_var)
                decision = decision_text(p_val, alpha_t)

                st.markdown("### Hypotheses")
                st.write("H0: μ₁ = μ₂")
                st.write("H1: μ₁ ≠ μ₂")

                st.write(f"**Group 1 mean:** {s1.mean():.2f}")
                st.write(f"**Group 2 mean:** {s2.mean():.2f}")
                st.write(f"**t statistic:** {stat:.2f}")
                st.write(f"**p-value:** {p_val:.2f}")
                st.success(decision)
            except Exception as e:
                st.error(f"Unable to run independent t-test: {e}")

    else:
        col1, col2 = st.columns(2)
        before_text = col1.text_area("Before values", "12, 15, 14, 16, 18", key="ttest_paired_before")
        after_text = col2.text_area("After values", "13, 16, 15, 17, 19", key="ttest_paired_after")

        if st.button("Run Paired T-Test", key="btn_run_paired_t"):
            try:
                before = np.array(parse_number_list(before_text), dtype=float)
                after = np.array(parse_number_list(after_text), dtype=float)

                if len(before) != len(after):
                    st.error("Before and after samples must have the same number of values.")
                else:
                    stat, p_val = ttest_rel(before, after)
                    decision = decision_text(p_val, alpha_t)
                    diff = after - before

                    st.markdown("### Hypotheses")
                    st.write("H0: μd = 0")
                    st.write("H1: μd ≠ 0")

                    st.write(f"**Mean difference (after - before):** {diff.mean():.2f}")
                    st.write(f"**t statistic:** {stat:.2f}")
                    st.write(f"**p-value:** {p_val:.2f}")
                    st.success(decision)
            except Exception as e:
                st.error(f"Unable to run paired t-test: {e}")

with anova_tab:
    st.markdown("Enter one group per line in the format: `Group Name: value1, value2, value3`")
    anova_text = st.text_area(
        "Group data",
        "Group A: 12, 15, 14, 16\nGroup B: 10, 11, 9, 12\nGroup C: 18, 17, 19, 20",
        height=180,
        key="anova_text",
    )
    _conf_a = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, key="anova_alpha")
    alpha_a = round(1 - _conf_a, 2)

    if st.button("Run One-Way ANOVA", key="btn_run_anova"):
        try:
            groups = parse_anova_groups(anova_text)
            if len(groups) < 2:
                st.error("Please provide at least two groups.")
            else:
                arrays = [np.array(v, dtype=float) for v in groups.values()]
                stat, p_val = f_oneway(*arrays)
                decision = decision_text(p_val, alpha_a)

                summary_rows = []
                for name, vals in groups.items():
                    arr = np.array(vals, dtype=float)
                    sd = arr.std(ddof=1) if len(arr) > 1 else 0.0
                    ci_half = (
                        t.ppf(1 - alpha_a / 2, df=len(arr) - 1) * sd / np.sqrt(len(arr))
                        if len(arr) > 1 else 0.0
                    )
                    summary_rows.append(
                        {
                            "Group": name,
                            "n": len(arr),
                            "Mean": arr.mean(),
                            "SD": sd,
                            "CI_half": ci_half,
                        }
                    )

                summary_df = pd.DataFrame(summary_rows)
                st.markdown("### Hypotheses")
                st.write("H0: μ1 = μ2 = μ3 = ... = μk")
                st.write("H1: At least one group mean is different")

                st.dataframe(round_dataframe_for_display(summary_df), use_container_width=True)
                st.write(f"**F statistic:** {stat:.2f}")
                st.write(f"**p-value:** {p_val:.2f}")
                st.success(decision)
        except Exception as e:
            st.error(f"Unable to run ANOVA: {e}")

with chisq_tab:
    cs_left, cs_right = st.columns([1, 2])

    with cs_left:
        st.markdown("#### STEP 1 — Select Your Test")
        cs_type = st.selectbox("Chi-Square Test Type",
            ["Goodness of Fit", "Test of Independence"], key="cs_type")
        _conf_c = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, key="chisq_alpha")
        alpha_c = round(1 - _conf_c, 2)
        st.caption(f"α = {alpha_c}")

        st.markdown("#### STEP 2 — Enter Your Data")

        if cs_type == "Goodness of Fit":
            st.caption(
                "Enter category name and Observed count (O) only. "
                "Expected (E) is auto-calculated as Total N ÷ k (equal distribution assumed)."
            )
            gof_n = st.number_input("Number of categories (k)",
                min_value=2, max_value=10, value=4, step=1, key="cs_gof_k")
            hc = st.columns([2, 1, 1])
            hc[0].markdown("**Category**")
            hc[1].markdown("**O (Observed)**")
            hc[2].markdown("**E (Auto)**")
            gof_cat_defaults  = ["Tea", "Coffee", "Others", "Drink", "", "", "", "", "", ""]
            gof_obs_defaults  = [45, 50, 25, 31, 0, 0, 0, 0, 0, 0]
            # First pass — collect obs to compute total N
            _gof_obs_vals = []
            for i in range(int(gof_n)):
                _d = gof_obs_defaults[i] if i < len(gof_obs_defaults) else 0
                _o = st.session_state.get(f"cs_gof_obs_{i}", _d)
                _gof_obs_vals.append(int(_o) if isinstance(_o, (int, float)) else _d)
            _gof_total_n = sum(_gof_obs_vals)
            _gof_e_each  = round(_gof_total_n / int(gof_n), 4) if gof_n > 0 and _gof_total_n > 0 else 0.0
            cs_gof_rows = []
            for i in range(int(gof_n)):
                _obs_d = gof_obs_defaults[i] if i < len(gof_obs_defaults) else 0
                _cat_d = gof_cat_defaults[i] if i < len(gof_cat_defaults) else ""
                rc = st.columns([2, 1, 1])
                cat = rc[0].text_input(f"Cat{i+1}", value=_cat_d, key=f"cs_gof_cat_{i}", label_visibility="collapsed")
                obs = rc[1].number_input(f"O{i+1}", value=int(_obs_d), min_value=0, step=1, format="%d", key=f"cs_gof_obs_{i}", label_visibility="collapsed")
                # E is auto-calculated — read-only display, NOT a number_input
                rc[2].markdown(
                    f'<div style="background:#EBF3FB;border-radius:4px;padding:6px 10px;'
                    f'font-size:13px;color:#1A1A1A;text-align:center;margin-top:4px;">'
                    f'{_gof_e_each:.2f}</div>',
                    unsafe_allow_html=True
                )
                if cat.strip():
                    cs_gof_rows.append({"cat": cat, "O": obs, "E": _gof_e_each})
            # Σ E = Σ O validation banner
            _sigma_e = round(_gof_e_each * int(gof_n), 2)
            if _gof_total_n > 0 and abs(_sigma_e - _gof_total_n) > 0.1:
                st.error(f"⚠ Σ E ({_sigma_e}) ≠ Σ O ({_gof_total_n}). Check your inputs.")
            elif _gof_total_n > 0:
                st.success(f"✅ Σ E = Σ O = {_gof_total_n}  ·  E per category = {_gof_e_each:.2f}")

        else:  # Test of Independence
            st.caption("Enter column/row labels and Observed counts. Expected counts are auto-computed.")
            dim_c = st.columns(2)
            toi_nrows = int(dim_c[0].number_input("Rows (r)", min_value=2, max_value=4, value=2, step=1, key="cs_toi_nrows"))
            toi_ncols = int(dim_c[1].number_input("Columns (c)", min_value=2, max_value=4, value=2, step=1, key="cs_toi_ncols"))
            col_label_defaults = ["Male", "Female", "Group C", "Group D"]
            row_label_defaults = ["Yes", "No", "Maybe", "Other"]
            obs_defaults = [[30, 20, 0, 0], [10, 40, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            st.markdown("**Column Labels**")
            col_labels = []
            cl_cols = st.columns(toi_ncols)
            for j in range(toi_ncols):
                lbl = cl_cols[j].text_input(f"Col{j+1}", value=col_label_defaults[j],
                    key=f"cs_col_lbl_{j}", label_visibility="collapsed")
                col_labels.append(lbl)
            st.markdown("**Row Labels & Observed Counts (O)**")
            toi_rlbls, toi_obs = [], []
            for i in range(toi_nrows):
                rc = st.columns([1] + [1] * toi_ncols)
                rl = rc[0].text_input(f"Row{i+1}", value=row_label_defaults[i],
                    key=f"cs_row_lbl_{i}", label_visibility="collapsed")
                toi_rlbls.append(rl)
                row_obs = []
                for j in range(toi_ncols):
                    o = rc[j+1].number_input(f"O{i}{j}", value=int(obs_defaults[i][j]),
                        min_value=0, step=1, format="%d",
                        key=f"cs_toi_obs_{i}_{j}", label_visibility="collapsed")
                    row_obs.append(o)
                toi_obs.append(row_obs)

    with cs_right:
        try:
            from scipy.stats import chi2 as chi2_dist

            if cs_type == "Goodness of Fit":
                active = [r for r in cs_gof_rows if r["cat"].strip() and r["O"] > 0]
                if len(active) < 2:
                    st.info("Enter at least 2 categories on the left to see results.")
                else:
                    obs_arr = np.array([r["O"] for r in active], dtype=float)
                    exp_arr = np.array([r["E"] for r in active], dtype=float)
                    cats    = [r["cat"] for r in active]
                    # Guard: E > 0 required (no division by zero)
                    if np.any(exp_arr <= 0):
                        st.error("⚠ One or more Expected (E) values are 0. "
                                 "Ensure all Observed counts are entered so E = Total N ÷ k > 0.")
                        raise ValueError("zero E")
                    # Guard: Σ E = Σ O
                    sigma_o = float(obs_arr.sum())
                    sigma_e = float(exp_arr.sum())
                    if abs(sigma_e - sigma_o) > 0.5:
                        st.error(f"⚠ Σ E ({sigma_e:.2f}) ≠ Σ O ({sigma_o:.2f}). Fix before running.")
                        raise ValueError("sigma mismatch")
                    contrib   = (obs_arr - exp_arr) ** 2 / exp_arr
                    chi2_stat = float(contrib.sum())
                    df_val    = len(active) - 1
                    p_val     = float(chi2_dist.sf(chi2_stat, df_val))
                    chi2_crit = float(chi2_dist.ppf(1 - alpha_c, df_val))
                    rejected  = p_val < alpha_c
                    h0_str    = "The observed frequencies match the expected distribution"
                    h1_str    = "The observed frequencies do not match the expected distribution"

                    st.markdown("### Hypotheses")
                    st.write(f"**H₀:** {h0_str}")
                    st.write(f"**H₁:** {h1_str}")

                    st.markdown("### Observed vs Expected")
                    tbl = pd.DataFrame({
                        "Category":  cats,
                        "O":         obs_arr.round(2),
                        "E":         exp_arr.round(2),
                        "(O−E)²÷E": contrib.round(4),
                        "E ≥ 5?":   ["✅" if e >= 5 else "⚠" for e in exp_arr],
                    })
                    st.dataframe(tbl, use_container_width=True, hide_index=True)
                    e_warn = [c for c, e in zip(cats, exp_arr) if e < 5]
                    if e_warn:
                        st.warning(f"⚠ E < 5 in: {', '.join(e_warn)}. Consider combining categories.")

                    st.write(f"**χ² statistic:** {chi2_stat:.4f}")
                    st.write(f"**Critical χ²:** {chi2_crit:.4f}  (df = {df_val}, α = {alpha_c})")
                    st.write(f"**p-value:** {p_val:.4f}")
                    st.write(f"**df:** {df_val}  (k − 1 = {len(active)} − 1)")
                    if rejected:
                        st.error(f"🔴 Reject H₀ — p = {p_val:.4f} < α = {alpha_c}")
                        st.markdown(f"> At the {int((1-alpha_c)*100)}% confidence level, there is sufficient "
                                    f"statistical evidence that the observed frequencies do not match the "
                                    f"expected distribution (χ²({df_val}) = {chi2_stat:.2f}, p = {p_val:.4f}).")
                    else:
                        st.success(f"🟢 Fail to Reject H₀ — p = {p_val:.4f} ≥ α = {alpha_c}")
                        st.markdown(f"> At the {int((1-alpha_c)*100)}% confidence level, there is insufficient "
                                    f"statistical evidence that the observed frequencies differ from the "
                                    f"expected distribution (χ²({df_val}) = {chi2_stat:.2f}, p = {p_val:.4f}).")

            else:  # Test of Independence
                obs_mat  = np.array(toi_obs, dtype=float)
                row_tots = obs_mat.sum(axis=1)
                col_tots = obs_mat.sum(axis=0)
                grand    = obs_mat.sum()
                if grand == 0:
                    st.info("Enter observed counts on the left to see results.")
                    raise ValueError("no data")
                exp_mat      = np.outer(row_tots, col_tots) / grand
                contrib_mat  = (obs_mat - exp_mat) ** 2 / exp_mat
                chi2_stat    = float(contrib_mat.sum())
                df_val       = (toi_nrows - 1) * (toi_ncols - 1)
                p_val        = float(chi2_dist.sf(chi2_stat, df_val))
                chi2_crit    = float(chi2_dist.ppf(1 - alpha_c, df_val))
                rejected     = p_val < alpha_c
                h0_str       = "The two categorical variables are independent"
                h1_str       = "There is an association between the two categorical variables"

                st.markdown("### Hypotheses")
                st.write(f"**H₀:** {h0_str}")
                st.write(f"**H₁:** {h1_str}")

                obs_df = pd.DataFrame(obs_mat, index=toi_rlbls[:toi_nrows], columns=col_labels[:toi_ncols])
                obs_df["Row Total"] = row_tots
                col_tots_row = list(col_tots) + [grand]
                obs_df.loc["Col Total"] = col_tots_row
                st.markdown("### Observed Counts (O)")
                st.dataframe(obs_df.round(0), use_container_width=True)

                exp_df = pd.DataFrame(exp_mat, index=toi_rlbls[:toi_nrows], columns=col_labels[:toi_ncols])
                st.markdown("### Expected Counts (E) — auto-computed  [E = Row Total × Col Total ÷ Grand Total]")
                st.dataframe(exp_df.round(2), use_container_width=True)

                min_e = float(exp_mat.min())
                if min_e < 5:
                    st.warning(f"⚠ Minimum expected count = {min_e:.2f} < 5. Consider combining categories.")
                else:
                    st.success(f"✅ All expected counts ≥ 5 (minimum = {min_e:.2f})")

                st.write(f"**χ² statistic:** {chi2_stat:.4f}")
                st.write(f"**Critical χ²:** {chi2_crit:.4f}  (df = {df_val}, α = {alpha_c})")
                st.write(f"**p-value:** {p_val:.4f}")
                st.write(f"**df:** {df_val}  ((r−1)(c−1) = ({toi_nrows}−1)×({toi_ncols}−1))")
                if rejected:
                    st.error(f"🔴 Reject H₀ — p = {p_val:.4f} < α = {alpha_c}")
                    st.markdown(f"> At the {int((1-alpha_c)*100)}% confidence level, there is sufficient "
                                f"statistical evidence of an association between the two categorical variables "
                                f"(χ²({df_val}) = {chi2_stat:.2f}, p = {p_val:.4f}). "
                                f"⚠ Chi-Square shows association, not causation.")
                else:
                    st.success(f"🟢 Fail to Reject H₀ — p = {p_val:.4f} ≥ α = {alpha_c}")
                    st.markdown(f"> At the {int((1-alpha_c)*100)}% confidence level, there is insufficient "
                                f"statistical evidence of an association between the two categorical variables "
                                f"(χ²({df_val}) = {chi2_stat:.2f}, p = {p_val:.4f}).")

        except ValueError:
            pass
        except Exception as e:
            st.error(f"Unable to run Chi-square test: {e}")
