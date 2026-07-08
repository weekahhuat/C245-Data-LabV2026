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

st.set_page_config(page_title="Probability Distributions", layout="wide")
_sidebar("Probability Distributions")

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

st.title("Probability Distributions")

dist_type = st.selectbox(
    "Choose Distribution",
    ["Binomial", "Poisson", "Normal"],
    key="dist_type",
)

st.divider()

left, right = st.columns([1, 2])

if dist_type == "Binomial":
    with left:
        st.markdown("#### Parameters")
        n = st.number_input("Number of trials (n)", min_value=1, value=10, step=1, key="binom_n")
        p = st.number_input("Probability of success (p)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="binom_p")

        st.markdown("#### Chart X-Axis Range")
        mean_binom = int(round(n * p))
        std_binom = max(1, int(round((n * p * (1 - p)) ** 0.5)))
        default_lo = max(0, mean_binom - 4 * std_binom)
        default_hi = min(int(n), mean_binom + 4 * std_binom)
        rc1, rc2 = st.columns(2)
        binom_xmin = rc1.number_input("From", min_value=0, max_value=int(n), value=default_lo, step=1, key="binom_xmin")
        binom_xmax = rc2.number_input("To", min_value=0, max_value=int(n), value=default_hi, step=1, key="binom_xmax")
        binom_range = (int(binom_xmin), int(binom_xmax))

        st.markdown("#### Probability")
        mode = st.selectbox(
            "Probability Type",
            ["P(X = x)", "P(X ≤ x)", "P(X < x)", "P(X ≥ x)", "P(X > x)", "P(a ≤ X ≤ b)"],
            key="binom_mode",
        )
        if mode == "P(a ≤ X ≤ b)":
            a_col, b_col = st.columns(2)
            a = a_col.number_input("a", min_value=0, max_value=int(n), value=2, step=1, key="binom_a")
            b = b_col.number_input("b", min_value=0, max_value=int(n), value=5, step=1, key="binom_b")
            lower = min(a, b)
            upper = max(a, b)
            value = binom.cdf(upper, n, p) - binom.cdf(lower - 1, n, p)
            label = f"P({lower} ≤ X ≤ {upper})"
            x_value = None
        else:
            x_value = st.number_input("x", min_value=0, max_value=int(n), value=3, step=1, key="binom_x")
            a = None
            b = None
            if mode == "P(X = x)":
                value = binom.pmf(x_value, n, p)
                label = f"P(X = {x_value})"
            elif mode == "P(X ≤ x)":
                value = binom.cdf(x_value, n, p)
                label = f"P(X ≤ {x_value})"
            elif mode == "P(X < x)":
                value = binom.cdf(x_value - 1, n, p)
                label = f"P(X < {x_value})"
            elif mode == "P(X ≥ x)":
                value = 1 - binom.cdf(x_value - 1, n, p)
                label = f"P(X ≥ {x_value})"
            else:
                value = 1 - binom.cdf(x_value, n, p)
                label = f"P(X > {x_value})"

        st.divider()
        format_probability_result(label, value)

    with right:
        st.markdown("#### Chart")
        xs = np.arange(0, n + 1)
        ys = binom.pmf(xs, n, p)
        plot_discrete_distribution(xs, ys, f"Binomial Distribution (n={n}, p={p})", mode, x_value, a, b, x_range=binom_range)

elif dist_type == "Poisson":
    with left:
        st.markdown("#### Parameters")
        lam = st.number_input("Poisson mean / rate (λ)", min_value=0.0001, value=4.0, step=0.1, key="pois_lambda")
        upper_x = max(20, int(math.ceil(lam + 4 * math.sqrt(lam))))

        st.markdown("#### Chart X-Axis Range")
        std_pois = max(1, int(round(lam ** 0.5)))
        default_lo_p = max(0, int(round(lam - 4 * std_pois)))
        default_hi_p = min(upper_x, int(round(lam + 4 * std_pois)))
        pc1, pc2 = st.columns(2)
        pois_xmin = pc1.number_input("From", min_value=0, max_value=upper_x, value=default_lo_p, step=1, key="pois_xmin")
        pois_xmax = pc2.number_input("To", min_value=0, max_value=upper_x, value=default_hi_p, step=1, key="pois_xmax")
        pois_range = (int(pois_xmin), int(pois_xmax))

        st.markdown("#### Probability")
        mode = st.selectbox(
            "Probability Type",
            ["P(X = x)", "P(X ≤ x)", "P(X < x)", "P(X ≥ x)", "P(X > x)", "P(a ≤ X ≤ b)"],
            key="pois_mode",
        )
        if mode == "P(a ≤ X ≤ b)":
            a_col, b_col = st.columns(2)
            a = a_col.number_input("a", min_value=0, value=2, step=1, key="pois_a")
            b = b_col.number_input("b", min_value=0, value=5, step=1, key="pois_b")
            lower = min(a, b)
            upper = max(a, b)
            value = poisson.cdf(upper, lam) - poisson.cdf(lower - 1, lam)
            label = f"P({lower} ≤ X ≤ {upper})"
            x_value = None
        else:
            x_value = st.number_input("x", min_value=0, value=3, step=1, key="pois_x")
            a = None
            b = None
            if mode == "P(X = x)":
                value = poisson.pmf(x_value, lam)
                label = f"P(X = {x_value})"
            elif mode == "P(X ≤ x)":
                value = poisson.cdf(x_value, lam)
                label = f"P(X ≤ {x_value})"
            elif mode == "P(X < x)":
                value = poisson.cdf(x_value - 1, lam)
                label = f"P(X < {x_value})"
            elif mode == "P(X ≥ x)":
                value = 1 - poisson.cdf(x_value - 1, lam)
                label = f"P(X ≥ {x_value})"
            else:
                value = 1 - poisson.cdf(x_value, lam)
                label = f"P(X > {x_value})"

        st.divider()
        format_probability_result(label, value)

    with right:
        st.markdown("#### Chart")
        xs = np.arange(0, upper_x + 1)
        ys = poisson.pmf(xs, lam)
        plot_discrete_distribution(xs, ys, f"Poisson Distribution (λ={lam})", mode, x_value, a, b, x_range=pois_range)

else:  # Normal
    with left:
        st.markdown("#### Parameters")
        mu = st.number_input("Mean (μ)", value=50.0, step=0.1, key="norm_mu")
        sigma = st.number_input("Standard deviation (σ)", min_value=0.0001, value=10.0, step=0.1, key="norm_sigma")

        st.markdown("#### Probability")
        mode = st.selectbox(
            "Probability Type",
            ["P(X ≤ a)", "P(X ≥ a)", "P(a ≤ X ≤ b)"],
            key="norm_mode",
        )
        if mode == "P(a ≤ X ≤ b)":
            a_col, b_col = st.columns(2)
            a = a_col.number_input("a", value=45.0, step=0.1, key="norm_a")
            b = b_col.number_input("b", value=55.0, step=0.1, key="norm_b")
            lower = min(a, b)
            upper = max(a, b)
            value = norm.cdf(upper, mu, sigma) - norm.cdf(lower, mu, sigma)
            label = f"P({lower} ≤ X ≤ {upper})"
        elif mode == "P(X ≤ a)":
            a = st.number_input("a", value=45.0, step=0.1, key="norm_below_a")
            value = norm.cdf(a, mu, sigma)
            label = f"P(X ≤ {a})"
        else:
            a = st.number_input("a", value=55.0, step=0.1, key="norm_above_a")
            value = 1 - norm.cdf(a, mu, sigma)
            label = f"P(X ≥ {a})"

        st.divider()
        format_probability_result(label, value)

    with right:
        st.markdown("#### Chart")
        if mode == "P(a ≤ X ≤ b)":
            plot_normal_distribution(mu, sigma, lower, upper, mode="between")
        elif mode == "P(X ≤ a)":
            plot_normal_distribution(mu, sigma, a=a, mode="below")
        else:
            plot_normal_distribution(mu, sigma, a=a, mode="above")
