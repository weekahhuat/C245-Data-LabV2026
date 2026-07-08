"""
Shared helper functions for C245 Statistics pages.
No st.set_page_config() calls here — each page sets its own.
"""
import os, math
from io import StringIO
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm, t, gaussian_kde

def round_dataframe_for_display(df, decimals=2):
    d = df.copy()
    nc = d.select_dtypes(include=[np.number]).columns
    if len(nc): d[nc] = d[nc].round(decimals)
    return d

def get_excel_sheet_names(path):
    ext = os.path.splitext(path.lower())[1]
    if ext not in {".xlsx",".xls"}: return []
    return pd.ExcelFile(path).sheet_names

def save_uploaded_data_file(uploaded_file, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    safe_name = os.path.basename(uploaded_file.name)
    save_path = os.path.join(data_dir, safe_name)
    with open(save_path,"wb") as f: f.write(uploaded_file.getbuffer())
    return save_path

def delete_data_file(filename, data_dir):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(data_dir, safe_name)
    if not os.path.isfile(file_path): raise FileNotFoundError(f"File not found: {safe_name}")
    os.remove(file_path)
    return file_path

def list_data_files(data_dir):
    if not os.path.exists(data_dir): return []
    allowed = {".csv",".xlsx",".xls"}
    return [n for n in sorted(os.listdir(data_dir))
            if os.path.isfile(os.path.join(data_dir,n))
            and os.path.splitext(n.lower())[1] in allowed]

@st.cache_data(show_spinner=False)
def load_data_file(path, sheet_name=None):
    ext = os.path.splitext(path.lower())[1]
    if ext == ".csv": return pd.read_csv(path)
    if ext in {".xlsx",".xls"}:
        sn = sheet_name or get_excel_sheet_names(path)[0]
        return pd.read_excel(path, sheet_name=sn)
    raise ValueError(f"Unsupported: {ext}")

def parse_number_list(text):
    if not text.strip(): return []
    text = text.replace("\n",",").replace(";",",")
    return [float(p.strip()) for p in text.split(",") if p.strip()]

def parse_anova_groups(text):
    groups = {}
    for line in text.strip().splitlines():
        if ":" not in line: continue
        name, values = line.split(":",1)
        vals = parse_number_list(values)
        if vals: groups[name.strip()] = vals
    return groups

def parse_contingency_table(text):
    return pd.read_csv(StringIO(text.strip()), index_col=0)

def z_critical(conf_level):
    return norm.ppf(1-(1-conf_level)/2)

def decision_text(p_value, alpha):
    return "Reject the null hypothesis" if p_value < alpha else "Fail to reject the null hypothesis"

def p_value_from_test_stat_z(z_stat, alternative):
    if alternative in ("two-sided",): return 2*(1-norm.cdf(abs(z_stat)))
    if alternative in ("greater", "right-tailed"): return 1-norm.cdf(z_stat)
    return norm.cdf(z_stat)

def p_value_from_test_stat_t(t_stat, df, alternative):
    if alternative in ("two-sided",): return 2*(1-t.cdf(abs(t_stat),df))
    if alternative in ("greater", "right-tailed"): return 1-t.cdf(t_stat,df)
    return t.cdf(t_stat,df)

def format_probability_result(label, value):
    st.markdown(f"### {label}")
    st.markdown(f"**Decimal:** {value:.2f}")
    st.markdown(f"**Percentage:** {value*100:.2f}%")
    st.code(f"{label} = {value:.2f} = {value*100:.2f}%")

def plot_discrete_distribution(xs, ys, title, prob_mode=None, x_value=None, a=None, b=None, x_range=None):
    colors = []
    for val in xs:
        h = False
        if prob_mode=="P(X = x)" and x_value is not None: h = val==x_value
        elif prob_mode=="P(X ≤ x)" and x_value is not None: h = val<=x_value
        elif prob_mode=="P(X < x)" and x_value is not None: h = val<x_value
        elif prob_mode=="P(X ≥ x)" and x_value is not None: h = val>=x_value
        elif prob_mode=="P(X > x)" and x_value is not None: h = val>x_value
        elif prob_mode=="P(a ≤ X ≤ b)" and a is not None and b is not None: h = a<=val<=b
        colors.append("tab:orange" if h else "lightgray")
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(xs, ys, color=colors, edgecolor="black")
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("Probability")
    if x_range is not None:
        ax.set_xlim(x_range[0] - 0.5, x_range[1] + 0.5)
    st.pyplot(fig, use_container_width=False); plt.close(fig)

def plot_normal_distribution(mu, sigma, a=None, b=None, mode="between"):
    x = np.linspace(mu-4*sigma, mu+4*sigma, 500)
    y = norm.pdf(x, mu, sigma)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, y, color="tab:blue")
    ax.set_title(f"Normal Distribution (μ={mu}, σ={sigma})")
    ax.set_xlabel("x"); ax.set_ylabel("Density")
    if mode=="below" and a is not None: ax.fill_between(x[x<=a],y[x<=a],alpha=0.3,color="tab:orange")
    elif mode=="above" and a is not None: ax.fill_between(x[x>=a],y[x>=a],alpha=0.3,color="tab:orange")
    elif mode=="between" and a is not None and b is not None:
        mask=(x>=a)&(x<=b); ax.fill_between(x[mask],y[mask],alpha=0.3,color="tab:orange")
    st.pyplot(fig); plt.close(fig)

def kde_mode_value(series):
    data = series.dropna().astype(float)
    if len(data)<2: return None
    try:
        kde = gaussian_kde(data); xs = np.linspace(data.min(),data.max(),500)
        return float(xs[np.argmax(kde(xs))])
    except: return None

def get_hist_edges(series, manual_width=None, manual_bins=None):
    data = series.dropna().astype(float)
    if manual_bins:
        n_bins = int(manual_bins)
        edges = np.linspace(data.min(),data.max(),n_bins+1)
        bw = float(edges[1]-edges[0]) if len(edges)>1 else 1.0
    elif manual_width:
        bw = float(manual_width)
        edges = np.arange(data.min(),data.max()+bw,bw); n_bins=len(edges)-1
    else:
        q75,q25 = np.percentile(data,[75,25]); iqr=q75-q25
        bw = 2*iqr/(len(data)**(1/3)) if iqr>0 else 1.0
        edges = np.arange(data.min(),data.max()+bw,bw); n_bins=len(edges)-1
    return edges, round(bw,4), n_bins
