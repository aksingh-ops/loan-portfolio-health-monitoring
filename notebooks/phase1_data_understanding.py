"""
Phase 1 — Data Loading and Understanding
==========================================
Lending Club-schema loan dataset: 50,000 records across 2007-2018.
Covers the full credit cycle including the 2008-2009 Global Financial Crisis.

Dataset profile
----------------
  Records      : 50,000 loans
  Features     : 25 raw
  Default rate : 13.2%  (6,604 loans)
  Imbalance    : 6.6:1  (paid vs defaulted)
  Volume       : $769M total portfolio
  Grade range  : A (3.2% default) to G (49.4% default)
  Vintage      : 2007-2018, GFC loans at 17.5% vs post-2012 at 12.9%

Why this dataset
-----------------
Lending Club is a real US peer-to-peer lender. Their public dataset contains
actual loan records with real borrower financials and real outcomes (paid off,
defaulted, late, charged off). It covers the full credit cycle including the
2008-2009 GFC, making it the standard benchmark dataset for consumer credit
default modelling. Same schema, same distributions, same structural breaks.

Outputs
-------
  outputs/fig1_data_understanding.png

Run
---
  python phase1_data_understanding.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

DATA_PATH  = "../data/lc_loans.csv"
OUTPUT_DIR = "../outputs"

C_DEF = "#A32D2D"
C_PAY = "#185FA5"
C_BG  = "#FAFAFA"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.5,
    "figure.facecolor": C_BG, "axes.facecolor": C_BG,
})


def load_and_profile(path):
    df = pd.read_csv(path, parse_dates=["issue_d"])

    print("Dataset profile")
    print(f"  Records         : {len(df):,}")
    print(f"  Features        : {df.shape[1]}")
    print(f"  Default rate    : {df['default'].mean()*100:.1f}%  ({df['default'].sum():,} loans)")
    paid   = (df['default']==0).sum()
    deflt  = (df['default']==1).sum()
    print(f"  Class imbalance : {paid/deflt:.1f}:1  (paid vs defaulted)")
    print(f"  Portfolio volume: ${df['loan_amnt'].sum()/1e6:.0f}M")
    print(f"  Avg loan amount : ${df['loan_amnt'].mean():,.0f}")
    print(f"  Avg int rate    : {df['int_rate'].mean():.1f}%")
    print(f"  Vintage range   : {df['issue_d'].dt.year.min()}-{df['issue_d'].dt.year.max()}")

    print("\nDefault rate by grade (primary risk signal):")
    for g in ["A","B","C","D","E","F","G"]:
        sub  = df[df["grade"]==g]
        rate = sub["default"].mean()*100
        bar  = "█" * int(rate/2)
        print(f"  Grade {g}: {rate:>5.1f}%  {bar}  (n={len(sub):,})")

    print("\nDefault rate by vintage:")
    df["issue_year"] = df["issue_d"].dt.year
    yearly = df.groupby("issue_year")["default"].mean()*100
    for yr, rate in yearly.items():
        tag = " <- GFC" if yr in [2007,2008,2009] else ""
        print(f"  {yr}: {rate:.1f}%{tag}")

    return df


def plot_data_understanding(df, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Default rate by grade
    ax = axes[0]
    grades   = ["A","B","C","D","E","F","G"]
    dr       = [df[df["grade"]==g]["default"].mean()*100 for g in grades]
    n_loans  = [len(df[df["grade"]==g]) for g in grades]
    colors_g = ["#3B6D11","#3B6D11","#BA7517","#BA7517","#A32D2D","#A32D2D","#A32D2D"]
    bars = ax.bar(grades, dr, color=colors_g, alpha=0.85, width=0.65)
    for bar, val, n in zip(bars, dr, n_loans):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
                f"n={n//1000}K", ha="center", fontsize=8, color="white")
    ax.axhline(df["default"].mean()*100, color="#888", ls="--", lw=1,
               label=f"Overall {df['default'].mean()*100:.1f}%")
    ax.set_title("Default rate by loan grade\n(primary risk segmentation)", fontweight="bold")
    ax.set_xlabel("Grade (A=best, G=worst)"); ax.set_ylabel("Default rate %")
    ax.legend(fontsize=8)

    # Default rate by FICO band
    ax = axes[1]
    df["fico_band"] = pd.cut(df["fico_range_low"],
        bins=[620,650,680,710,740,770,800,850],
        labels=["620-650","650-680","680-710","710-740","740-770","770-800","800+"])
    fico_dr = df.groupby("fico_band")["default"].mean()*100
    colors_f = ["#A32D2D","#A32D2D","#BA7517","#BA7517","#3B6D11","#3B6D11","#3B6D11"]
    ax.bar(range(len(fico_dr)), fico_dr.values, color=colors_f, alpha=0.85, width=0.65)
    ax.set_xticks(range(len(fico_dr))); ax.set_xticklabels(fico_dr.index, rotation=30, ha="right", fontsize=8)
    for i, val in enumerate(fico_dr.values):
        ax.text(i, val+0.2, f"{val:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_title("Default rate by FICO score band\n(key underwriting signal)", fontweight="bold")
    ax.set_ylabel("Default rate %")

    # DTI vs interest rate scatter
    ax = axes[2]
    sample = df.dropna(subset=["dti"]).sample(3000, random_state=42)
    paid_s  = sample[sample["default"]==0]
    deflt_s = sample[sample["default"]==1]
    ax.scatter(paid_s["dti"],  paid_s["int_rate"],  alpha=0.15, s=8, color=C_PAY, label="Paid")
    ax.scatter(deflt_s["dti"], deflt_s["int_rate"], alpha=0.25, s=10,color=C_DEF, label="Defaulted")
    ax.set_xlabel("Debt-to-income ratio (DTI %)"); ax.set_ylabel("Interest rate %")
    ax.set_title("DTI vs interest rate\n(defaults cluster top-right)", fontweight="bold")
    ax.legend(fontsize=9)

    # Vintage default rates
    ax = axes[3]
    df["issue_year"] = df["issue_d"].dt.year
    yearly = df.groupby("issue_year")["default"].agg(["mean","count"]).reset_index()
    yearly["rate"] = yearly["mean"]*100
    bar_colors = ["#A32D2D" if y in [2007,2008,2009] else
                  "#BA7517" if y in [2010,2011] else C_PAY
                  for y in yearly["issue_year"]]
    ax.bar(yearly["issue_year"], yearly["rate"], color=bar_colors, alpha=0.85, width=0.7)
    for _,row in yearly.iterrows():
        ax.text(row["issue_year"], row["rate"]+0.2,
                f"{row['rate']:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.axvspan(2006.5, 2009.5, color="#FCEBEB", alpha=0.3)
    ax.text(2008, yearly["rate"].max()*0.9, "GFC", ha="center",
            fontsize=10, color=C_DEF, fontweight="bold")
    ax.set_title("Default rate by vintage year\n(GFC 2007-2009 visible)", fontweight="bold")
    ax.set_xlabel("Issue year"); ax.set_ylabel("Default rate %")

    # Grade volume by outcome
    ax = axes[4]
    grade_vol = df.groupby(["grade","default"])["loan_amnt"].sum().unstack()/1e6
    grade_vol.columns = ["Paid","Defaulted"]
    x = np.arange(7); w = 0.35
    ax.bar(x-w/2, grade_vol["Paid"],      w, color=C_PAY, alpha=0.82, label="Paid")
    ax.bar(x+w/2, grade_vol["Defaulted"], w, color=C_DEF, alpha=0.82, label="Defaulted")
    ax.set_xticks(x); ax.set_xticklabels(grades)
    ax.set_ylabel("Volume ($M)")
    ax.set_title("Portfolio volume by grade\n(risk-weighted exposure view)", fontweight="bold")
    ax.legend(fontsize=9)

    # Feature correlations with default
    ax = axes[5]
    num_feats = ["int_rate","dti","fico_range_low","revol_util","open_acc",
                 "delinq_2yrs","pub_rec","inq_last_6mths","annual_inc","loan_amnt"]
    corrs = pd.Series(
        {f: df[f].corr(df["default"]) for f in num_feats if f in df.columns}
    ).sort_values()
    colors_c = ["#A32D2D" if v>0 else C_PAY for v in corrs.values]
    ax.barh(corrs.index, corrs.values, color=colors_c, alpha=0.85, height=0.65)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with default")
    ax.set_title("Feature correlation with default\n(target variable)", fontweight="bold")
    for i,(feat,val) in enumerate(corrs.items()):
        offset = 0.003 if val>=0 else -0.003
        ax.text(val+offset, i, f"{val:.3f}", va="center",
                ha="left" if val>=0 else "right", fontsize=8.5)

    fig.suptitle("Phase 1 — Loan Portfolio Data Understanding", fontsize=13, fontweight="bold")
    plt.tight_layout(pad=2)
    plt.savefig(f"{output_dir}/fig1_data_understanding.png",
                dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved: {output_dir}/fig1_data_understanding.png")


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_and_profile(DATA_PATH)
    plot_data_understanding(df, OUTPUT_DIR)
    print("\nPhase 1 complete. Run phase2_eda_cleaning.py next.")
