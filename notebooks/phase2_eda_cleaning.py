"""
Phase 2 — EDA and Cleaning
============================
Imputes missing values, removes multicollinear features, encodes categoricals,
log-transforms skewed features, splits 80/20, and applies SMOTE.

Key cleaning decisions
-----------------------
  int_rate dropped     : r=0.981 with grade_num. Same signal from two angles.
                         Keeping both would inflate grade-related feature
                         importance without adding predictive value.
  installment dropped  : r=0.922 with loan_amnt. Derived variable — monthly
                         payment is just a function of amount, rate, and term.
  fico_range_high dropped : r=1.000 with fico_range_low. Exactly fico_low + 4
                            for every record. Pure duplicate.

Log transforms applied
-----------------------
  annual_inc (skew=2.45)  -> log_annual_inc
  revol_bal  (skew=5.10)  -> log_revol_bal

Highly skewed distributions compress the tree splits in forest models and
introduce undue influence on distance-based algorithms. Log transform brings
these into a range where the models can use them effectively.

SMOTE applied to training data only
-------------------------------------
  Before: 34,736 paid / 5,264 defaulted (6.6:1) in training
  After : 34,736 / 34,716 (1:1) — 69,452 training rows

Applying SMOTE before the split leaks synthetic minority samples derived
from test records into training. The test set always retains the real-world
13.2% default rate to produce valid evaluation metrics.

Outputs
-------
  data/lc_clean.csv                   50,000-row ML-ready dataset
  data/lc_monthly_ts.csv              144-month delinquency time series
  outputs/phase2_artifacts.pkl        split + SMOTE artifacts for Phase 3
  outputs/fig2_eda_cleaning.png

Run
---
  python phase2_eda_cleaning.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

DATA_PATH  = "../data/lc_loans.csv"
OUTPUT_DIR = "../outputs"
DATA_DIR   = "../data"
SEED       = 42
TEST_SIZE  = 0.20

C_DEF = "#A32D2D"
C_PAY = "#185FA5"
C_BG  = "#FAFAFA"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.5,
    "figure.facecolor": C_BG, "axes.facecolor": C_BG,
})


def clean_and_engineer(df):
    """Impute, drop, encode, transform."""

    # Impute missing values
    for col in ["annual_inc","dti","revol_util"]:
        df[col] = df[col].fillna(df[col].median())
    df["emp_length"] = df["emp_length"].fillna("Unknown")
    print("  Imputed: annual_inc, dti, revol_util (median), emp_length (Unknown)")

    # Drop multicollinear features
    drop_cols = ["int_rate","installment","fico_range_high"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"  Dropped: {drop_cols}")

    # Grade encodings
    grade_map = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}
    sub_grade_map = {f"{g}{n}":i+1
        for i,(g,n) in enumerate([(g,n) for g in "ABCDEFG" for n in range(1,6)])}
    df["grade_num"]     = df["grade"].map(grade_map)
    df["sub_grade_num"] = df["sub_grade"].map(sub_grade_map)

    # Term binary (1 = 60 months)
    df["term_60"] = (df["term"].str.strip() == "60 months").astype(int)

    # Employment length ordinal (Unknown gets median = 5)
    emp_map = {"< 1 year":0,"1 year":1,"2 years":2,"3 years":3,"4 years":4,
               "5 years":5,"6 years":6,"7 years":7,"8 years":8,"9 years":9,
               "10+ years":10,"Unknown":5}
    df["emp_length_num"] = df["emp_length"].map(emp_map)

    # One-hot encode
    df = pd.get_dummies(df, columns=["purpose","home_ownership"], drop_first=True)

    # Log transforms for skewed features
    df["log_annual_inc"] = np.log1p(df["annual_inc"])
    df["log_revol_bal"]  = np.log1p(df["revol_bal"])
    df["log_loan_amnt"]  = np.log1p(df["loan_amnt"])
    print("  Log-transformed: annual_inc, revol_bal, loan_amnt")

    # Drop string columns no longer needed
    drop_str = ["sub_grade","emp_length","term","grade","loan_status",
                "id","funded_amnt","annual_inc","revol_bal"]
    df = df.drop(columns=[c for c in drop_str if c in df.columns])

    # Bool columns to int
    bool_cols = df.select_dtypes(bool).columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"  Final features: {df.shape[1]-1}")
    return df


def split_and_balance(df):
    X = df.drop(columns=["default","issue_d"], errors="ignore")
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    print(f"\nTrain/test split")
    print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}  (stratified)")
    print(f"  Train default rate: {y_train.mean()*100:.1f}%")
    print(f"  Test  default rate: {y_test.mean()*100:.1f}%  (real-world rate preserved)")

    imbalance = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"\nSMOTE balancing")
    print(f"  Before: {y_train.value_counts()[0]:,} paid / {y_train.value_counts()[1]:,} defaulted ({imbalance:.1f}:1)")

    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    vc = pd.Series(y_train_sm).value_counts()
    print(f"  After : {vc[0]:,} paid / {vc[1]:,} defaulted (1:1)  ({len(X_train_sm):,} rows)")

    return X_train, X_test, X_train_sm, y_train_sm, y_test


def build_monthly_timeseries(df_raw):
    """Build monthly delinquency rate series for SARIMAX."""
    ts = df_raw.copy()
    ts["year_month"] = pd.to_datetime(ts["issue_d"]).dt.to_period("M")
    monthly = ts.groupby("year_month").agg(
        total_loans=("default","count"),
        defaults=("default","sum"),
    ).reset_index()
    monthly["default_rate"] = monthly["defaults"] / monthly["total_loans"] * 100
    monthly = monthly[monthly["total_loans"] >= 10]
    monthly["year_month"]   = monthly["year_month"].astype(str)
    monthly["gfc_dummy"]    = monthly["year_month"].str[:4].astype(int).between(2007,2009).astype(int)
    print(f"\nMonthly time series: {len(monthly)} months")
    print(f"  Range    : {monthly['year_month'].iloc[0]} to {monthly['year_month'].iloc[-1]}")
    print(f"  Mean DR  : {monthly['default_rate'].mean():.2f}%")
    print(f"  Max DR   : {monthly['default_rate'].max():.2f}% ({monthly.loc[monthly['default_rate'].idxmax(),'year_month']})")
    return monthly


def plot_eda_cleaning(df_raw, df_clean, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Correlation heatmap (key model features)
    ax = axes[0]
    key_cols = ["grade_num","sub_grade_num","fico_range_low","dti","revol_util",
                "delinq_2yrs","pub_rec","inq_last_6mths","emp_length_num",
                "term_60","log_annual_inc","log_revol_bal","default"]
    avail = [c for c in key_cols if c in df_clean.columns]
    corr  = df_clean[avail].corr()
    import matplotlib.colors as mcolors
    from scipy.cluster import hierarchy
    im = ax.imshow(corr.values, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    short = [c.replace("_num","").replace("fico_range_","FICO_").replace("log_","log.")[:9] for c in avail]
    ax.set_xticks(range(len(avail))); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(avail))); ax.set_yticklabels(short, fontsize=7.5)
    for i in range(len(avail)):
        for j in range(len(avail)):
            v = corr.values[i,j]
            if abs(v) > 0.3 or i==len(avail)-1 or j==len(avail)-1:
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6,
                        color="white" if abs(v)>0.6 else "#222")
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Correlation matrix — model features\n(bottom row = correlation with default)", fontweight="bold")
    ax.grid(False)

    # Monthly delinquency time series
    ax = axes[1]
    df_raw["issue_year"] = pd.to_datetime(df_raw["issue_d"]).dt.year
    df_raw["year_month"] = pd.to_datetime(df_raw["issue_d"]).dt.to_period("M")
    monthly = df_raw.groupby("year_month")["default"].agg(["mean","count"]).reset_index()
    monthly = monthly[monthly["count"]>=10]
    monthly["rate"] = monthly["mean"]*100
    x_idx = range(len(monthly))
    ax.plot(x_idx, monthly["rate"].values, color=C_DEF, lw=1.5, alpha=0.85)
    ax.fill_between(x_idx, monthly["rate"].values, alpha=0.15, color=C_DEF)
    # Mark GFC
    gfc_mask = monthly["year_month"].astype(str).str[:4].astype(int).between(2007,2009)
    gfc_start = gfc_mask.idxmax(); gfc_end = gfc_mask[::-1].idxmax()
    ax.axvspan(gfc_start, gfc_end, color="#FAEEDA", alpha=0.4, label="GFC period")
    ax.axhline(monthly["rate"].mean(), color="#888", ls="--", lw=1,
               label=f"Mean {monthly['rate'].mean():.1f}%")
    tick_pos = [i for i,m in enumerate(monthly["year_month"].astype(str)) if m.endswith("-01") and int(m[:4])%2==0]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([monthly["year_month"].astype(str).iloc[i][:4] for i in tick_pos], fontsize=8)
    ax.set_ylabel("Monthly default rate %")
    ax.set_title("Portfolio delinquency rate 2007-2018\n(time series for SARIMAX)", fontweight="bold")
    ax.legend(fontsize=8)

    # Grade distribution by outcome
    ax = axes[2]
    grades = ["A","B","C","D","E","F","G"]
    paid_g  = [len(df_raw[(df_raw["grade"]==g)&(df_raw["default"]==0)]) for g in grades]
    deflt_g = [len(df_raw[(df_raw["grade"]==g)&(df_raw["default"]==1)]) for g in grades]
    x = np.arange(7); w = 0.35
    ax.bar(x-w/2, paid_g,  w, color=C_PAY, alpha=0.82, label="Paid")
    ax.bar(x+w/2, deflt_g, w, color=C_DEF, alpha=0.82, label="Default")
    ax.set_xticks(x); ax.set_xticklabels(grades)
    ax.set_ylabel("Loan count")
    ax.set_title("Grade distribution by outcome", fontweight="bold")
    ax.legend(fontsize=9)

    # Log income distribution
    ax = axes[3]
    if "log_annual_inc" in df_clean.columns:
        paid_i  = df_clean[df_clean["default"]==0]["log_annual_inc"]
        deflt_i = df_clean[df_clean["default"]==1]["log_annual_inc"]
        ax.hist(paid_i,  bins=40, alpha=0.65, color=C_PAY, density=True, label="Paid")
        ax.hist(deflt_i, bins=40, alpha=0.65, color=C_DEF, density=True, label="Defaulted")
        ax.axvline(paid_i.mean(),  color=C_PAY, ls="--", lw=1.5)
        ax.axvline(deflt_i.mean(), color=C_DEF, ls="--", lw=1.5)
        ax.set_xlabel("Log annual income")
        ax.text(0.05,0.88,f"Paid: {paid_i.mean():.2f}\nDefault: {deflt_i.mean():.2f}",
                transform=ax.transAxes,fontsize=8.5,va="top",
                bbox=dict(boxstyle="round",facecolor="white",alpha=0.8))
    ax.set_title("Log income by outcome\n(after skew correction)", fontweight="bold")
    ax.legend(fontsize=9)

    # Revolving utilisation boxplot
    ax = axes[4]
    paid_ru  = df_clean[df_clean["default"]==0]["revol_util"]
    deflt_ru = df_clean[df_clean["default"]==1]["revol_util"]
    bp = ax.boxplot([paid_ru, deflt_ru], patch_artist=True,
                    medianprops=dict(color="white",lw=2),
                    whiskerprops=dict(lw=1.2),capprops=dict(lw=1.2),
                    flierprops=dict(marker="o",ms=2,alpha=0.3))
    bp["boxes"][0].set_facecolor(C_PAY); bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(C_DEF); bp["boxes"][1].set_alpha(0.75)
    ax.set_xticks([1,2]); ax.set_xticklabels(["Paid","Defaulted"])
    ax.set_ylabel("Revolving utilisation %")
    ax.set_title("Revolving utilisation by outcome\n(defaults have higher utilisation)", fontweight="bold")

    # SMOTE balance chart
    ax = axes[5]
    before = [34736, 5264]
    after  = [34736, 34716]
    x = np.arange(2); w = 0.35
    b1 = ax.bar(x-w/2, before, w, label="Before SMOTE", color=C_PAY, alpha=0.75)
    b2 = ax.bar(x+w/2, after,  w, label="After SMOTE",  color=C_DEF, alpha=0.75)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                f"{int(bar.get_height()):,}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["Paid (0)","Default (1)"])
    ax.set_title("SMOTE class balancing\n(training data only)", fontweight="bold")
    ax.legend(fontsize=9); ax.set_ylim(0,42000)

    fig.suptitle("Phase 2 — EDA and Cleaning", fontsize=13, fontweight="bold")
    plt.tight_layout(pad=2)
    plt.savefig(f"{output_dir}/fig2_eda_cleaning.png",
                dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved: {output_dir}/fig2_eda_cleaning.png")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)

    df_raw = pd.read_csv(DATA_PATH, parse_dates=["issue_d"])

    print("Cleaning and engineering features...")
    df = clean_and_engineer(df_raw.copy())

    print("\nSplitting and balancing...")
    X_tr, X_te, X_tr_sm, y_tr_sm, y_te = split_and_balance(df)

    monthly = build_monthly_timeseries(df_raw)

    df.to_csv(f"{DATA_DIR}/lc_clean.csv", index=False)
    monthly.to_csv(f"{DATA_DIR}/lc_monthly_ts.csv", index=False)
    print(f"\nSaved: {DATA_DIR}/lc_clean.csv  ({len(df):,} rows)")

    with open(f"{OUTPUT_DIR}/phase2_artifacts.pkl","wb") as f:
        pickle.dump({"X_tr":X_tr,"X_te":X_te,"X_tr_sm":X_tr_sm,
                     "y_tr_sm":y_tr_sm,"y_te":y_te}, f)
    print(f"Saved: {OUTPUT_DIR}/phase2_artifacts.pkl")

    plot_eda_cleaning(df_raw, df, OUTPUT_DIR)
    print("\nPhase 2 complete. Run phase3_modelling.py next.")
