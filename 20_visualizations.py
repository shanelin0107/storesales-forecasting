"""
20_visualizations.py — 生成 README 用的視覺化圖表
儲存至 images/ 資料夾
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

OUT    = Path("outputs")
IMGDIR = Path("images")
IMGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = "#2563EB"   # blue
RED     = "#DC2626"
GREEN   = "#16A34A"
GRAY    = "#6B7280"

print("載入資料...")
train = pd.read_parquet(OUT / "train_fe.parquet")
# 還原 log1p → 實際銷售量
train["sales_raw"] = np.expm1(train["sales"])

# 家族名稱對照（family_enc → 原始名稱）
family_names = {
    0: "AUTOMOTIVE", 1: "BABY CARE", 2: "BEAUTY", 3: "BEVERAGES",
    4: "BOOKS", 5: "BREAD/BAKERY", 6: "CELEBRATION", 7: "CLEANING",
    8: "DAIRY", 9: "DELI", 10: "EGGS", 11: "FROZEN FOODS",
    12: "GROCERY I", 13: "GROCERY II", 14: "HARDWARE", 15: "HOME AND KITCHEN I",
    16: "HOME AND KITCHEN II", 17: "HOME CARE", 18: "LADIESWEAR", 19: "LAWN AND GARDEN",
    20: "LINGERIE", 21: "LIQUOR,WINE,BEER", 22: "MAGAZINES", 23: "MEATS",
    24: "PERSONAL CARE", 25: "PET SUPPLIES", 26: "PLAYERS AND ELECTRONICS",
    27: "POULTRY", 28: "PREPARED FOODS", 29: "PRODUCE", 30: "SCHOOL AND OFFICE SUPPLIES",
    31: "SEAFOOD", 32: "SEAFOOD"
}

# =============================================================================
# 1. Score Progression
# =============================================================================
print("Plot 1: Score Progression...")

versions = ["v3\n(Global)", "v11\n(Per-Family)", "v12\n(+Dense Lags\n16~63)",
            "v14\n(+Recursive\nForecast)", "v15\n(+Zero-Sales\nFilter)"]
scores   = [0.430, 0.42084, 0.41781, 0.397, 0.38465]
colors   = [GRAY, GRAY, GRAY, GRAY, GREEN]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(versions, scores, color=colors, width=0.55, zorder=3)

# 標記分數
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{score:.4f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

# 競賽目標線
ax.axhline(0.37984, color=RED, linestyle="--", linewidth=1.3, label="Top competitor (0.37984)")
ax.set_ylim(0.36, 0.44)
ax.set_ylabel("LB Score (RMSLE)", fontsize=11)
ax.set_title("Model Improvement Journey — Kaggle LB Score", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
plt.tight_layout()
plt.savefig(IMGDIR / "01_score_progression.png", bbox_inches="tight")
plt.close()

# =============================================================================
# 2. Weekly Seasonality
# =============================================================================
print("Plot 2: Weekly Seasonality...")

dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
weekly  = (
    train.groupby("day_of_week")["sales_raw"]
    .mean()
    .rename(index=dow_map)
    .reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(weekly.index, weekly.values, color=PALETTE, width=0.6, zorder=3)
ax.set_ylabel("Average Daily Sales (units)", fontsize=11)
ax.set_title("Weekly Sales Pattern\n(Why lag_7 is the strongest short-term predictor)", fontsize=12, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
for i, (day, val) in enumerate(weekly.items()):
    ax.text(i, val + 0.3, f"{val:.0f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(IMGDIR / "02_weekly_seasonality.png", bbox_inches="tight")
plt.close()

# =============================================================================
# 3. Autocorrelation (ACF) — 用一條代表性 series
# =============================================================================
print("Plot 3: Autocorrelation...")

# 取銷售量最高的 store×family（BEVERAGES, store 1）
series = (
    train[(train["family_enc"] == 3) & (train["store_nbr"] == 1)]
    .sort_values("date")["sales_raw"]
    .values
)
series = series[series > 0]   # 排除零值
MAX_LAG = 56

# 手動計算 ACF
mean   = series.mean()
var    = np.var(series)
acf    = [1.0]
for lag in range(1, MAX_LAG + 1):
    cov = np.mean((series[lag:] - mean) * (series[:-lag] - mean))
    acf.append(cov / var)

lags = np.arange(MAX_LAG + 1)
conf = 1.96 / np.sqrt(len(series))

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(lags[1:], acf[1:], color=[RED if l % 7 == 0 else PALETTE for l in lags[1:]], width=0.7, zorder=3)
ax.axhline(conf,  color=GRAY, linestyle="--", linewidth=1, label=f"95% CI (±{conf:.3f})")
ax.axhline(-conf, color=GRAY, linestyle="--", linewidth=1)
ax.axhline(0, color="black", linewidth=0.8)

# 標記 7 的倍數
for lag in [7, 14, 21, 28, 35, 42, 49, 56]:
    ax.text(lag, acf[lag] + 0.01, str(lag), ha="center", va="bottom",
            fontsize=8, color=RED, fontweight="bold")

ax.set_xlabel("Lag (days)", fontsize=11)
ax.set_ylabel("Autocorrelation", fontsize=11)
ax.set_title("Autocorrelation Function (ACF) — BEVERAGES\n(Red bars = multiples of 7: strong weekly cycle)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig(IMGDIR / "03_autocorrelation.png", bbox_inches="tight")
plt.close()

# =============================================================================
# 4. Sales Distribution by Family（為何需要 Per-Family 模型）
# =============================================================================
print("Plot 4: Sales by Family...")

selected_families = [3, 5, 7, 8, 11, 12, 23, 24, 27, 29]  # 多樣化的 10 個
fam_labels = [family_names.get(f, str(f)) for f in selected_families]

data_fam = [
    np.log1p(train[train["family_enc"] == f]["sales_raw"].values)
    for f in selected_families
]

fig, ax = plt.subplots(figsize=(12, 5))
bp = ax.boxplot(data_fam, labels=fam_labels, patch_artist=True,
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                flierprops=dict(marker=".", markersize=2, alpha=0.3))

for patch in bp["boxes"]:
    patch.set_facecolor(PALETTE)
    patch.set_alpha(0.75)

ax.set_xticklabels(fam_labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("log(Sales + 1)", fontsize=11)
ax.set_title("Sales Distribution by Product Family\n(Different scale and variance → per-family models needed)",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(IMGDIR / "04_sales_by_family.png", bbox_inches="tight")
plt.close()

# =============================================================================
# 5. Zero-Sales Heatmap（store × family）
# =============================================================================
print("Plot 5: Zero-Sales Heatmap...")

zero_rate = (
    train.groupby(["store_nbr", "family_enc"])["sales_raw"]
    .apply(lambda x: (x == 0).mean())
    .unstack(fill_value=0)
)

fig, ax = plt.subplots(figsize=(16, 7))
sns.heatmap(
    zero_rate.T, ax=ax,
    cmap="YlOrRd", vmin=0, vmax=1,
    linewidths=0.2, linecolor="white",
    cbar_kws={"label": "Fraction of days with zero sales", "shrink": 0.8},
    xticklabels=True, yticklabels=[family_names.get(i, str(i)) for i in zero_rate.columns]
)
ax.set_xlabel("Store Number", fontsize=11)
ax.set_ylabel("")
ax.set_title("Zero-Sales Rate: Store × Product Family\n(Dark = rarely zero | Yellow/Red = frequently zero → candidates for zero-sales filter)",
             fontsize=12, fontweight="bold", pad=12)
ax.tick_params(axis="y", labelsize=8)
ax.tick_params(axis="x", labelsize=8)
plt.tight_layout()
plt.savefig(IMGDIR / "05_zero_sales_heatmap.png", bbox_inches="tight")
plt.close()

# =============================================================================
# 6. Overall Sales Trend（2013–2017）
# =============================================================================
print("Plot 6: Sales Trend Over Time...")

daily = train.groupby("date")["sales_raw"].sum().reset_index()
daily["date"] = pd.to_datetime(daily["date"])
daily["rolling_28"] = daily["sales_raw"].rolling(28, center=True, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(daily["date"], daily["sales_raw"], color=PALETTE, alpha=0.25, linewidth=0.6, label="Daily total")
ax.plot(daily["date"], daily["rolling_28"], color=PALETTE, linewidth=2, label="28-day rolling mean")

# 地震標記
eq_date = pd.Timestamp("2016-04-16")
ax.axvline(eq_date, color=RED, linestyle="--", linewidth=1.3, label="2016 Earthquake")
ax.text(eq_date, ax.get_ylim()[1] * 0.95, " Earthquake\n 2016-04-16",
        color=RED, fontsize=8, va="top")

ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Total Daily Sales (all stores & families)", fontsize=11)
ax.set_title("Overall Sales Trend — 2013 to 2017", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(IMGDIR / "06_sales_trend.png", bbox_inches="tight")
plt.close()

print(f"\n完成！6 張圖已儲存至 {IMGDIR}/")
for f in sorted(IMGDIR.glob("*.png")):
    print(f"  {f.name}")
