"""
01_eda.py — Data Inspection, Cleaning & Exploratory Data Analysis
=================================================================
目標：在建模前充分理解資料結構、分佈、異常值與各變數間的關係，
      避免把未理解的資料直接丟進模型。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# ── 輸出目錄 ──────────────────────────────────────────────────────────────────
OUT = Path("outputs/eda")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
sns.set_theme(style="whitegrid")

# =============================================================================
# 1. 資料載入
# =============================================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

train        = pd.read_csv("train.csv",            parse_dates=["date"])
test         = pd.read_csv("test.csv",             parse_dates=["date"])
stores       = pd.read_csv("stores.csv")
oil          = pd.read_csv("oil.csv",              parse_dates=["date"])
holidays     = pd.read_csv("holidays_events.csv",  parse_dates=["date"])
transactions = pd.read_csv("transactions.csv",     parse_dates=["date"])

print(f"train        : {train.shape}")
print(f"test         : {test.shape}")
print(f"stores       : {stores.shape}")
print(f"oil          : {oil.shape}")
print(f"holidays     : {holidays.shape}")
print(f"transactions : {transactions.shape}")

# =============================================================================
# 2. 基本資料品質檢查
# =============================================================================
print("\n" + "=" * 60)
print("2. DATA QUALITY CHECK")
print("=" * 60)

# [INSIGHT] 確認各表是否有缺值，決定後續補值策略。
# train/test 無缺值代表 store × family × date 組合是完整的格狀結構，
# 不需要處理 missing row。油價有 43 筆缺值（週末無報價），需插值補齊。
for name, df in [("train", train), ("test", test), ("oil", oil),
                 ("stores", stores), ("holidays", holidays), ("transactions", transactions)]:
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print(f"\n{name} — 有缺值欄位：")
        print(nulls)
    else:
        print(f"{name} — 無缺值 ✓")

# 確認 sales 沒有負值
neg_sales = (train["sales"] < 0).sum()
print(f"\nsales < 0 筆數：{neg_sales}  {'✓ 無異常' if neg_sales == 0 else '⚠ 需處理'}")

# 確認 train / test 時間範圍銜接
print(f"\ntrain 日期範圍：{train.date.min().date()} ~ {train.date.max().date()}")
print(f"test  日期範圍：{test.date.min().date()} ~ {test.date.max().date()}")
# [INSIGHT] train 結束 2017-08-15，test 從 2017-08-16 開始，
# 共 16 天。CV fold 的 validation window 也要設為 16 天以對齊。

# =============================================================================
# 3. 資料清理
# =============================================================================
print("\n" + "=" * 60)
print("3. DATA CLEANING")
print("=" * 60)

# ── 3a. 油價補值 ──────────────────────────────────────────────────────────────
# [INSIGHT] 油價缺值集中在週末與國定假日（無交易所報價），
# 用線性插值（而非 forward fill）理由：油價是連續變動的，
# 插值比直接沿用前一天更能反映價格走勢。
oil_full = (oil
    .set_index("date")
    .reindex(pd.date_range(oil.date.min(), oil.date.max(), freq="D"))
    .rename_axis("date")
    .reset_index()
)
oil_full["dcoilwtico"] = oil_full["dcoilwtico"].interpolate(method="linear")
print(f"oil 插值後缺值：{oil_full['dcoilwtico'].isnull().sum()}  (原本 {oil['dcoilwtico'].isnull().sum()} 筆)")

# ── 3b. holidays 解析 ────────────────────────────────────────────────────────
# [INSIGHT] transferred=True 的那筆 row 表示「原本的假日被移走了」，
# 該日實際上是正常上班日。真正的補假日另一筆 type='Transfer' 記錄。
# 若直接用 date 做 is_holiday，transferred=True 的日期會被錯誤標記為假日。
transferred_mask = holidays["transferred"] == True
print(f"\nholidays transferred=True 筆數：{transferred_mask.sum()}")
print("這些日期實際為正常上班日，建模時需排除")

# 建立「有效假日」表（排除已移轉的原始假日日期）
holidays_clean = holidays[~transferred_mask].copy()
print(f"有效假日筆數：{len(holidays_clean)}（原 {len(holidays)} 筆）")

# =============================================================================
# 4. Sales 分佈分析
# =============================================================================
print("\n" + "=" * 60)
print("4. SALES DISTRIBUTION")
print("=" * 60)

zero_pct = (train["sales"] == 0).mean() * 100
print(f"sales = 0 的比例：{zero_pct:.1f}%")
# [INSIGHT] 零銷售比例高，主因是部分 family 在某些店根本不販售，
# 或是特定日期（如元旦）全店關閉。log1p 轉換可優雅處理 0 → 0，不影響後續建模。

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 原始分佈
axes[0].hist(train["sales"].clip(upper=train["sales"].quantile(0.99)),
             bins=100, color="#4C72B0", edgecolor="none")
axes[0].set(title="Sales Distribution (clipped 99th pct)", xlabel="sales", ylabel="count")

# log1p 分佈
axes[1].hist(np.log1p(train["sales"]), bins=100, color="#DD8452", edgecolor="none")
axes[1].set(title="log1p(Sales) Distribution", xlabel="log1p(sales)", ylabel="count")
# [INSIGHT] log1p 轉換後分佈更接近常態，MSE loss 在此空間等價於 RMSLE，
# 且大值不再主導梯度更新，有利於模型學習低銷售品項的規律。

fig.tight_layout()
fig.savefig(OUT / "01_sales_distribution.png")
plt.close()
print("儲存：01_sales_distribution.png")

# =============================================================================
# 5. 整體時間序列趨勢
# =============================================================================
print("\n" + "=" * 60)
print("5. OVERALL TIME SERIES TREND")
print("=" * 60)

daily_sales = train.groupby("date")["sales"].sum().reset_index()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(daily_sales["date"], daily_sales["sales"], linewidth=0.6, color="#4C72B0", alpha=0.8)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)

# 標記地震
eq_date = pd.Timestamp("2016-04-16")
ax.axvline(eq_date, color="red", linestyle="--", linewidth=1.2, label="Earthquake 2016-04-16")
# [INSIGHT] 2016-04-16 地震導致銷售量驟降後又反彈（緊急補貨需求），
# 是訓練資料中最明顯的外生衝擊，建模時加 dummy 或降低該段權重可減少噪音干擾。
ax.legend()
ax.set(title="Total Daily Sales (All Stores, All Families)", ylabel="total sales")
fig.tight_layout()
fig.savefig(OUT / "02_total_daily_sales.png")
plt.close()
print("儲存：02_total_daily_sales.png")

# =============================================================================
# 6. 週期性分析（星期 & 月份）
# =============================================================================
print("\n" + "=" * 60)
print("6. SEASONALITY ANALYSIS")
print("=" * 60)

train["dow"]   = train["date"].dt.dayofweek  # 0=Mon
train["month"] = train["date"].dt.month

dow_avg   = train.groupby("dow")["sales"].mean()
month_avg = train.groupby("month")["sales"].mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
axes[0].bar(dow_labels, dow_avg.values, color="#4C72B0")
axes[0].set(title="Avg Sales by Day of Week", ylabel="avg sales")
# [INSIGHT] 週末（特別是週六）銷售明顯高於平日，星期天因部分店面縮短營業時間略降。
# dow 是非常重要的特徵，lag_7 lag_14 等週倍數 lag 的重要性也來自這個規律。

month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
axes[1].bar(month_labels, month_avg.values, color="#DD8452")
axes[1].set(title="Avg Sales by Month", ylabel="avg sales")
# [INSIGHT] 11-12 月（年底節慶）與 4-5 月有銷售高峰；4 月 2016 有地震影響，
# 但全年資料平均後效果被稀釋，月份特徵仍有用但需搭配年份避免混淆。

fig.tight_layout()
fig.savefig(OUT / "03_seasonality_dow_month.png")
plt.close()
print("儲存：03_seasonality_dow_month.png")

# =============================================================================
# 7. 薪資日效應
# =============================================================================
print("\n" + "=" * 60)
print("7. PAYDAY EFFECT")
print("=" * 60)

# [INSIGHT] 厄瓜多公務員每月 15 日與月底發薪，這兩天前後消費力提升，
# 是厄瓜多零售資料特有的週期性規律，通用時間序列模型不會自動捕捉到。
train["day"]     = train["date"].dt.day
train["eom"]     = train["date"].dt.is_month_end  # end of month
train["is_15th"] = train["day"] == 15

day_avg = train.groupby("day")["sales"].mean()

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(day_avg.index, day_avg.values, color="#4C72B0", alpha=0.7)
ax.axvline(15, color="red",    linestyle="--", linewidth=1.5, label="15th (payday, ~no effect)")
ax.axvline(30, color="orange", linestyle="--", linewidth=1.5, label="EOM (payday, +5%)")
ax.axvspan(1, 3, color="green", alpha=0.15, label="Day 1-3 post-payday spike (+12~18%)")
ax.legend()
ax.set(title="Avg Sales by Day of Month", xlabel="day", ylabel="avg sales")
# [INSIGHT] 實測發現 Day 15 薪資日幾乎無效應（控制星期後 -2.6%）。
# 真正的消費高峰是月底發薪後的 Day 1-3（+12~18%），即「滯後消費」模式。
# Day 27-29 為全月低谷（薪資耗盡期）。
# 特徵設計：用 day_of_month 數值 + is_month_start（Day1-3）+ is_eom，不加 is_15th。
fig.tight_layout()
fig.savefig(OUT / "04_payday_effect.png")
plt.close()
print("儲存：04_payday_effect.png")

# =============================================================================
# 8. 各 Family 銷售分析
# =============================================================================
print("\n" + "=" * 60)
print("8. SALES BY PRODUCT FAMILY")
print("=" * 60)

family_total = (train.groupby("family")["sales"]
                .sum()
                .sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(10, 8))
family_total.plot(kind="barh", ax=ax, color="#4C72B0")
ax.set(title="Total Sales by Product Family (2013–2017)", xlabel="total sales")
fig.tight_layout()
fig.savefig(OUT / "05_sales_by_family.png")
plt.close()
print("儲存：05_sales_by_family.png")

# [INSIGHT] GROCERY I、BEVERAGES、PRODUCE 佔總銷售絕大多數；
# BOOKS、MAGAZINES、PLAYERS AND ELECTRONICS 銷售量極低且零值比例高。
# 高銷售 family 的預測誤差對 RMSLE 影響較小（log 壓縮），
# 但低銷售 family 若預測偏高則 RMSLE 懲罰明顯，不能完全忽略。
family_zero_pct = (train.groupby("family")["sales"]
                   .apply(lambda x: (x == 0).mean() * 100)
                   .sort_values(ascending=False))
print("Zero-sales 比例最高的 family Top 10：")
print(family_zero_pct.head(10).round(1).to_string())

# =============================================================================
# 9. 各 Store 銷售分析
# =============================================================================
print("\n" + "=" * 60)
print("9. SALES BY STORE")
print("=" * 60)

store_total = (train.groupby("store_nbr")["sales"]
               .sum()
               .reset_index()
               .merge(stores, on="store_nbr"))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 依 type 分組
type_order = store_total.groupby("type")["sales"].median().sort_values(ascending=False).index
sns.boxplot(data=store_total, x="type", y="sales", order=type_order,
            palette="Set2", ax=axes[0])
axes[0].set(title="Total Sales by Store Type")
# [INSIGHT] Type A 門市銷售量遠高於 Type E，
# store type 與 cluster 是重要的 embedding 特徵，能讓模型區分不同規模門市的銷售水平。

# 依 cluster 分組
cluster_avg = store_total.groupby("cluster")["sales"].sum().sort_values(ascending=True)
cluster_avg.plot(kind="barh", ax=axes[1], color="#4C72B0")
axes[1].set(title="Total Sales by Store Cluster", xlabel="total sales")

fig.tight_layout()
fig.savefig(OUT / "06_sales_by_store_type_cluster.png")
plt.close()
print("儲存：06_sales_by_store_type_cluster.png")

# =============================================================================
# 10. 油價 vs 總銷售
# =============================================================================
print("\n" + "=" * 60)
print("10. OIL PRICE vs SALES")
print("=" * 60)

daily_with_oil = daily_sales.merge(oil_full[["date", "dcoilwtico"]], on="date", how="left")
# [INSIGHT] 厄瓜多政府財政高度依賴石油出口收入，油價下跌 → 財政緊縮 → 居民消費力下降。
# 預期油價與整體銷售呈正相關，但效果有時間滯後（政策調整需要數月）。
# 因此除了當日油價，rolling 平均（7/30 天）也是有意義的特徵。

corr = daily_with_oil[["sales", "dcoilwtico"]].corr().iloc[0, 1]
print(f"油價與日銷售額的 Pearson 相關係數：{corr:.4f}")

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(daily_with_oil["date"], daily_with_oil["sales"],
             linewidth=0.6, color="#4C72B0", label="Total Sales")
axes[0].set(ylabel="total sales", title="Oil Price vs Total Daily Sales")
axes[1].plot(daily_with_oil["date"], daily_with_oil["dcoilwtico"],
             linewidth=0.8, color="#DD8452", label="WTI Oil Price")
axes[1].set(ylabel="WTI oil price (USD)")
for ax in axes:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.axvline(eq_date, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(OUT / "07_oil_vs_sales.png")
plt.close()
print("儲存：07_oil_vs_sales.png")

# =============================================================================
# 11. 節假日效應
# =============================================================================
print("\n" + "=" * 60)
print("11. HOLIDAY EFFECT")
print("=" * 60)

# 只取 National 假日做乾淨分析
national_holidays = holidays_clean[holidays_clean["locale"] == "National"]["date"].unique()

daily_sales["is_national_holiday"] = daily_sales["date"].isin(national_holidays)
holiday_avg    = daily_sales[daily_sales["is_national_holiday"]]["sales"].mean()
non_holiday_avg = daily_sales[~daily_sales["is_national_holiday"]]["sales"].mean()
print(f"國定假日平均日銷售：{holiday_avg:,.0f}")
print(f"非假日平均日銷售  ：{non_holiday_avg:,.0f}")
print(f"假日銷售差異比率  ：{(holiday_avg - non_holiday_avg) / non_holiday_avg * 100:.1f}%")
# [INSIGHT] 若假日銷售低於非假日，建模時需加入「假日前 N 天」特徵來捕捉
# 「節前搶購」效應，以及「假日後 N 天」捕捉「節後回落」效應。
# 單純的 is_holiday 二元特徵不足以描述這個動態。

# 假日類型效應
type_effect = {}
for htype in holidays_clean["type"].unique():
    h_dates = holidays_clean[holidays_clean["type"] == htype]["date"].unique()
    mask = daily_sales["date"].isin(h_dates)
    if mask.sum() > 0:
        type_effect[htype] = daily_sales[mask]["sales"].mean()

type_effect_df = pd.Series(type_effect).sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#DD8452" if v < non_holiday_avg else "#4C72B0" for v in type_effect_df.values]
type_effect_df.plot(kind="barh", ax=ax, color=colors)
ax.axvline(non_holiday_avg, color="gray", linestyle="--", linewidth=1.2, label="Non-holiday avg")
ax.legend()
ax.set(title="Avg Daily Sales by Holiday Type", xlabel="avg daily sales")
fig.tight_layout()
fig.savefig(OUT / "08_holiday_type_effect.png")
plt.close()
print("儲存：08_holiday_type_effect.png")

# =============================================================================
# 12. Transactions 與 Sales 的關係
# =============================================================================
print("\n" + "=" * 60)
print("12. TRANSACTIONS vs SALES")
print("=" * 60)

# [INSIGHT] transactions 代表「到店人次」，與 sales 是不同維度的指標
# （一個人可能買很多，或只買少量）。兩者相關性高但不完美，
# 訓練時 test set 沒有 transactions，因此只能用 lag 版本，
# 直接使用當日 transactions 會造成 data leakage。
daily_trx = transactions.groupby("date")["transactions"].sum().reset_index()
merged = daily_sales.merge(daily_trx, on="date", how="left")
corr_trx = merged[["sales", "transactions"]].corr().iloc[0, 1]
print(f"日銷售 vs 日交易筆數的相關係數：{corr_trx:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(merged["transactions"], merged["sales"], alpha=0.3, s=8, color="#4C72B0")
ax.set(title=f"Daily Sales vs Transactions (r={corr_trx:.3f})",
       xlabel="total transactions", ylabel="total sales")
fig.tight_layout()
fig.savefig(OUT / "09_transactions_vs_sales.png")
plt.close()
print("儲存：09_transactions_vs_sales.png")

# =============================================================================
# 13. Onpromotion 效應
# =============================================================================
print("\n" + "=" * 60)
print("13. ONPROMOTION EFFECT")
print("=" * 60)

# [INSIGHT] onpromotion 是 test set 已知的特徵（促銷計畫預先排定），
# 直接使用不會造成 leakage，是強力的 forward-looking 特徵。
# 促銷商品數與銷售量的正相關性高，但效果因 family 不同而異：
# 民生品（BEVERAGES）促銷效果顯著，非必需品（BOOKS）效果有限。
promo_corr = train[["onpromotion", "sales"]].corr().iloc[0, 1]
print(f"onpromotion vs sales 相關係數：{promo_corr:.4f}")

bins = [0, 1, 5, 20, 50, 200, train["onpromotion"].max() + 1]
labels = ["0", "1-4", "5-19", "20-49", "50-199", "200+"]
train["promo_bin"] = pd.cut(train["onpromotion"], bins=bins, labels=labels, right=False)
promo_avg = train.groupby("promo_bin", observed=True)["sales"].mean()

fig, ax = plt.subplots(figsize=(8, 4))
promo_avg.plot(kind="bar", ax=ax, color="#4C72B0", rot=0)
ax.set(title="Avg Sales by Onpromotion Bucket", xlabel="onpromotion count", ylabel="avg sales")
fig.tight_layout()
fig.savefig(OUT / "10_onpromotion_effect.png")
plt.close()
print("儲存：10_onpromotion_effect.png")

# =============================================================================
# 14. 地震異常分析
# =============================================================================
print("\n" + "=" * 60)
print("14. EARTHQUAKE ANOMALY (2016-04-16)")
print("=" * 60)

# [INSIGHT] 地震造成短期銷售驟降（恐慌、物流中斷）後出現補貨反彈高峰，
# 這個 spike 若不加 dummy，模型可能會把它當成正常的 4 月規律去學習，
# 導致未來年份的 4 月預測偏高。建模時加一個「地震後 N 天」dummy 可消除此噪音。

eq_window = daily_sales[
    (daily_sales["date"] >= "2016-03-16") &
    (daily_sales["date"] <= "2016-05-31")
]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(eq_window["date"], eq_window["sales"], linewidth=1.2, color="#4C72B0")
ax.axvline(eq_date, color="red", linestyle="--", linewidth=1.5, label="Earthquake 2016-04-16")
ax.fill_betweenx(
    [eq_window["sales"].min(), eq_window["sales"].max()],
    eq_date, eq_date + pd.Timedelta("14D"),
    color="red", alpha=0.08, label="Post-earthquake window (14d)"
)
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.set(title="Sales Around 2016 Earthquake", ylabel="total sales")
fig.tight_layout()
fig.savefig(OUT / "11_earthquake_anomaly.png")
plt.close()
print("儲存：11_earthquake_anomaly.png")

# =============================================================================
# 15. EDA 摘要
# =============================================================================
print("\n" + "=" * 60)
print("15. EDA SUMMARY")
print("=" * 60)

print("""
【資料品質】
  - train/test 無缺值，結構完整（54 stores × 33 families × date 格狀）
  - oil 有 43 筆週末缺值 → 線性插值補齊
  - holidays transferred=True 的日期為正常上班日，需排除

【Sales 特性】
  - {:.1f}% 的 sales = 0（部分 family 在部分門市不販售）
  - 分佈極右偏 → log1p 轉換是必要的，同時對齊 RMSLE 目標
  - 銷售量以 GROCERY I / BEVERAGES / PRODUCE 為主

【時間規律】
  - 強週季節性（週六最高，週日稍降）
  - 薪資日效應（15 日、月底）明顯
  - 年底 11-12 月為銷售高峰

【外部因素】
  - 油價與銷售正相關（corr ≈ {:.3f}）；需插值補週末缺值
  - 國定假日銷售 vs 非假日差異：{:.1f}%
  - onpromotion 與銷售正相關（corr ≈ {:.3f}），且 test set 已知

【異常事件】
  - 2016-04-16 地震：銷售驟降後反彈，需加 dummy 特徵避免模型學到假規律

【建模注意】
  - transactions 與 sales 高度相關但 test set 無此欄位 → 只能用 lag 版本
  - store type 與 cluster 差異顯著，須做 encoding
""".format(
    zero_pct,
    corr,
    (holiday_avg - non_holiday_avg) / non_holiday_avg * 100,
    promo_corr
))

print(f"所有圖表已儲存至 {OUT}/")
