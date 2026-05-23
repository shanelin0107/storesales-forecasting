"""
02_feature_engineering.py — Feature Engineering
================================================
輸入：原始 CSV 檔
輸出：outputs/train_fe.parquet、outputs/test_fe.parquet

執行流程：
  1. 資料清理（oil 插值、holidays 過濾）
  2. 合併 train + test（讓 lag 特徵跨越邊界時連續）
  3. 靜態特徵（store、family encoding）
  4. 日期特徵
  5. 油價特徵
  6. 假日特徵
  7. Lag / Rolling 特徵（sales、transactions、onpromotion）
  8. 拆回 train / test，儲存
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# =============================================================================
# 1. 載入資料
# =============================================================================
print("載入資料...")
train        = pd.read_csv("train.csv",            parse_dates=["date"])
test         = pd.read_csv("test.csv",             parse_dates=["date"])
stores       = pd.read_csv("stores.csv")
oil          = pd.read_csv("oil.csv",              parse_dates=["date"])
holidays     = pd.read_csv("holidays_events.csv",  parse_dates=["date"])
transactions = pd.read_csv("transactions.csv",     parse_dates=["date"])

# =============================================================================
# 2. 資料清理
# =============================================================================

# ── 2a. Target log1p 轉換 ────────────────────────────────────────────────────
# [INSIGHT] log1p(sales) 讓 MSE loss 等價於 RMSLE，同時把極右偏分佈壓縮，
# 避免大銷售量主導梯度。預測時 expm1 還原。
train["sales"] = np.log1p(train["sales"])

# ── 2b. 油價插值 ─────────────────────────────────────────────────────────────
# [INSIGHT] 缺值集中在週末（交易所休市），線性插值比 forward-fill
# 更能反映價格連續變動的特性。
oil_full = (
    oil.set_index("date")
       .reindex(pd.date_range(oil.date.min(), oil.date.max(), freq="D"))
       .rename_axis("date")
       .reset_index()
)
oil_full["dcoilwtico"] = oil_full["dcoilwtico"].interpolate(method="linear")

# ── 2c. 有效假日表（排除已移轉日期）────────────────────────────────────────
# [INSIGHT] transferred=True 的那筆記錄代表「原始假日被移走了」，
# 該日實為正常上班日；真正的補假另有一筆 type='Transfer'。
# 若直接用 date join，這些日期會被錯誤標為假日。
holidays_clean = holidays[holidays["transferred"] == False].copy()

# =============================================================================
# 3. 合併 train + test，建立完整時間序列
# =============================================================================
# [INSIGHT] lag 特徵必須在合併後計算，否則 test set 最前幾筆的 lag 會是 NaN，
# 例如 lag_16 on 2017-08-16 需要 2017-08-01 的銷售，只有 train 有這筆資料。
train["is_train"] = 1
test["is_train"]  = 0
test["sales"]     = np.nan

df = (pd.concat([train, test], sort=False)
        .sort_values(["store_nbr", "family", "date"])
        .reset_index(drop=True))

print(f"合併後總筆數：{len(df):,}")

# =============================================================================
# 4. 靜態特徵：Store & Family
# =============================================================================

# ── 4a. Join store 資料 ───────────────────────────────────────────────────────
df = df.merge(stores, on="store_nbr", how="left")

# ── 4b. Label Encoding ───────────────────────────────────────────────────────
# [INSIGHT] LightGBM 可直接處理整數 encoding 的類別特徵（配合 categorical_feature 參數）。
# type (A-E)、city、state 均為名義類別，無順序關係，encoding 後需告知模型為 categorical。
label_encoders = {}
for col in ["type", "city", "state", "family"]:
    le = LabelEncoder()
    df[f"{col}_enc"] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Store/Family encoding 完成")
print(f"  family: {df['family'].nunique()} 類別")
print(f"  type  : {df['type'].nunique()} 類別  → {sorted(df['type'].unique())}")
print(f"  city  : {df['city'].nunique()} 類別")
print(f"  state : {df['state'].nunique()} 類別")

# =============================================================================
# 5. 日期特徵
# =============================================================================
df["day_of_week"]    = df["date"].dt.dayofweek        # 0=Mon, 6=Sun
df["day_of_month"]   = df["date"].dt.day
df["month"]          = df["date"].dt.month
df["year"]           = df["date"].dt.year
df["week_of_year"]   = df["date"].dt.isocalendar().week.astype(int)
df["quarter"]        = df["date"].dt.quarter
df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
df["is_eom"]         = df["date"].dt.is_month_end.astype(int)

# [INSIGHT] EDA 發現 Day 1-3 是全月銷售最高峰（+12~18%），
# 係月底薪資的「滯後消費」。Day 15 薪資日無明顯效應，不加入。
df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)

# [INSIGHT] 2016-04-16 地震造成異常銷售衝擊（驟降後反彈），
# 加 14 天 dummy 避免模型把這段當成正常季節規律學進去。
eq_start = pd.Timestamp("2016-04-16")
eq_end   = pd.Timestamp("2016-04-30")
df["is_earthquake"] = ((df["date"] >= eq_start) & (df["date"] <= eq_end)).astype(int)

print("日期特徵建立完成")

# =============================================================================
# 6. 油價特徵
# =============================================================================
oil_full = oil_full.sort_values("date").reset_index(drop=True)

# [INSIGHT] 單日油價雜訊大（週內波動）；rolling 平均更能反映中期經濟趨勢。
# oil_ma_7 捕捉週內趨勢，oil_ma_28 捕捉月度趨勢。
oil_full["oil_ma_7"]  = oil_full["dcoilwtico"].rolling(7,  min_periods=1).mean()
oil_full["oil_ma_28"] = oil_full["dcoilwtico"].rolling(28, min_periods=1).mean()

df = df.merge(
    oil_full[["date", "dcoilwtico", "oil_ma_7", "oil_ma_28"]],
    on="date", how="left"
)
print("油價特徵建立完成")

# =============================================================================
# 7. 假日特徵
# =============================================================================

# ── 7a. Store locale 對齊 ────────────────────────────────────────────────────
# [INSIGHT] 假日影響範圍因 locale 不同：
#   National → 全 54 家店
#   Regional → 只影響特定 state 的店（如 Pichincha 的假日不影響 Guayas）
#   Local    → 只影響特定 city 的店
# 若不做 locale 對齊，Quito 的地方假日會被誤標到全國所有店。
store_city  = stores.set_index("store_nbr")["city"].to_dict()
store_state = stores.set_index("store_nbr")["state"].to_dict()

national_dates  = set(holidays_clean[holidays_clean["locale"] == "National"]["date"])
regional_df     = holidays_clean[holidays_clean["locale"] == "Regional"][["date", "locale_name"]]
local_df        = holidays_clean[holidays_clean["locale"] == "Local"][["date", "locale_name"]]

regional_set = set(zip(regional_df["date"], regional_df["locale_name"]))
local_set    = set(zip(local_df["date"],    local_df["locale_name"]))

df["_store_state"] = df["store_nbr"].map(store_state)
df["_store_city"]  = df["store_nbr"].map(store_city)

df["is_national_holiday"] = df["date"].isin(national_dates).astype(int)
df["is_regional_holiday"] = [
    1 if (d, s) in regional_set else 0
    for d, s in zip(df["date"], df["_store_state"])
]
df["is_local_holiday"] = [
    1 if (d, c) in local_set else 0
    for d, c in zip(df["date"], df["_store_city"])
]
df["is_holiday"] = (
    (df["is_national_holiday"] + df["is_regional_holiday"] + df["is_local_holiday"]) > 0
).astype(int)

# ── 7b. 假日類型 encoding ────────────────────────────────────────────────────
# [INSIGHT] Holiday / Event / Bridge / Work Day 對銷售的影響方向不同：
# Holiday 銷售可能下降（商店關閉）或上升（提前採購）；
# Work Day (補班日) 對銷售影響相反；Bridge 有跨假日效應。
# 類型 encoding 讓模型分別學習各類假日的效應。
htype_map = (holidays_clean
             .sort_values("locale")  # National 優先（排序：Local < National < Regional）
             .groupby("date")["type"]
             .last()
             .to_dict())
df["holiday_type"] = df["date"].map(htype_map).fillna("None")
le_htype = LabelEncoder()
df["holiday_type_enc"] = le_htype.fit_transform(df["holiday_type"])

# ── 7c. 距離最近假日天數（National only）────────────────────────────────────
# [INSIGHT] 節前 N 天有搶購效應，節後 N 天有回落效應，
# 兩個方向的特徵都有助於捕捉這段動態。clip(0,28) 超過 4 週視為無關。
nat_hol_df = pd.DataFrame({"hol_date": sorted(national_dates)})
date_df = (df[["date"]].drop_duplicates()
                        .sort_values("date")
                        .reset_index(drop=True))

# 距上次假日天數
tmp_back = pd.merge_asof(
    date_df,
    nat_hol_df.rename(columns={"hol_date": "last_hol"}),
    left_on="date", right_on="last_hol",
    direction="backward"
)
tmp_back["days_after_holiday"] = (
    (tmp_back["date"] - tmp_back["last_hol"]).dt.days
     .clip(0, 28)
     .fillna(28)
     .astype(int)
)

# 距下次假日天數
tmp_fwd = pd.merge_asof(
    date_df,
    nat_hol_df.rename(columns={"hol_date": "next_hol"}),
    left_on="date", right_on="next_hol",
    direction="forward"
)
tmp_fwd["days_to_holiday"] = (
    (tmp_fwd["next_hol"] - tmp_fwd["date"]).dt.days
     .clip(0, 28)
     .fillna(28)
     .astype(int)
)

date_hol = date_df.copy()
date_hol["days_after_holiday"] = tmp_back["days_after_holiday"].values
date_hol["days_to_holiday"]    = tmp_fwd["days_to_holiday"].values

df = df.merge(date_hol, on="date", how="left")
print("假日特徵建立完成")

# =============================================================================
# 8. Lag & Rolling 特徵
# =============================================================================
print("計算 lag/rolling 特徵（最耗時步驟）...")

# ── 8a. Sales lag ────────────────────────────────────────────────────────────
# [INSIGHT] 最小 lag = 16，原因：test set 預測 Aug 16–31（16 天）。
# 對 Aug 31 而言，lag_16 = Aug 15（最後一筆 train 資料），剛好可用。
# lag_7 / lag_14 在部分 test 日期會看到未知的未來值 → 嚴格禁止。
LAG_LIST = [16, 21, 28, 35, 42]
for lag in LAG_LIST:
    col = f"sales_lag_{lag}"
    df[col] = df.groupby(["store_nbr", "family"])["sales"].shift(lag)
    print(f"  {col} 完成")

# ── 8b. Rolling mean / std（以 lag_16 為基準窗口）────────────────────────────
# [INSIGHT] rolling(w).mean().shift(16)：在時間點 t，取 [t-16-w+1, ..., t-16] 的均值。
# 這保證窗口內所有資料在預測時都是已知的，不存在 leakage。
for window in [7, 14, 28]:
    col = f"sales_mean_{window}_lag16"
    df[col] = (df.groupby(["store_nbr", "family"])["sales"]
                 .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(16)))
    print(f"  {col} 完成")

df["sales_std_7_lag16"] = (
    df.groupby(["store_nbr", "family"])["sales"]
      .transform(lambda x: x.rolling(7, min_periods=2).std().shift(16))
)
print("  sales_std_7_lag16 完成")

# ── 8c. Transactions lag ─────────────────────────────────────────────────────
# [INSIGHT] transactions 與 sales 高度相關（r=0.68），但 test set 沒有當日值。
# 只能用 lag 版本；lag_16 確保所有 test dates 都有對應的歷史值。
trx = (transactions
       .groupby(["store_nbr", "date"])["transactions"]
       .sum()
       .reset_index())
df = df.merge(trx, on=["store_nbr", "date"], how="left")

for lag in [7, 14, 16]:
    col = f"transactions_lag_{lag}"
    df[col] = df.groupby("store_nbr")["transactions"].shift(lag)
    print(f"  {col} 完成")

# [INSIGHT] transactions 資料並非每家店每天都有記錄（部分日期無交易），
# NaN 填 0 比填均值保守且不造成 leakage；模型對 0 的處理等價於「當日無資料」。
for lag in [7, 14, 16]:
    df[f"transactions_lag_{lag}"] = df[f"transactions_lag_{lag}"].fillna(0)

# ── 8d. Onpromotion lag ─────────────────────────────────────────────────────
# [INSIGHT] onpromotion 本身在 test set 已知（直接用）。
# 但加入歷史促銷 lag 有助於捕捉「促銷慣性」：
# 長期高促銷的品項，其基礎銷量本就較高。
df["promo_lag_7"] = df.groupby(["store_nbr", "family"])["onpromotion"].shift(7)
df["promo_ma_7"]  = (df.groupby(["store_nbr", "family"])["onpromotion"]
                       .transform(lambda x: x.rolling(7, min_periods=1).mean().shift(1)))
print("  promo lag/ma 完成")

# =============================================================================
# 9. 整理欄位、拆分 train / test
# =============================================================================

# 最終特徵清單
FEATURE_COLS = [
    # store 靜態
    "store_nbr", "family_enc", "type_enc", "cluster", "city_enc", "state_enc",
    # 日期
    "day_of_week", "day_of_month", "month", "year", "week_of_year", "quarter",
    "is_weekend", "is_month_start", "is_eom", "is_earthquake",
    # 油價
    "dcoilwtico", "oil_ma_7", "oil_ma_28",
    # 假日
    "is_holiday", "is_national_holiday", "is_regional_holiday", "is_local_holiday",
    "holiday_type_enc", "days_after_holiday", "days_to_holiday",
    # 促銷
    "onpromotion", "promo_lag_7", "promo_ma_7",
    # sales lag
    "sales_lag_16", "sales_lag_21", "sales_lag_28", "sales_lag_35", "sales_lag_42",
    # sales rolling
    "sales_mean_7_lag16", "sales_mean_14_lag16", "sales_mean_28_lag16", "sales_std_7_lag16",
    # transactions lag
    "transactions_lag_7", "transactions_lag_14", "transactions_lag_16",
]

# 拆分
train_fe = df[df["is_train"] == 1].copy()
test_fe  = df[df["is_train"] == 0].copy()

# train 去掉 lag 不足的早期資料（42 天後才有完整 lag_42）
# [INSIGHT] 前 42 筆 lag 必然是 NaN，保留會讓模型學到錯誤的空值規律。
# 42 天 ≈ 1782 series × 42 rows = ~75K rows，佔總訓練資料 < 3%，捨棄影響不大。
min_date = train["date"].min() + pd.Timedelta(days=42)
train_fe = train_fe[train_fe["date"] >= min_date]

print(f"\n訓練集：{len(train_fe):,} 筆（移除前 42 天後）")
print(f"測試集：{len(test_fe):,} 筆")

# 確認 test 的 lag 特徵無 NaN
lag_cols = [c for c in FEATURE_COLS if "lag" in c or "mean" in c or "std" in c]
test_null = test_fe[lag_cols].isnull().sum()
test_null_any = test_null[test_null > 0]
if len(test_null_any):
    print("\n⚠ Test set lag 特徵有缺值：")
    print(test_null_any)
else:
    print("\nTest set lag 特徵無缺值 ✓")

# train lag 特徵的缺值比例
train_null_pct = train_fe[lag_cols].isnull().mean() * 100
significant = train_null_pct[train_null_pct > 5]
if len(significant):
    print("\nTrain lag 特徵缺值比例 > 5%（正常，為早期資料）：")
    print(significant.round(1))

# =============================================================================
# 10. 儲存
# =============================================================================
# store_nbr / family 已在 FEATURE_COLS 內（store_nbr 直接當數值特徵，family_enc 為 encoding），
# 額外保留 date / family（原始字串）方便後續分析，用 dict.fromkeys 去重保持順序。
save_cols_train = list(dict.fromkeys(["id", "date", "store_nbr", "family", "sales"] + FEATURE_COLS))
save_cols_test  = list(dict.fromkeys(["id", "date", "store_nbr", "family"] + FEATURE_COLS))

train_fe[save_cols_train].to_parquet(OUT / "train_fe.parquet", index=False)
test_fe[save_cols_test].to_parquet(OUT / "test_fe.parquet",   index=False)

print(f"\n儲存完成：")
print(f"  outputs/train_fe.parquet  → {len(train_fe):,} rows × {len(save_cols_train)} cols")
print(f"  outputs/test_fe.parquet   → {len(test_fe):,} rows × {len(save_cols_test)} cols")

# =============================================================================
# 11. 特徵摘要
# =============================================================================
print("\n" + "=" * 60)
print("特徵摘要")
print("=" * 60)
print(f"總特徵數：{len(FEATURE_COLS)}")
print("\n類別特徵（需告知 LightGBM）：")
cat_cols = ["store_nbr", "family_enc", "type_enc", "cluster",
            "city_enc", "state_enc", "holiday_type_enc",
            "day_of_week", "month", "quarter"]
print(" ", cat_cols)

print("\nTrain 各特徵缺值率（Top 10）：")
null_rates = train_fe[FEATURE_COLS].isnull().mean().sort_values(ascending=False).head(10)
print(null_rates[null_rates > 0].round(4).to_string())
