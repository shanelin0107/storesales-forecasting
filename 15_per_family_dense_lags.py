"""
15_per_family_dense_lags.py — Per-Family + Dense Lags (16~63) → submission_v12.csv
====================================================================================
改動：lag_16~63 連續填滿（原本只有 lag_16,21,28,35,42 五個稀疏點）

[INSIGHT] 為何補滿 lag_16~63：
  - 競賽者 0.37 用 lags=63（連續 1~63 天），讓模型看到完整近兩個月每日變化
  - 我們原本只有 5 個稀疏點，模型無法學習 lag_42~lag_21 之間的趨勢變化
  - Per-Family 每個模型只有 ~88K 筆，加 48 個 lag 特徵記憶體完全可控
  - 補滿後模型可自行學習任意 moving average、斜率、週期等中程模式
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
OUT = Path("outputs")

# =============================================================================
# 1. 載入資料
# =============================================================================
print("載入資料...")
train = pd.read_parquet(OUT / "train_fe.parquet")
test  = pd.read_parquet(OUT / "test_fe.parquet")
print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")

# =============================================================================
# 2. 修正 test transactions 污染
# =============================================================================
test.loc[test["date"] >= "2017-08-23", "transactions_lag_7"]  = \
    test.loc[test["date"] >= "2017-08-23",  "transactions_lag_16"]
test.loc[test["date"] >= "2017-08-30", "transactions_lag_14"] = \
    test.loc[test["date"] >= "2017-08-30", "transactions_lag_16"]

# =============================================================================
# 3. 計算 lag_16~63 + 年度 lag（train+test 合併）
# =============================================================================
print("計算 dense lags (16~63) + 年度 lag 特徵...")

combined = pd.concat([
    train[["date", "store_nbr", "family_enc", "sales"]],
    test[["date", "store_nbr", "family_enc"]].assign(sales=np.nan),
], ignore_index=True).sort_values(["store_nbr", "family_enc", "date"])

# [INSIGHT] lag_16~63：連續 48 個 lag，讓模型自行組合成任意中程模式。
# 年度 lag 同 v10/v11，覆蓋去年同期信號。
DENSE_LAGS = list(range(16, 64))           # 16~63，共 48 個
YEAR_LAGS  = [364, 371, 728]
YEAR_ROLLS = [(7, 357), (7, 364), (28, 350), (28, 714)]

def add_all_lags(group):
    s = group["sales"]
    res = {}
    for lag in DENSE_LAGS:
        res[f"sales_lag_{lag}"] = s.shift(lag).values
    for lag in YEAR_LAGS:
        res[f"sales_lag_{lag}"] = s.shift(lag).values
    for window, lag in YEAR_ROLLS:
        res[f"sales_mean_{window}_lag{lag}"] = (
            s.shift(lag).rolling(window, min_periods=1).mean().values
        )
    return pd.DataFrame(res, index=group.index)

lag_feat_df = combined.groupby(
    ["store_nbr", "family_enc"], sort=False, group_keys=False
).apply(add_all_lags)

combined = pd.concat([combined, lag_feat_df], axis=1)

DENSE_LAG_COLS = [f"sales_lag_{l}" for l in DENSE_LAGS]
YEAR_LAG_COLS  = (
    [f"sales_lag_{l}" for l in YEAR_LAGS] +
    [f"sales_mean_{w}_lag{l}" for w, l in YEAR_ROLLS]
)
ALL_NEW_COLS = DENSE_LAG_COLS + YEAR_LAG_COLS

merge_keys = ["date", "store_nbr", "family_enc"]

# 先移除 parquet 中已有的同名欄位（避免 merge 產生 _x/_y 衝突）
train = train.drop(columns=[c for c in ALL_NEW_COLS if c in train.columns])
test  = test.drop(columns=[c for c in ALL_NEW_COLS if c in test.columns])

train = train.merge(combined[merge_keys + ALL_NEW_COLS], on=merge_keys, how="left")
test  = test.merge(combined[merge_keys + ALL_NEW_COLS], on=merge_keys, how="left")

# NaN 填補（訓練資料最早幾筆的 lag 超出範圍）
for col in ALL_NEW_COLS:
    med = train[col].median()
    train[col] = train[col].fillna(med)
    test[col]  = test[col].fillna(med)

print(f"Dense lag 特徵：lag_16~63（{len(DENSE_LAG_COLS)} 個）")
print(f"年度 lag 特徵：{len(YEAR_LAG_COLS)} 個")

# =============================================================================
# 4. 特徵定義
# =============================================================================
# [INSIGHT] 移除原本稀疏的 sales_lag_21/28/35/42（已被 dense lags 涵蓋）
# 保留 rolling stats（mean_7/14/28, std_7 on lag_16）作為顯式移動平均特徵
FEATURE_COLS = [
    # store 靜態（不含 family_enc）
    "store_nbr", "type_enc", "cluster", "city_enc", "state_enc",
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
    # rolling stats (lag_16 基準)
    "sales_mean_7_lag16", "sales_mean_14_lag16", "sales_mean_28_lag16", "sales_std_7_lag16",
    # transactions
    "transactions_lag_7", "transactions_lag_14", "transactions_lag_16",
    # dense lags 16~63
    *DENSE_LAG_COLS,
    # 年度 lag
    *YEAR_LAG_COLS,
]

CAT_COLS = [
    "store_nbr", "type_enc", "cluster",
    "city_enc", "state_enc", "holiday_type_enc",
    "day_of_week", "month", "quarter",
]

TARGET = "sales"
print(f"總特徵數：{len(FEATURE_COLS)}")

# =============================================================================
# 5. Fold & Params
# =============================================================================
FOLDS = [
    ("Fold1", "2017-06-30", "2017-07-01", "2017-07-16"),
    ("Fold2", "2017-07-15", "2017-07-16", "2017-07-31"),
    ("Fold3", "2017-07-31", "2017-08-01", "2017-08-15"),
]

PARAMS = {
    "boosting_type":     "gbdt",
    "objective":         "regression",
    "metric":            "rmse",
    "num_leaves":        511,
    "learning_rate":     0.03,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "min_child_samples": 20,
    "reg_alpha":         0.0,
    "reg_lambda":        0.0,
    "random_state":      42,
    "verbose":           -1,
    "n_jobs":            -1,
}

def rmsle(y_true_log, y_pred_log):
    return float(np.sqrt(np.mean((y_true_log - np.clip(y_pred_log, 0, None)) ** 2)))

families = sorted(train["family_enc"].unique())
print(f"Family 數量：{len(families)}")

# =============================================================================
# 6. Walk-Forward CV
# =============================================================================
print("\n" + "="*60)
print("Walk-Forward CV（3 folds，per-family，dense lags）")
print("="*60)

cv_scores    = []
all_best_iters = []

for fold_name, train_end, val_start, val_end in FOLDS:
    val_preds_all = []
    val_true_all  = []
    fold_iters    = []

    for fam in families:
        tr_fam = train[train["family_enc"] == fam]

        X_tr  = tr_fam[tr_fam["date"] <= train_end][FEATURE_COLS]
        y_tr  = tr_fam[tr_fam["date"] <= train_end][TARGET]
        X_val = tr_fam[(tr_fam["date"] >= val_start) & (tr_fam["date"] <= val_end)][FEATURE_COLS]
        y_val = tr_fam[(tr_fam["date"] >= val_start) & (tr_fam["date"] <= val_end)][TARGET]

        ds_tr  = lgb.Dataset(X_tr, label=y_tr,  categorical_feature=CAT_COLS, free_raw_data=False)
        ds_val = lgb.Dataset(X_val, label=y_val, categorical_feature=CAT_COLS, free_raw_data=False)

        model = lgb.train(
            PARAMS, ds_tr,
            num_boost_round=3000,
            valid_sets=[ds_val],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(99999)],
        )
        val_preds_all.extend(model.predict(X_val).tolist())
        val_true_all.extend(y_val.values.tolist())
        fold_iters.append(model.best_iteration)

    score    = rmsle(np.array(val_true_all), np.array(val_preds_all))
    avg_iter = int(np.mean(fold_iters))
    cv_scores.append(score)
    all_best_iters.append(avg_iter)
    print(f"  [{fold_name}] RMSLE={score:.5f}  avg_iter={avg_iter}")

cv_mean         = np.mean(cv_scores)
global_avg_iter = int(np.mean(all_best_iters))
print(f"\n  CV Mean  : {cv_mean:.5f}  (v11: 0.38806, v3: 0.38619)")
print(f"  vs v3    : {(0.38619 - cv_mean)/0.38619*100:+.2f}%")
print(f"  平均最佳迭代數 : {global_avg_iter}")

# =============================================================================
# 7. Full Model Training
# =============================================================================
print("\n" + "="*60)
print("Full Model Training（per-family，dense lags）")
print("="*60)

full_iter = int(global_avg_iter * 1.1)
print(f"迭代數：{full_iter}  (avg={global_avg_iter} × 1.1)")

all_preds = []

for fam in families:
    tr_fam = train[train["family_enc"] == fam]
    te_fam = test[test["family_enc"] == fam]

    ds_full = lgb.Dataset(
        tr_fam[FEATURE_COLS], label=tr_fam[TARGET],
        categorical_feature=CAT_COLS, free_raw_data=False,
    )
    model = lgb.train(
        PARAMS, ds_full,
        num_boost_round=full_iter,
        callbacks=[lgb.log_evaluation(99999)],
    )
    preds_log   = model.predict(te_fam[FEATURE_COLS])
    preds_sales = np.clip(np.expm1(preds_log), 0, None)
    all_preds.append(pd.DataFrame({"id": te_fam["id"].values, "sales": preds_sales}))
    print(f"  [family={fam}] done")

# =============================================================================
# 8. 儲存提交
# =============================================================================
submission = pd.concat(all_preds).sort_values("id").reset_index(drop=True)
submission.to_csv("submission_v12.csv", index=False)

preds_all = submission["sales"].values
print(f"\nsubmission_v12.csv 儲存完成 ({len(submission):,} 筆)")
print(f"  min={preds_all.min():.3f}  median={np.median(preds_all):.3f}  "
      f"max={preds_all.max():.1f}  負值={(preds_all<0).sum()}")

pred_by_date = (
    pd.DataFrame({"date": test["date"].values,
                  "sales": test["id"].map(submission.set_index("id")["sales"])})
    .groupby("date")["sales"].mean().round(1)
)
print("\n每日平均預測：")
print(pred_by_date.to_string())

# =============================================================================
# 9. 結果摘要
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  架構      : Per-Family（{len(families)} 個）+ Dense Lags 16~63")
print(f"  特徵數    : {len(FEATURE_COLS)}")
for i, (fname, *_) in enumerate(FOLDS):
    print(f"  [{fname}] RMSLE={cv_scores[i]:.5f}  iter={all_best_iters[i]}")
print(f"  CV Mean   : {cv_mean:.5f}  (v11: 0.38806, v3: 0.38619)")
print(f"  vs v3     : {(0.38619 - cv_mean)/0.38619*100:+.2f}%")
print(f"  Full iter : {full_iter}")
print(f"  → submission_v12.csv 已產出")
