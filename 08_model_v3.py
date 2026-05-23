"""
08_model_v3.py — 移除低效 year-ago 特徵，保留 sales_mean_56_lag16 → submission_v5.csv
=======================================================================================
v4 LB=0.432 退步原因分析：
  - year-ago 3 個特徵（sales_lag_364, mean_7_lag364, mean_28_lag364）gain 極低（366K~615K）
  - sales_mean_56_lag16 gain 高達 22.8M（第 3 名），值得保留
  - NaN 填補（前 364 天用 mean_28_lag16 代理）可能引入雜訊，干擾模型對年度模式的學習

本版修正：
  - 移除 3 個 year-ago 特徵
  - 保留 sales_mean_56_lag16（高重要性）
  - 特徵數：45 → 42
  - 其餘與 07_model_v2.py 完全相同（params、CV folds、transactions 修正）

可回復：若 LB 比 v3（0.43）更差，上傳 submission_v3.csv 即可。
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
OUT = Path("outputs")

# =============================================================================
# 1. 載入 v2 特徵資料（含 sales_mean_56_lag16）
# =============================================================================
print("載入 v2 資料...")
train = pd.read_parquet(OUT / "train_fe_v2.parquet")
test  = pd.read_parquet(OUT / "test_fe_v2.parquet")
print(f"Train: {len(train):,} rows × {train.shape[1]} cols")
print(f"Test : {len(test):,}  rows × {test.shape[1]}  cols")

# =============================================================================
# 2. 修正 test transactions 污染（同 05/07）
# =============================================================================
t7_before = (test["transactions_lag_7"] == 0).sum()
test.loc[test["date"] >= "2017-08-23", "transactions_lag_7"]  = \
    test.loc[test["date"] >= "2017-08-23",  "transactions_lag_16"]
test.loc[test["date"] >= "2017-08-30", "transactions_lag_14"] = \
    test.loc[test["date"] >= "2017-08-30", "transactions_lag_16"]
print(f"\ntransactions_lag_7 =0 修正：{t7_before} → {(test['transactions_lag_7']==0).sum()}")

# =============================================================================
# 3. 特徵定義（42 個：base 41 + sales_mean_56_lag16，移除 year-ago 3 個）
# =============================================================================
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
    # sales rolling（v1 原有）
    "sales_mean_7_lag16", "sales_mean_14_lag16", "sales_mean_28_lag16", "sales_std_7_lag16",
    # sales rolling（v2 新增，保留高重要性）
    # [INSIGHT] sales_mean_56_lag16 在 v4 中 gain=22.8M（第 3 名），明顯捕捉中長期趨勢。
    # 移除 year-ago 特徵（gain 僅 366K~615K），因它們在 LB 上未能泛化。
    "sales_mean_56_lag16",
    # transactions
    "transactions_lag_7", "transactions_lag_14", "transactions_lag_16",
]

CAT_COLS = [
    "store_nbr", "family_enc", "type_enc", "cluster",
    "city_enc", "state_enc", "holiday_type_enc",
    "day_of_week", "month", "quarter",
]

TARGET = "sales"

print(f"\n特徵數：{len(FEATURE_COLS)}（base 41 + sales_mean_56_lag16，移除 year-ago 3 個）")

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

FOLDS = [
    ("Fold1", "2017-06-30", "2017-07-01", "2017-07-16"),
    ("Fold2", "2017-07-15", "2017-07-16", "2017-07-31"),
    ("Fold3", "2017-07-31", "2017-08-01", "2017-08-15"),
]

def rmsle(y_true_log, y_pred_log):
    return float(np.sqrt(np.mean((y_true_log - np.clip(y_pred_log, 0, None)) ** 2)))

# =============================================================================
# 4. Walk-Forward CV
# =============================================================================
print("\n" + "="*60)
print("Walk-Forward CV（3 folds）")
print("="*60)

cv_scores, best_iters = [], []

for fold_name, train_end, val_start, val_end in FOLDS:
    X_tr  = train[train["date"] <= train_end][FEATURE_COLS]
    y_tr  = train[train["date"] <= train_end][TARGET]
    X_val = train[(train["date"] >= val_start) & (train["date"] <= val_end)][FEATURE_COLS]
    y_val = train[(train["date"] >= val_start) & (train["date"] <= val_end)][TARGET]

    ds_tr  = lgb.Dataset(X_tr,  label=y_tr,  categorical_feature=CAT_COLS, free_raw_data=False)
    ds_val = lgb.Dataset(X_val, label=y_val, categorical_feature=CAT_COLS, free_raw_data=False)

    model = lgb.train(
        PARAMS, ds_tr,
        num_boost_round=5000,
        valid_sets=[ds_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=500),
        ],
    )
    score     = rmsle(y_val.values, model.predict(X_val))
    best_iter = model.best_iteration
    cv_scores.append(score)
    best_iters.append(best_iter)
    print(f"  [{fold_name}] RMSLE={score:.5f}  best_iter={best_iter}")

cv_mean = np.mean(cv_scores)
cv_std  = np.std(cv_scores)
avg_iter = int(np.mean(best_iters))

print(f"\nCV Mean: {cv_mean:.5f} ± {cv_std:.5f}")
print(f"vs v3-base (0.38619) : {(0.38619 - cv_mean)/0.38619*100:+.2f}%")
print(f"vs v4-full (0.38614) : {(0.38614 - cv_mean)/0.38614*100:+.2f}%")
print(f"平均最佳迭代數: {avg_iter}")

# =============================================================================
# 5. Full Model Training
# =============================================================================
print("\n" + "="*60)
print("Full Model Training")
print("="*60)

full_iter = int(avg_iter * 1.1)
print(f"迭代數：{full_iter}  (avg={avg_iter} × 1.1)")

ds_full = lgb.Dataset(
    train[FEATURE_COLS], label=train[TARGET],
    categorical_feature=CAT_COLS, free_raw_data=False,
)
full_model = lgb.train(
    PARAMS, ds_full,
    num_boost_round=full_iter,
    callbacks=[lgb.log_evaluation(period=500)],
)
print("Full model 訓練完成")

# =============================================================================
# 6. 預測與提交
# =============================================================================
print("\n" + "="*60)
print("Generate submission_v5.csv")
print("="*60)

preds_log   = full_model.predict(test[FEATURE_COLS])
preds_sales = np.clip(np.expm1(preds_log), 0, None)

submission = pd.DataFrame({"id": test["id"].values, "sales": preds_sales})
submission.to_csv("submission_v5.csv", index=False)

print(f"submission_v5.csv 儲存完成 ({len(submission):,} 筆)")
print(f"  min={preds_sales.min():.3f}  median={np.median(preds_sales):.3f}  "
      f"max={preds_sales.max():.1f}  負值={(preds_sales<0).sum()}")

pred_by_date = (
    pd.DataFrame({"date": test["date"].values, "sales": preds_sales})
    .groupby("date")["sales"].mean().round(1)
)
print("\n每日平均預測（應連續）：")
print(pred_by_date.to_string())

# =============================================================================
# 7. 結果摘要
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  特徵數              : {len(FEATURE_COLS)}（base 41 + mean_56_lag16）")
for i, (fname, *_) in enumerate(FOLDS):
    print(f"  {fname} RMSLE      : {cv_scores[i]:.5f}  (best_iter={best_iters[i]})")
print(f"  CV Mean ± Std       : {cv_mean:.5f} ± {cv_std:.5f}")
print(f"  vs v3 (0.38619)     : {(0.38619-cv_mean)/0.38619*100:+.2f}%")
print(f"  Full model iter     : {full_iter}")
print(f"  → submission_v5.csv 已產出")
