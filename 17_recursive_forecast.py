"""
17_recursive_forecast.py — Per-Family + Dense Lags 1~63 + Recursive Forecasting → submission_v14.csv
=====================================================================================================
改動（相對 v12）：
1. DENSE_LAGS 從 lag_16~63 擴充至 lag_1~63（多 15 個短期 lag）
2. 測試集預測改為逐天遞迴：先預測 Aug16，結果存入 buffer，作為 Aug17 的 lag_1 使用
3. CV 驗證也採用遞迴模式（正確模擬 test 情境，而非用 oracle lag）

[INSIGHT] 為何加入 lag_1~15（遞迴）效果應更好：
  - lag_7 是超市銷售最強的短期週期特徵（週間重複模式）
  - v12 被迫從 lag_16 開始，是因為直接預測時 test 最早日 Aug16 的 lag_7 = Aug9 雖可用，
    但 Aug23 之後 lag_7 就落入 test 期間了，不能直接用
  - 遞迴方案：逐天預測，把前一天的預測結果填入 buffer 供後續 lag 使用
  - 代價：預測誤差會向後傳播（lag_7 帶入誤差），但 0.37 分競賽者採用相同策略
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
# 3. 計算 lag_1~63 + 年度 lag（train+test 合併）
# =============================================================================
print("計算 dense lags (1~63) + 年度 lag 特徵...")

combined = pd.concat([
    train[["date", "store_nbr", "family_enc", "sales"]],
    test[["date", "store_nbr", "family_enc"]].assign(sales=np.nan),
], ignore_index=True).sort_values(["store_nbr", "family_enc", "date"])

# [INSIGHT] 加入 lag_1~15：訓練時用真實歷史值學習週期模式；
# 推論時前 7 天（Aug16~22）lag_7 仍為真實值，Aug23 後才需遞迴填補
DENSE_LAGS = list(range(1, 64))           # lag_1~63，共 63 個
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

# 先移除 parquet 中已有的同名欄位
train = train.drop(columns=[c for c in ALL_NEW_COLS if c in train.columns])
test  = test.drop(columns=[c for c in ALL_NEW_COLS if c in test.columns])

# 訓練資料合併所有 lag（包含 dense lags，作為訓練特徵）
train = train.merge(combined[merge_keys + ALL_NEW_COLS], on=merge_keys, how="left")
# 測試資料只合併年度 lag（dense lags 在遞迴預測時動態填入）
test  = test.merge(combined[merge_keys + YEAR_LAG_COLS],  on=merge_keys, how="left")

# NaN 填補
for col in ALL_NEW_COLS:
    med = train[col].median()
    train[col] = train[col].fillna(med)
for col in YEAR_LAG_COLS:
    med = train[col].median()
    test[col] = test[col].fillna(med)

# 儲存 dense lag 的 median（遞迴預測中 buffer 缺值時的 fallback）
DENSE_LAG_MEDIANS = {col: float(train[col].median()) for col in DENSE_LAG_COLS}

print(f"Dense lag 特徵：lag_1~63（{len(DENSE_LAG_COLS)} 個）")
print(f"年度 lag 特徵：{len(YEAR_LAG_COLS)} 個")

# =============================================================================
# 4. 特徵定義
# =============================================================================
FEATURE_COLS = [
    # store 靜態
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
    # rolling stats（基準為 lag_16，test 期間 lag_16 永遠是訓練資料）
    "sales_mean_7_lag16", "sales_mean_14_lag16", "sales_mean_28_lag16", "sales_std_7_lag16",
    # transactions
    "transactions_lag_7", "transactions_lag_14", "transactions_lag_16",
    # dense lags 1~63（遞迴填入）
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
# 6. Helper：建立 sales buffer
# =============================================================================
def build_buffer(source_df, up_to_date, buffer_days=63):
    """
    從 source_df 中取最近 buffer_days 天的 sales，
    建立 dict: (date, store_nbr, family_enc) -> log1p(sales)
    """
    start = pd.Timestamp(up_to_date) - pd.Timedelta(days=buffer_days - 1)
    recent = source_df[
        (source_df["date"] >= start) & (source_df["date"] <= up_to_date)
    ][["date", "store_nbr", "family_enc", "sales"]]
    return recent.set_index(["date", "store_nbr", "family_enc"])["sales"].to_dict()


def fill_dense_lags_recursive(rows, pred_date, fam, buffer):
    """
    用 buffer 動態填入 rows 的 sales_lag_1~63 欄位。
    buffer 包含訓練期真實值 + 已預測天的預測值（log1p 空間）。
    """
    rows = rows.copy()
    for lag in DENSE_LAGS:
        lag_date = pred_date - pd.Timedelta(days=lag)
        rows[f"sales_lag_{lag}"] = [
            buffer.get((lag_date, int(s), int(fam)), DENSE_LAG_MEDIANS[f"sales_lag_{lag}"])
            for s in rows["store_nbr"].values
        ]
    return rows

# =============================================================================
# 7. Walk-Forward CV（遞迴預測）
# =============================================================================
print("\n" + "="*60)
print("Walk-Forward CV（3 folds，per-family，遞迴預測）")
print("="*60)

cv_scores      = []
all_best_iters = []

for fold_name, train_end, val_start, val_end in FOLDS:
    # --- 7a. 訓練各 family 模型（early stopping 用 oracle lags 加速）---
    fold_models = {}
    fold_iters  = []

    for fam in families:
        tr_fam  = train[(train["family_enc"] == fam) & (train["date"] <= train_end)]
        val_fam = train[
            (train["family_enc"] == fam) &
            (train["date"] >= val_start) & (train["date"] <= val_end)
        ]

        X_tr  = tr_fam[FEATURE_COLS]
        y_tr  = tr_fam[TARGET]
        # [INSIGHT] early stopping 使用 oracle lag（實際值），目的是快速收斂找最佳迭代數；
        # 真實 CV score 則由後續遞迴推論計算，兩者分離
        X_val = val_fam[FEATURE_COLS]
        y_val = val_fam[TARGET]

        ds_tr  = lgb.Dataset(X_tr, label=y_tr,  categorical_feature=CAT_COLS, free_raw_data=False)
        ds_val = lgb.Dataset(X_val, label=y_val, categorical_feature=CAT_COLS, free_raw_data=False)

        model = lgb.train(
            PARAMS, ds_tr,
            num_boost_round=3000,
            valid_sets=[ds_val],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(99999)],
        )
        fold_models[fam] = model
        fold_iters.append(model.best_iteration)

    # --- 7b. 遞迴預測 val 期間，正確模擬 test 情境 ---
    fold_buffer = build_buffer(train, train_end, buffer_days=63)

    val_preds_all = []
    val_true_all  = []
    val_dates     = sorted(
        train[(train["date"] >= val_start) & (train["date"] <= val_end)]["date"].unique()
    )

    for val_date in val_dates:
        for fam in families:
            val_rows = train[
                (train["family_enc"] == fam) & (train["date"] == val_date)
            ]
            if len(val_rows) == 0:
                continue

            # 動態填入 dense lag（用 buffer，而非 oracle）
            val_rows = fill_dense_lags_recursive(val_rows, val_date, fam, fold_buffer)

            preds_log = fold_models[fam].predict(val_rows[FEATURE_COLS])
            val_preds_all.extend(preds_log.tolist())
            val_true_all.extend(val_rows[TARGET].values.tolist())

            # [INSIGHT] buffer 更新用「預測值」而非真實值，
            # 確保誤差累積與 test 情境一致
            for store, pred in zip(val_rows["store_nbr"].values, preds_log):
                fold_buffer[(val_date, int(store), int(fam))] = float(pred)

    score    = rmsle(np.array(val_true_all), np.array(val_preds_all))
    avg_iter = int(np.mean(fold_iters))
    cv_scores.append(score)
    all_best_iters.append(avg_iter)
    print(f"  [{fold_name}] RMSLE={score:.5f}  avg_iter={avg_iter}")

cv_mean         = np.mean(cv_scores)
global_avg_iter = int(np.mean(all_best_iters))
print(f"\n  CV Mean  : {cv_mean:.5f}  (v12: 0.38707, v3: 0.38619)")
print(f"  vs v12   : {(0.38707 - cv_mean)/0.38707*100:+.2f}%")
print(f"  平均最佳迭代數 : {global_avg_iter}")

# =============================================================================
# 8. Full Model Training
# =============================================================================
print("\n" + "="*60)
print("Full Model Training（per-family，遞迴）")
print("="*60)

full_iter = int(global_avg_iter * 1.1)
print(f"迭代數：{full_iter}  (avg={global_avg_iter} × 1.1)")

full_models = {}
for fam in families:
    tr_fam = train[train["family_enc"] == fam]
    ds_full = lgb.Dataset(
        tr_fam[FEATURE_COLS], label=tr_fam[TARGET],
        categorical_feature=CAT_COLS, free_raw_data=False,
    )
    model = lgb.train(
        PARAMS, ds_full,
        num_boost_round=full_iter,
        callbacks=[lgb.log_evaluation(99999)],
    )
    full_models[fam] = model
    print(f"  [family={fam}] done")

# =============================================================================
# 9. 遞迴預測 test set（逐天 Aug16 → Aug31）
# =============================================================================
print("\n遞迴預測 test set...")

TRAIN_END  = pd.Timestamp("2017-08-15")
test_buffer = build_buffer(train, TRAIN_END, buffer_days=63)

test_dates = sorted(test["date"].unique())
all_preds  = []

for test_date in test_dates:
    for fam in families:
        te_rows = test[(test["date"] == test_date) & (test["family_enc"] == fam)].copy()
        if len(te_rows) == 0:
            continue

        te_rows = fill_dense_lags_recursive(te_rows, test_date, fam, test_buffer)

        preds_log   = full_models[fam].predict(te_rows[FEATURE_COLS])
        preds_sales = np.clip(np.expm1(preds_log), 0, None)

        # 更新 buffer（log1p 空間，供後續天的 lag 使用）
        for store, pred in zip(te_rows["store_nbr"].values, preds_log):
            test_buffer[(test_date, int(store), int(fam))] = float(pred)

        all_preds.append(pd.DataFrame({"id": te_rows["id"].values, "sales": preds_sales}))

# =============================================================================
# 10. 儲存提交
# =============================================================================
submission = pd.concat(all_preds).sort_values("id").reset_index(drop=True)
submission.to_csv("submission_v14.csv", index=False)

preds_all = submission["sales"].values
print(f"\nsubmission_v14.csv 儲存完成 ({len(submission):,} 筆)")
print(f"  min={preds_all.min():.3f}  median={np.median(preds_all):.3f}  "
      f"max={preds_all.max():.1f}  負值={(preds_all<0).sum()}")

pred_by_date = (
    test[["id", "date"]].merge(submission, on="id")
    .groupby("date")["sales"].mean().round(1)
)
print("\n每日平均預測：")
print(pred_by_date.to_string())

# =============================================================================
# 11. 結果摘要
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  架構      : Per-Family（{len(families)} 個）+ Dense Lags 1~63（遞迴預測）")
print(f"  特徵數    : {len(FEATURE_COLS)}")
for i, (fname, *_) in enumerate(FOLDS):
    print(f"  [{fname}] RMSLE={cv_scores[i]:.5f}  iter={all_best_iters[i]}")
print(f"  CV Mean   : {cv_mean:.5f}  (v12: 0.38707, v3: 0.38619)")
print(f"  vs v12    : {(0.38707 - cv_mean)/0.38707*100:+.2f}%")
print(f"  Full iter : {full_iter}")
print(f"  → submission_v14.csv 已產出")
