"""
18_zero_sales_postprocess.py — Zero-Sales 後處理 → submission_v15.csv
======================================================================
套用在 submission_v14.csv 上，不需重新訓練。

[INSIGHT] 為何有效：
  若某 store×family 在預測前 21 天銷售全為 0，該商品在該店極可能已下架或缺貨。
  模型仍會預測出小正數（受其他店或歷史資料影響），但正確答案是 0。
  直接歸零可消除這批系統性正偏差，改善 RMSLE。
  21 天 = 3 週，足夠排除短暫缺貨，確認是持續零售狀態。
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("outputs")

print("載入資料...")
train = pd.read_parquet(OUT / "train_fe.parquet")
test  = pd.read_parquet(OUT / "test_fe.parquet")
sub   = pd.read_csv("submission_v14.csv")

# =============================================================================
# 找出「最後 21 天全為 0 銷售」的 store×family 組合
# =============================================================================
# train["sales"] 為 log1p 空間，log1p(0) = 0.0
ZERO_WINDOW = 21
cutoff = pd.Timestamp("2017-08-15") - pd.Timedelta(days=ZERO_WINDOW - 1)  # 2017-07-26

last_n = train[train["date"] >= cutoff][["store_nbr", "family_enc", "sales"]]

zero_mask = (
    last_n.groupby(["store_nbr", "family_enc"])["sales"]
    .apply(lambda x: (x == 0).all())
)
zero_pairs = set(zero_mask[zero_mask].index.tolist())
print(f"Zero series（最後 {ZERO_WINDOW} 天全為 0）：{len(zero_pairs)} 個 store×family 組合")

# =============================================================================
# 套用後處理
# =============================================================================
test_keys = test[["id", "store_nbr", "family_enc"]].copy()
sub_merged = sub.merge(test_keys, on="id")

mask = pd.Series([
    (int(r.store_nbr), int(r.family_enc)) in zero_pairs
    for r in sub_merged.itertuples()
], index=sub_merged.index)

print(f"歸零預測筆數：{mask.sum()} / {len(sub_merged)}")

sub_merged.loc[mask, "sales"] = 0.0

sub_out = sub_merged[["id", "sales"]].sort_values("id").reset_index(drop=True)
sub_out.to_csv("submission_v15.csv", index=False)

print(f"\nsubmission_v15.csv 儲存完成（{len(sub_out):,} 筆）")
print(f"  v14 median : {sub['sales'].median():.3f}  zeros={(sub['sales']==0).sum()}")
print(f"  v15 median : {sub_out['sales'].median():.3f}  zeros={(sub_out['sales']==0).sum()}")

# 顯示哪些 family 被歸零最多
zero_by_family = (
    sub_merged[mask]
    .groupby("family_enc")["id"].count()
    .rename("zeroed_rows")
    .sort_values(ascending=False)
)
print(f"\n被歸零最多的 family（top 10）：")
print(zero_by_family.head(10).to_string())
