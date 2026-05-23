"""
19_zero_window_search.py — 測試不同 zero-sales 窗口大小
=========================================================
以 submission_v14.csv 為基礎，試 7 / 14 / 21 / 28 / 35 天窗口，
各產出一個 CSV，方便逐一提交找最佳值。
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("outputs")

print("載入資料...")
train = pd.read_parquet(OUT / "train_fe.parquet")
test  = pd.read_parquet(OUT / "test_fe.parquet")
sub   = pd.read_csv("submission_v14.csv")

test_keys = test[["id", "store_nbr", "family_enc"]].copy()
sub_merged = sub.merge(test_keys, on="id")

TRAIN_END = pd.Timestamp("2017-08-15")
WINDOWS   = [7, 14, 21, 28, 35]

print(f"\n{'窗口':>6}  {'Zero series':>12}  {'歸零筆數':>10}  {'檔名'}")
print("-" * 55)

for w in WINDOWS:
    cutoff   = TRAIN_END - pd.Timedelta(days=w - 1)
    last_n   = train[train["date"] >= cutoff][["store_nbr", "family_enc", "sales"]]
    zero_mask = (
        last_n.groupby(["store_nbr", "family_enc"])["sales"]
        .apply(lambda x: (x == 0).all())
    )
    zero_pairs = set(zero_mask[zero_mask].index.tolist())

    mask = pd.Series([
        (int(r.store_nbr), int(r.family_enc)) in zero_pairs
        for r in sub_merged.itertuples()
    ], index=sub_merged.index)

    tmp = sub_merged.copy()
    tmp.loc[mask, "sales"] = 0.0
    out = tmp[["id", "sales"]].sort_values("id").reset_index(drop=True)

    fname = f"submission_v15_w{w}.csv"
    out.to_csv(fname, index=False)

    tag = " ← v15（已知）" if w == 21 else ""
    print(f"  w={w:2d}   {len(zero_pairs):>8} 組    {mask.sum():>8} 筆   {fname}{tag}")
