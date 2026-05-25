# 競賽進度記錄

## 競賽資訊
- **平台**：Kaggle — Store Sales - Time Series Forecasting
- **資料**：54 家店 × 33 種商品，預測 2017/8/16~8/31（16 天）
- **指標**：RMSLE（越低越好）

---

## 成績歷程

| 版本 | 腳本 | 架構 | CV | LB | 主要改動 |
|------|------|------|----|----|---------|
| v3 | 08_model_v3.py | Global LightGBM | 0.38619 | 0.430 | 基礎特徵工程 |
| v10 | 13_year_lag.py | Global + 年度 lag | 0.38742 | 0.43254 | 加年度 lag + 限 2015 起資料（退步）|
| v11 | 14_per_family.py | Per-Family × 33 | 0.38806 | 0.42084 | 改成 33 個獨立模型 |
| v12 | 15_per_family_dense_lags.py | Per-Family + Dense 16~63 | 0.38707 | 0.41781 | 連續 lag_16~63 |
| v13 | 16_per_family_lag168.py | Per-Family + Dense 16~168 | 0.38881 | 0.41895 | 延伸至 168（退步，確認甜蜜點是 16~63）|
| v14 | 17_recursive_forecast.py | Per-Family + Recursive | — | 0.397 | 遞迴預測，加入 lag_1~15 |
| **v15** | 18_zero_sales_postprocess.py | v14 + Zero-Sales Filter | — | **0.38465** | 後處理：最後 3~7 天全零→預測 0 |

**當前最佳：v15，LB = 0.38465**

---

## 關鍵發現

1. **Per-Family > Global**：33 種商品模式差異大，獨立模型更準
2. **Dense lag 甜蜜點 = 16~63**：64 以上是不同季節的雜訊
3. **Recursive Forecasting 是最大單次跳躍**：解鎖 lag_7，LB 從 0.41781 → 0.397
4. **Zero-Sales Filter**：最後 3~7 天全零的 series 直接預測 0，LB 再降 0.012
5. **限制訓練資料到 2015+ 反而退步**：樣本量比「資料新鮮度」更重要
6. **CV ≈ 0.387，LB ≈ 0.384**：有系統性差距，CV 趨勢仍可信

---

## Zero-Sales Window 搜尋結果

| 窗口 | LB |
|------|----|
| 35 天 | 未測 |
| 28 天 | 未測 |
| 21 天 | 0.38695 |
| 14 天 | 0.38639 |
| 7 天 | 0.38467 |
| **3 天** | **0.38465** ← 最佳 |
| 2、1 天 | 未測（預期邊際效益小）|

---

## 已排除的方向

- **Tweedie loss**（v9）：CV 更差，放棄
- **Optuna 調參**（v8）：邊際改善不顯著
- **lag_64~168**（v13）：確認是雜訊，不值得

---

## 下一步待試（優先順序）

### 高優先
1. **Full + 2015-only 模型平均**（競賽者做法）
   - 用相同 v14 架構，只限 2015 起資料再訓練一組模型
   - 兩組預測取平均，再套 zero-sales filter
   - 目標：互補偏差，預計可再降 0.003~0.005

### 中優先
2. **Ensemble v12 + v15**
   - 直接平均兩個不同架構的 CSV，不需重新訓練
   - 直接預測 vs 遞迴預測可能互補

3. **Zero-sales window w=1、w=2** 補測
   - 預期效益小，但成本低（不需重新訓練）

### 低優先
4. **特徵工程深化**
   - 促銷 lead time 特徵（onpromotion 的前幾天效應）
   - Store cluster × family 交互特徵

---

## 與競賽目標的差距

- 當前最佳：**0.38465**
- 競賽 0.37 分者：**0.37984**
- 差距：**0.005**
- 競賽者做了但我們還沒試：Full + 2015-only 模型平均

---

## 專案架構

```
02_feature_engineering.py   → outputs/train_fe.parquet, test_fe.parquet
17_recursive_forecast.py    → submission_v14.csv（最終模型）
18_zero_sales_postprocess.py → submission_v15.csv（當前最佳）
19_zero_window_search.py    → 搜尋最佳 window size
20_visualizations.py        → images/*.png
```

## 如何繼續
下次對話時告訴 Claude：
「讀一下 PROGRESS.md，我要繼續優化 Store Sales 競賽」
