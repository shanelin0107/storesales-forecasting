# Store Sales - Time Series Forecasting｜專案指引

## 競賽概覽

- **平台**：Kaggle — Store Sales - Time Series Forecasting
- **資料來源**：厄瓜多連鎖超市 Corporación Favorita（54 家分店，33 種商品類別）
- **預測目標**：2017/8/16 ~ 2017/8/31 各分店各類別的銷售量（`sales`）
- **評估指標**：RMSLE（Root Mean Squared Logarithmic Error）— 越低越好

## 專案目標

1. **以高分上排行榜為主要目標**，不以快速完成為優先
2. 所有關鍵設計決策，須在程式碼中以 `# [INSIGHT]` 標記，說明：
   - **為什麼這樣做**（背後的數據/統計/業務理由）
   - **這個設計預期帶來什麼效果**（對 RMSLE 的影響方向）

---

## 建模策略

### 執行順序
1. **Naive Seasonal Baseline**：以去年同期銷售量作為預測，建立最低標準（benchmark）
2. **LightGBM Global Model**：主力模型，全部 store × family 一起訓練
3. **擴展試驗**：XGBoost、CatBoost
4. **Ensemble / Stacking**：視個別模型分數再決定是否組合

### Global Model 的理由
- 54 家店 × 33 類別 = 1782 條時間序列，資料量足夠支撐 global model
- Global model 能跨序列學習共同模式（如節假日效應、油價影響），比 per-series 模型更具泛化力
- LightGBM 對 tabular 時間序列特徵處理效率高，適合本競賽規模

### Target 轉換
- 訓練前對 `sales` 做 `log1p` 轉換，預測後 `expm1` 還原
- **理由**：用 MSE loss 訓練 log1p(sales) 在數學上等價於直接優化 RMSLE，同時壓縮右偏分佈、減少大值異常點的影響

---

## 特徵工程指引

### 日期特徵
- 星期幾（day_of_week）
- 月份（month）
- 週幾（week_of_year）
- 是否週末（is_weekend）
- **月底 flag** (`is_eom`)：月底發薪當日銷售開始回升（+2~9%）
- **月初 flag** (`is_month_start`, Day 1–3)：月底薪資的滯後消費，銷售高出均值 12–18%，是全月最強信號
- `day_of_month` 直接作為數值特徵（讓模型學習全月的非線性消費曲線）
- **注意**：`is_15th`（15 日薪資日）在控制星期效應後幾乎無影響（-2.6%），**不加入特徵**

### Lag Features（以 store × family 為單位計算）
- lag_7、lag_14、lag_16、lag_21、lag_28
- **理由**：超市銷售具有強週期性（7 的倍數），lag_16 對齊 test set 預測起點

### Rolling Statistics
- 7 天、14 天、28 天滾動平均（rolling_mean）
- 7 天滾動標準差（rolling_std）— 捕捉近期波動

### 節假日特徵（來自 `holidays_events.csv`）
- `is_holiday`：當日是否為假日
- `is_transferred`：移轉假日視為正常上班日，需特別標記
- `holiday_type`：Holiday / Bridge / Event / Work Day 等
- `locale`：National / Regional / Local（需與 store 的 city/state 對齊）
- `days_to_holiday`、`days_after_holiday`：假日前後 N 天效應
- **地震 dummy**：2016/4/16 及其後 ~2 週，銷售行為明顯異常

### 油價特徵（來自 `oil.csv`）
- 缺值用線性插值（`interpolate`）補齊（非交易日無報價）
- `oil_7d_avg`：7 天滾動平均油價 — 平滑短期波動

### 交易筆數特徵（來自 `transactions.csv`）
- `transactions_lag_7`、`transactions_lag_14`
- **理由**：transactions 與 sales 高度相關，但 test set 無此欄位，只能用 lag

### 分店與商品類別特徵（來自 `stores.csv`）
- `type`、`cluster`：Label Encoding
- `family`：Label Encoding
- `city`、`state`：Label Encoding（搭配 locale 做假日匹配）

### `onpromotion` 特徵
- 直接使用（test set 已提供此欄位）
- 可加 lag / rolling 版本捕捉促銷慣性

---

## 驗證策略（Cross-Validation）

### 原則
- **絕對不能 random shuffle**，必須保持時間順序
- 使用 **Walk-Forward CV**（time-based split）
- 每個 fold 的 validation 長度對齊 test set（16 天）

### 建議 Fold 設定
| Fold | Train 結束 | Validation 期間 |
|------|-----------|----------------|
| 1    | 2017-06-30 | 2017-07-01 ~ 07-16 |
| 2    | 2017-07-15 | 2017-07-16 ~ 07-31 |
| 3    | 2017-07-31 | 2017-08-01 ~ 08-15 |

Fold 3 最接近真實 test set，權重可加重。

### 評估
- 每個 fold 計算 RMSLE，取平均與標準差
- 若 CV score 與 LB score 差距大，優先排查 data leakage 或 CV 設計問題

---

## 程式碼規範

- 語言：**Python**
- 關鍵步驟必須加 `# [INSIGHT] ...` 說明設計理由與預期效果
- 特徵工程、模型訓練、驗證各自獨立為函式或 section，方便迭代
- 每次實驗記錄：模型版本、CV RMSLE、LB score、主要變動

---

## 迭代方向（優先順序）

1. 建立 baseline → 確認 CV pipeline 正確
2. 加入完整特徵工程 → LightGBM v1
3. 參數調整（num_leaves、learning_rate、feature_fraction 等）
4. 嘗試 XGBoost / CatBoost
5. Ensemble（weighted average 或 stacking）

---

## 注意事項

- `sales` 可為 0（正常），不可為負數，預測後需 clip 至 0 以上
- `onpromotion` 在 test set 已知，是重要的 forward-looking 特徵
- 油價資料有週末缺值，補值方式影響特徵品質
- 移轉假日（`transferred=True`）的日期本身**不是假日**，原假日才是，需正確解析
- 地震期間（2016/4/16 前後）為異常值，考慮加 dummy 或降低該期間權重
