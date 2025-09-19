# GRAFT: **G**RAPE **R**educers **A**dapters for **F**ormer/**T**opology (整合 GRAPE・LUNAR・T2G-FORMER)

---

## 🌱 專案簡述（Overview）

GRAFT 目標：**在不手動前處理/修剪資料**的前提下，
把三篇代表作 

- **GRAPE**（缺失值補全/標籤學習的二分圖 GNN）
- **LUNAR**（以 GNN 統一在地離群偵測）
- **T2G-FORMER**（把表格特徵重組成關係圖，促進異質特徵互動）

**組裝成單一可重現的流水線**，
並提供**隨機刪減**等對照組以做消融實驗，最後以統一指標與流程評估。

* 任務涵蓋：**標籤預測（Label Prediction）**。
* 設計原則：以 **模組化 Adapter/Selector/Builder** 連接三篇方法，避免侵入式改寫上游實作。
* 實驗方式：固定資料/切分，跑多個 random seeds，彙整 mean±std。

---

## 🧭 開發里程碑（Roadmap）

> 依你既有的「GRAFT-開發指南」建議順序撰寫，做完就把核取方塊勾起來。

1. [ ] **打通 `pipelines/run_baseline_grape.py`（不刪減）**
   產出：GRAPE 原始結果（Imputation & Label）。
2. [ ] **接上 `t2g_adapter → t2g_feature_selector + bipartite_builder`**
   產出：將 T2G-FORMER 的 FR-Graph/選特徵 轉為 GRAPE 可用的二分圖（特徵↔觀測）。
3. [ ] **接上 `lunar_adapter → lunar_row_selector（硬刪/軟權雙模式）`**
   產出：在觀測維度做列選擇/權重化，以模擬 LUNAR 對正常樣本區域的偏好。
4. [ ] **整合為 `pipelines/run_combo.py`**（可切換：僅 GRAPE / +T2G / +LUNAR / All）。
5. [ ] **加入三個 `random_*` 對照**

   * `random_feature_drop`（隨機丟特徵）
   * `random_row_drop`（隨機丟樣本）
   * `random_graph_edges`（隨機建邊/權重）
6. [ ] **視需要加入 `em_loops.py` 1–3 輪**
   在補全↔訓練之間做 EM 式迭代：E（補全）→ M（訓練）。

> ✅ 完成標記與關聯 PR：

* Baseline：
* T2G Adapter：
* LUNAR Adapter：
* Combo：

---

## 📦 環境與需求（Requirements & Environment）

> 建議同時支援 **Colab** 與 **本機/伺服器（conda）**。

**建議版本（可在 Colab 實測後填）：**

* Python：`TODO（例如 3.10.x）`
* PyTorch / CUDA：`TODO`
* PyTorch Geometric：`TODO`
* pandas / numpy / scikit-learn：`TODO`
* fancyimpute / cvxpy（若需要）：`TODO`

### A. Colab 一鍵安裝

### B. 本機（conda）

---

## 🗂️ 專案結構（Repo Structure）

- T2G-LUNAR / LUNAR-T2G
- GRAPE結尾，保留端到端特性

## 🚀 快速開始（Quickstart）

### 1) 下載資料（UCI）與切分

### 2) 跑 GRAPE Baseline

### 3) 加入 T2G-FORMER 元件（特徵圖/選特徵）

### 4) 加入 LUNAR 元件（列選擇/軟權）

### 5) 全部整合 + 隨機對照

---

## 📊 評估與報告（Evaluation & Reporting）

* **主指標（Imputation）**：`MAE`（越低越好）。
* **重複實驗**：**5 seeds**（建議：`[0,1,2,3,4]`），報告 `mean ± std`。
* **匯整腳本**：`tools/export_results.py` 讀取 `result.pkl/csv`，匯出到 `results/tables/summary.csv`。

結果表格欄位建議：

```
dataset, task, missing_rate, method, use_t2g, use_lunar, random_ctrl, seed, MAE, RMSE, time_sec
```

---

## ⚙️ 重要介面（Adapters / Selectors / Builders）

### T2G Adapter（特徵層）

* 輸入：原始表格 `X ∈ R^{N×D}`。
* 產出：FR-Graph（例如 cosine/top-k），`t2g_feature_selector` 回傳被選特徵子集 `D'`。
* `bipartite_builder`：建立 GRAPE 需要的二分圖（觀測↔特徵）。

### LUNAR Adapter（觀測層）

* `lunar_row_selector`：

  * **硬刪（hard-delete）**：過濾掉低信度/低密度樣本。
  * **軟權（soft-weight）**：保留所有樣本，但以樣本權重參與損失。

### Randomizers（消融對照）

* `random_feature_drop` / `random_row_drop` / `random_graph_edges`：可調比例/機率。

---

## 🧪 實驗設定（Examples）

> 建議把你在 Colab 的實際指令貼到這裡，確保他人能重現。

---

## 🧷 Reproducibility（可重現性）

* 固定 random seed：`--seeds 5`（內部使用 `torch`, `numpy`, `random` 同步設置）。
* 記錄環境：把 `python -V`、`pip list`、`nvidia-smi` 另存到 `results/logs/env_*.txt`。
* 日誌/儀表板：建議接 `wandb`（專案名：`graft`）。

---

## 📑 Citation（引用）

> 請把 BibTeX 放在這裡（GRAPE / LUNAR / T2G-FORMER）。

* GRAPE: *Handling Missing Data with Graph Representation Learning*（年/會議/網址）
* LUNAR: *Unifying Local Outlier Detection Methods via Graph Neural Networks*（年/會議/網址）
* T2G-FORMER: *Organizing Tabular Features into Relation Graphs ...*（年/會議/網址）

---

## 🤝 Contributing（貢獻）

1. Fork & PR，請附：

   * 影響範圍（pipelines/adapters/...）
   * 實驗結果（至少 1 組 dataset × seed≥3）
2. 風格：`black` / `ruff`，型別註解 `typing` 可選。

---

## 📜 License

* License：`TODO（MIT/Apache-2.0/...）`

---

## 🗓️ Changelog（請每次更新補上）

* 2025-08-18：初始化 README 草稿。

---

## ✅ 待辦清單（Checklist）

* [ ] 在 Colab 成功跑完 GRAPE baseline 並記錄環境版本
* [ ] 完成 T2G Adapter（含選特徵）+ 單元測試
* [ ] 完成 LUNAR Row Selector（硬刪/軟權）+ 單元測試
* [ ] `run_combo.py` 串接成功（四種模式可切換）
* [ ] 三個 randomizers 加入並在表格中呈現消融
* [ ] 匯表腳本 `export_results.py` 產出 `summary.csv`
* [ ] 撰寫 `DEVLOG.md`、`EXPERIMENTS.md` 並鏈到 README
* [ ] 補上 Citation 與 License
