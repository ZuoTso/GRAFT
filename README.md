# GRAFT：Graph-Aided Feature/Row Trimming Orchestrator for GRAPE / LUNAR / T2G-Former

> **TL;DR**：這個專案提供一個「單一入口的 Pipeline Orchestrator」，把 **T2G‑Former**（以 FR‑Graph 權重做「軟式特徵刪除」）、**LUNAR**（行/列遮罩產生器）與 **隨機遮罩** 組裝起來，最後以 **GRAPE** 跑兩個任務：**MDI（特徵插補）** 與 **Y（標籤預測）**。整條流程以嚴格的 **overlay manifest**（`GRAFT_OVERLAY_MANIFEST`）控管遮罩如何疊合（`AND` 或 `OR`），確保不洩漏測試資料，並把所有中介物規範化到 `artifact_dir` 之下。

---

## 目錄

* [特色](#特色)
* [安裝需求](#安裝需求)
* [檔案與目錄結構](#檔案與目錄結構)
* [快速開始](#快速開始)

  * [A) 只匯出 GRAPE Baseline 中介物（prep-only）](#a-只匯出-grape-baseline-中介物prep-only)
  * [B) 完整串接：T2G →（可選 LUNAR）→ GRAPE](#b-完整串接t2g-可選-lunar-grape)
  * [C) 已有 T2G 權重時的最短流程](#c-已有-t2g-權重時的最短流程)
  * [D) PACK Domain（自備 baseline 目錄）](#d-pack-domain自備-baseline-目錄)
* [關鍵參數與環境變數](#關鍵參數與環境變數)

  * [run\_pipeline.py（主 orchestrator）](#run_pipelinepy主-orchestrator)
  * [T2G Adapter（軟刪欄）](#t2g-adapter軟刪欄)
  * [LUNAR Adapter（列/欄遮罩）](#lunar-adapter列欄遮罩)
  * [Baseline Runner（GRAPE 收斂／蒐集）](#baseline-runnergrape-收斂蒐集)
* [Artifact 版面（輸出規格）](#artifact-版面輸出規格)
* [常見問題（FAQ / Debug）](#常見問題faq--debug)
* [研究建議與消融測試備註](#研究建議與消融測試備註)
* [授權與鳴謝](#授權與鳴謝)

---

## 特色

* **單一入口**：`pipelines/run_pipeline.py` 指定 `--modules` 與 `--order`，自動建立一次性 **variants** 工作區，跑完各 stage 並呼叫 GRAPE。
* **嚴格 manifest**：以 `overlay_manifest.json` 明確列出要疊合的遮罩、順序與運算（`AND` / `OR`），再由 GRAPE 訓練腳本讀取環境變數 `GRAFT_OVERLAY_MANIFEST` 實際套用，避免吃到舊檔。
* **安全與可重現**：

  * Baseline 中介物（`X_norm.npy`、`mask.npy`、`split_idx.json`、…）缺時可 `--auto_prep` 自動從 GRAPE 匯出。
  * 每次執行生成 **run\_token** 對應的 `variants/<token>/`，不互相覆蓋。
* **多資料域**：支援 **UCI domain**（GRAPE 內建）與 **PACK domain**（自備 baseline 目錄：`X_norm.npy / y.npy / mask.npy / split_idx.json`）。
* **雙任務**：

  * **MDI（插補）**：輸出 `impute/metrics.json`（RMSE/MAE）與邊級預測。
  * **Y（標籤）**：輸出 `label/metrics.json`（RMSE/MAE）與測試集預測。

---

## 安裝需求

* Python ≥ 3.10
* PyTorch（依 GPU/環境安裝對應版本）
* NumPy、pandas、joblib 等常用套件
* 一個可用的 **GRAPE** 專案目錄（含 `train_mdi.py`、`train_y.py`）
* 選配：**T2G‑Former** 專案（用於匯出 FR‑Graph 權重）

> **建議**：在虛擬環境或 Colab 建立乾淨環境，並設定 `artifact_dir=/content/grapt_artifacts`（可自由更改）。

---

## 檔案與目錄結構

專案核心檔案（節錄）：

```
./pipelines/
  run_pipeline.py               # 主 orchestrator（variants + overlay manifest 嚴格模式）
  run_baseline_grape.py         # 呼叫/蒐集 GRAPE（或只匯出中介物）
  t2g_export_from_grape.py      # 以 GRAPE baseline 當資料來源，forward-hook 匯出 T2G FR-Graph 權重
  t2g_adapter.py                # 讀取 FR-Graph 權重 → 合成列×欄遮罩 mask_t2g.npy（軟刪欄）
  lunar_adapter.py              # 呼叫/包裝 LUNAR，輸出 mask_lunar.npy（可選 edge_keep）
  pack_subparser.py / pack_data.py  # PACK domain 支援（自備 baseline）

third_party/
  GRAPE/  LUNAR/  T2GFormer/    # 建議以子模組或路徑指向（可依使用者目錄調整）

<artifact_dir>/
  baseline/<dataset>/seed<k>/   # 所有中介與結果統一放這裡（見下節）
```

---

## 快速開始

參考`GRAFT_my_operations.ipynb`

### A) 只匯出 GRAPE Baseline 中介物（prep-only）

> 當 `X_norm.npy / mask.npy / split_idx.json` 尚未就緒時，可先做一次 **prep**。

---

### B) 完整串接：T2G →（可選 LUNAR）→ GRAPE

* 若要加入 **LUNAR**：把 `--modules` 與 `--order` 改成 `t2gexp,t2g,lunar,grape` 與 `t2g>lunar>grape`，並追加 LUNAR 參數（如 `--lunar.keep_ratio 0.9`）。
* `--mask_op AND|OR` 控制 baseline 與各遮罩的疊合方式。

---

### C) 已有 T2G 權重時的最短流程

如果你已手上有 `W_layer*.npy`（同一維度的方陣，建議對稱化），可直接跳過
