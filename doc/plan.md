# SARM Reward Model 訓練計劃

## 專案概覽

本計劃使用 HuggingFace 開放資料集，在 Google Colab Pro (A100 80GB) 上訓練
LeRobot 的 **SARM (Stage-Aware Reward Modeling)** Reward Model。

SARM 是一個**影像式獎勵模型**，能從機器人操作的示範影片中學習「任務進度」的概念：
給定影片畫面與任務描述，輸出機器人完成任務的進度分數 (0 → 1)。

**參考論文**: [SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation](https://arxiv.org/abs/2509.25358)
**官方文件**: [https://huggingface.co/docs/lerobot/sarm](https://huggingface.co/docs/lerobot/sarm)

---

## 資料集選擇

### 選用: `tailong-wu/furniture_bench_dataset_lerobot_v30`

| 項目 | 內容 |
|------|------|
| HuggingFace 連結 | [tailong-wu/furniture_bench_dataset_lerobot_v30](https://huggingface.co/datasets/tailong-wu/furniture_bench_dataset_lerobot_v30) |
| 格式 | LeRobot v3.0 (SARM 原生支援) |
| 任務 | 9 種家具組裝（燈具、桌子、椅子等）(單臂 Franka 機器人) |
| Episodes 數量 | 5,100（9 任務 × ~567 episodes） |
| 使用任務 | **one_leg**（單腿組裝，task_index=0，~567 episodes） |
| 總 Frames 數 | 3,948,057 |
| FPS | 10 |
| 機器人 | Franka（單臂，7-DOF + 夾爪） |
| 攝影機 Key | `observation.images.image` (224×224), `observation.images.wrist_image` (224×224) |
| 狀態 Key | `observation.state` (8 維: xyz + 四元數 + 夾爪) |
| 授權 | Apache-2.0 |

**選擇理由**:
- 家具組裝是典型的長序列多階段任務，天生具備明確的子任務邊界（抓取 → 對齊 → 插入 → 固定）
- LeRobot v3.0 格式直接與 SARM 訓練腳本相容
- Franka 機器人與布料折疊的 OpenArms 完全不同，適合測試 SARM 的跨資料集泛化性
- 5,100 個 episodes，選單一任務（one_leg）即有 ~567 個，資料量充足
- 用於測試 SARM 在不同任務/機器人/環境的表現

---

## Annotation Mode 說明

本計劃使用 **`dense_only`** 模式：

- 不需要手動定義高層次 stages (sparse head 自動生成)
- 只需提供細粒度子任務名稱
- VLM (Qwen3-VL-30B) 自動分析影片並標注每個子任務的起止 frame
- 適合布料折疊這種有清晰動作順序的任務

**Dense 子任務 (one_leg 任務建議)**:
```
"Pick up furniture leg from workspace,Orient leg vertically above table socket,Insert leg tip into table socket,Push leg down until fully seated"
```

---

## 工作流程

```
Phase 0  環境設置
├── 安裝 LeRobot + SARM 套件 + Flash Attention 2
├── 修補 Transformers 5.x 已知 Bug (CRITICAL)
└── HuggingFace 登入 (Write Token)

Phase 1  資料集預覽
├── 確認 image key、state shape
└── 觀看 2-3 個 episodes，確認子任務名稱是否合適

Phase 2  VLM 自動標注  ← 最耗時
├── 建立標注資料集 repo: user/high_quality_folding_annotated
├── Qwen3-VL-30B 分析每個 episode 影片
└── 輸出: dense_subtask 欄位 + temporal_proportions.json

Phase 3  標注結果驗證
├── 視覺化 5 個 episodes 的子任務分割時間軸
└── 確認邊界是否正確 (若錯誤需調整子任務描述重新跑)

Phase 4  訓練 SARM
├── 使用標注後的資料集訓練
├── batch_size=64 (A100 優化)
├── 5,000 steps
└── 訓練完自動 Push 模型到 HuggingFace Hub

Phase 5  預測結果視覺化
└── 確認 reward model 對每個 episode 的進度預測是否單調遞增
```

---

## 時間估計 (Colab Pro A100 80GB)

| 階段 | 任務 | 估計時間 |
|------|------|---------|
| Phase 0 | 安裝 + Bug 修補 | 5–10 分鐘 |
| Phase 1 | 資料集預覽 | 2–5 分鐘 |
| Phase 2 | VLM 標注 (~567 episodes, one_leg) | **1.5–3 小時** |
| Phase 3 | 標注視覺化驗證 | 5 分鐘 |
| Phase 4 | SARM 訓練 (5,000 steps) | 45–90 分鐘 |
| Phase 5 | 預測視覺化 | 5–10 分鐘 |
| **總計** | | **~3–5 小時** |

> **建議**: 標注階段 (Phase 2) 最好隔夜執行。
> 支援中斷續跑 (`--skip-existing` 參數)，Colab 斷線後可繼續。

---

## 環境需求

| 項目 | 需求 |
|------|------|
| Colab 方案 | Pro 或 Pro+ |
| GPU | A100 80GB |
| HuggingFace 帳號 | 需要 (用於儲存標注資料集和訓練模型) |
| HuggingFace Token | Write 權限 (在 Settings → Access Tokens 建立) |
| Python | 3.12+ (Colab 內建) |

---

## 重要注意事項

### 1. Transformers 5.x Bug（必修補）
LeRobot SARM 依賴 transformers 5.x，但 5.x 版將 CLIP 特徵提取的回傳值改為物件而非 tensor，
導致 `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'detach'`。
**必須在安裝後立即修補**，詳見 spec.md Cell 3。

### 2. 多任務資料集需過濾單一任務
`furniture_bench_dataset_lerobot_v30` 包含 9 種不同家具組裝任務（task_index 0–8）。
SARM 訓練前必須先確認要使用哪個任務（建議 one_leg，task_index=0），
並記錄該任務的 episode indices，在標注時僅處理那些 episodes。
詳見 spec.md Cell 5b（任務過濾）。

### 3. 標注時間與 Colab 限制
標注 ~567 個 episodes 需要 1.5–3 小時。
解決方案：使用 `--skip-existing` 參數，中斷後重啟 runtime 可從斷點繼續。

### 4. 子任務名稱品質決定標注品質
若子任務描述不夠精確，VLM 可能標注錯誤。
**建議先用 10 個 episodes 測試**，確認標注正確後再跑全部 ~567 個。

### 5. 資料集不需要本地下載
透過 `--output-repo-id` 機制，標注腳本會串流讀取原始影片，
只將標注欄位 (parquet + metadata) 上傳到你的 HF repo，
不需要在 Colab 下載整個資料集。

---

## 輸出成果

| 輸出 | 位置 |
|------|------|
| 標注資料集 | `your-username/furniture_bench_one_leg_annotated` (HuggingFace Hub) |
| SARM Reward Model | `your-username/sarm-furniture-one-leg` (HuggingFace Hub) |
| 本地 Checkpoint | `outputs/train/sarm_furniture_one_leg/` (Colab runtime) |
| 標注視覺化 | `./annotation_viz/` (Colab runtime) |
| 預測視覺化 | `./sarm_predictions_viz/` (Colab runtime) |
