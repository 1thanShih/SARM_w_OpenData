# SARM — Charging Bimanual Reward Model

訓練 [SARM (Stage-Aware Reward Modeling)](https://arxiv.org/abs/2509.25358) reward model，
使用 AllenAI bimanual charging 示範資料集，在 Google Colab Pro A100 上執行。

---

## Goal

讓 SARM reward model 學會辨識「手機充電」任務的四個階段進度（0 → 1），
作為後續 Reward-Aligned Behavior Cloning（RA-BC）的 progress signal 來源。

**任務四階段**

| # | Subtask |
|---|---------|
| 1 | Pick up the phone |
| 2 | Flip the phone sideways |
| 3 | Pick up the charging cable and plug it into the phone |
| 4 | Turn on the power of the extension cord |

---

## Dataset

| 項目 | 內容 |
|------|------|
| Source repos | `allenai/19012026-charging-01` ～ `charging-06` |
| Merged repo | `{user}/charging_bimanual_merged` |
| Annotated repo | `{user}/charging_bimanual_annotated` |
| Model repo | `{user}/sarm-charging-bimanual` |
| Episodes | 284（6 repos 合計） |
| FPS | 30 |
| Robot | bi_yam_follower（bimanual） |
| Camera used | `observation.images.top` |
| Annotation mode | `dense_only` |

---

## Quick Start

> 全流程在 `SARM_Training_Colab.ipynb` 執行，平台為 Colab Pro A100 80GB。

### 1. 開啟 Notebook

將 `SARM_Training_Colab.ipynb` 上傳至 Google Colab，確認 GPU 為 A100 80GB。

### 2. 設定 HuggingFace Token

在 Colab Secrets 加入 `HF_TOKEN`（需要 Write 權限）。

### 3. 按序執行 Cells

```
Cell 1   — 設定變數（repo IDs、subtask 名稱等）
Cell 1b  — 定義 progress_stage() context manager
Cell 2   — 安裝 LeRobot + SARM 依賴
Cell 3   — Transformers 5.x bug 修補（必做）
Cell 4   — HuggingFace 登入
Cell 4b  — 合併六個 source datasets → charging_bimanual_merged
Cell 5   — Dataset 預覽與驗證
Cell 9/10 — Subtask 標注（VLM 或手動，見下方）
Cell 12  — SARM 訓練
Cell 13  — 預測視覺化
```

### 4. Subtask 標注方式

**選項 A：VLM 自動標注（Qwen3-VL-30B，快但品質不穩定）**

```bash
python src/lerobot/data_processing/sarm_annotations/subtask_annotation.py \
  --repo-id {user}/charging_bimanual_merged \
  --dense-only \
  --dense-subtasks "Pick up the phone,Flip the phone sideways,..." \
  --video-key observation.images.top \
  --skip-existing
```

**選項 B：手動標注（lerobot-annotate UI，穩定但需人工操作）**

詳見 [`doc/guideline.html`](doc/guideline.html)，用瀏覽器打開查看完整流程。

### 5. 訓練

```bash
lerobot-train \
  --dataset.repo_id={user}/charging_bimanual_annotated \
  --policy.type=sarm \
  --policy.annotation_mode=dense_only \
  --policy.image_key=observation.images.top \
  --policy.frame_gap=30 \
  --batch_size=32 \
  --steps=5000 \
  --policy.repo_id={user}/sarm-charging-bimanual
```

---

## Project Progress

### ✅ 2026-05-21 — Dataset 切換

- 從 FurnitureBench 切換到 AllenAI `molmoact2-bimanualyam` charging 資料集
- 確認六個 source repos（`charging-01` ～ `charging-06`），實際 episodes = **284**
- Cell 4b 完成 `aggregate_datasets()` 合併邏輯，解決 E12 / E13 / E14 / E15 四個合併錯誤

### ✅ 2026-05-22 — 手動標注方案建立

- Qwen3-VL-30B 自動標注效果不佳，決定改用手動標注（Route C）
- 研究 [lerobot-annotate](https://github.com/huggingface/lerobot-annotate) 工具，確認與 SARM 的格式差異
- 撰寫格式轉換腳本 `convert_annotate_to_sarm.py`（lerobot-annotate output → SARM episodes parquet）
- 建立完整手動標注教學 [`doc/guideline.html`](doc/guideline.html)

### 🔄 進行中 — 手動標注執行

- [ ] 架設 lerobot-annotate 本機伺服器
- [ ] 對各 source repo 各抽 5 集（共 30 集）做手動標記
- [ ] 執行格式轉換並驗證 `temporal_proportions_dense.json`
- [ ] Push annotated dataset 到 HF Hub

### ⏳ 待進行 — SARM 訓練

- [ ] 標注驗證通過後，在 Colab A100 執行 SARM 訓練（5,000 steps）
- [ ] 預測視覺化確認 progress 曲線單調遞增
- [ ] （Optional）RA-BC 訓練

---

## 查看 HTML 文件（Live Server）

`doc/guideline.html` 是完整的手動標注教學，建議用 VS Code Live Server 開啟，
可以在修改後自動重新整理頁面。

### 安裝

1. 開啟 VS Code
2. 前往 Extensions（`Ctrl+Shift+X`）
3. 搜尋 **Live Server**（作者：Ritwick Dey）
4. 點 Install

### 使用

**方法 A：從檔案右鍵**
1. 在 VS Code 的檔案總管裡，對 `doc/guideline.html` 按右鍵
2. 選 **Open with Live Server**
3. 瀏覽器會自動開啟 `http://127.0.0.1:5500/doc/guideline.html`

**方法 B：從狀態列**
1. 用 VS Code 打開專案資料夾
2. 點視窗右下角的 **Go Live** 按鈕
3. 手動在瀏覽器網址列輸入 `http://127.0.0.1:5500/doc/guideline.html`

> 如果不想裝 VS Code Extension，直接用瀏覽器開啟也可以：
> ```bash
> xdg-open doc/guideline.html   # Linux
> open doc/guideline.html        # macOS
> ```

---

## Key Files

| 檔案 | 說明 |
|------|------|
| `SARM_Training_Colab.ipynb` | 主要訓練 Notebook（在 Colab 執行） |
| `doc/guideline.html` | 手動標注完整教學（瀏覽器開啟） |
| `doc/debug.md` | 已知錯誤排除手冊（E01 ～ E15） |
| `doc/spec.md` | Cell 設計規格 |
| `doc/plan.md` | 原始訓練計劃（FurnitureBench，已過期） |

---

## References

- [SARM Paper (arXiv)](https://arxiv.org/abs/2509.25358)
- [LeRobot SARM Docs](https://huggingface.co/docs/lerobot/en/sarm)
- [AllenAI Charging Dataset](https://huggingface.co/collections/allenai/molmoact2-bimanualyam-dataset)
- [lerobot-annotate Tool](https://github.com/huggingface/lerobot-annotate)
