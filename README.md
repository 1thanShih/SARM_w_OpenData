# SARM — Charging Bimanual Reward Model

訓練 [SARM (Stage-Aware Reward Modeling)](https://arxiv.org/abs/2509.25358) reward model，
使用 AllenAI bimanual charging 示範資料集，在 Google Colab Pro A100 上執行。

---

## Goal

讓 SARM reward model 學會辨識「手機充電」任務的四個階段進度（0 → 1），
作為後續 Reward-Aligned Behavior Cloning（RA-BC）的 progress signal 來源。

**任務四階段**

| # | Subtask | subtask_index |
|---|---------|---------------|
| 1 | Pick up the phone | 0 |
| 2 | Flip the phone sideways | 1 |
| 3 | Pick up the charging cable and plug it into the phone | 2 |
| 4 | Turn on the power of the extension cord | 3 |

---

## Dataset

> ✅ 標注 + 修正已完成，可直接用於訓練（不需再 merge / 標注）。

| 項目 | 內容 |
|------|------|
| Source repos | `allenai/19012026-charging-01` ～ `charging-06`（共 284 eps） |
| **訓練用 repo** | **`Lebruhbruh/SARM-opendata-annotated-fixed`** |
| Model repo | `{user}/sarm-charging-bimanual` |
| Episodes | **273**（284 排除 11 個問題 episodes） |
| Frames | 585,425 |
| FPS | 30 |
| Robot | bi_yam_follower（bimanual，14-dim state） |
| Cameras | `observation.images.{top,left,right}`，訓練主視角 = `top` |
| Annotation mode | `dense_only` |

**這個 repo 已處理過的事項：**
- `subtask_index` 在每個 episode 內依 stage 執行順序 `0 → 1 → 2 → 3` 單調遞增
- `meta/episodes/*.parquet` 的 `dense_subtask_*` 欄位全部 materialize（無 NULL）
- `stats.json` 已重算、`v3.0` git tag 已建立

---

## Quick Start（Lite Notebooks）

> 推薦流程：三個 **Lite** notebook 都在 Colab Pro A100 80GB 執行，照順序跑即可。
> （`SARM_Training_Colab.ipynb` 是含 merge / 標注的完整版，資料已備妥時用 Lite 版更快。）

| Notebook | 用途 | 預計時間 |
|----------|------|---------|
| `SARM_Sanity_Test_Colab.ipynb` | 200-step 健檢：確認 dense loss 會掉、reward 不是 0 | 5–10 分鐘 |
| `SARM_Training_Lite.ipynb` | 正式訓練（5000 steps）並 push 模型 | 45–90 分鐘 |
| `SARM_Predict_Visualization_Lite.ipynb` | 生成「影片 + 同步進度條」MP4 | 約 10 分鐘 |

### 1. 設定 HuggingFace Token

在每個 notebook 的 Cell 2（使用者設定）填入有 **Write 權限**的 `HF_TOKEN`，
並確認 `HF_USERNAME` / `ANNOTATED_DATASET` 正確。

### 2. 先跑 Sanity Test（建議）

`SARM_Sanity_Test_Colab.ipynb` 會：
- 清空 Colab 端 dataset cache 並強制從 Hub 重抓
- **assert dense annotation 完整**（無 NULL，避免「預測全 0」）
- 跑 200 steps，解析 log 確認 `dense_*` loss 有在下降
- 對 ep 21 中段 frame 跑一次推論，確認 reward 不是 0

### 3. 正式訓練

`SARM_Training_Lite.ipynb` Cell 7 的訓練指令（SARM 是 `RewardModelConfig`，
所以用 `--reward_model.type=sarm`，**不是** `--policy.type`）：

```bash
lerobot-train \
  --dataset.repo_id=Lebruhbruh/SARM-opendata-annotated-fixed \
  --reward_model.type=sarm \
  --reward_model.annotation_mode=dense_only \
  --reward_model.image_key=observation.images.top \
  --reward_model.state_key=observation.state \
  --reward_model.n_obs_steps=8 \
  --reward_model.frame_gap=30 \
  --reward_model.repo_id={user}/sarm-charging-bimanual \
  --reward_model.push_to_hub=true \
  --batch_size=64 \
  --steps=5000 \
  --save_freq=2500 \
  --tolerance_s=0.001 \
  --wandb.enable=false
```

### 4. 視覺化驗收

`SARM_Predict_Visualization_Lite.ipynb` 生成預測影片，確認 progress 曲線
在有標注的 episode 上單調遞增（而非黏在 0）。

---

## ⚠️ 已知最大地雷：預測全 0

SARM 訓練讀的是 `meta/episodes/*.parquet` 的 `dense_subtask_names` 等欄位，
**不是** `meta/lerobot_annotations.json`。若大量 episode 該欄為 NULL，
`find_stage_and_tau` 會回傳 `(0, 0.0)`，dense target ≡ 0，model 塌成「永遠輸出 0」。

> 已踩過：`SARM-opendata-annotated-fixed` 早期版本只有 ep 0–7 有 dense 欄位，
> 265 個是 NULL → 訓練 5000 steps 後預測全 0。

**push dataset 後必驗：**

```python
import pandas as pd
df = pd.read_parquet('meta/episodes/chunk-000/file-000.parquet')
null_cnt = df['dense_subtask_names'].isna().sum()
assert null_cnt == 0, f'{null_cnt}/{len(df)} episodes 缺 dense_subtask_names → SARM 會塌'
```

完整排查 checklist 見 `CLAUDE.md` 的「SARM 預測全 0 的除錯 checklist」。

---

## Helper Scripts

資料準備 / 修復用的獨立腳本（皆需 `export HF_TOKEN=hf_xxxx`）：

| 腳本 | 說明 |
|------|------|
| `reexport_dataset.py` | 從原始標注 repo 讀資料、排除問題 episodes、修正 `subtask_index` 順序、push 到自己的 repo |
| `materialize_dense_annotations.py` | 把 `lerobot_annotations.json` 的 subtask materialize 進 `meta/episodes/*.parquet` 的 `dense_subtask_*` 欄（修「預測全 0」的核心步驟） |
| `recompute_stats.py` | 重算 `stats.json`（scalar/vector 全重算，image stats 保留），補 `subtask_index` / `task_index_high_level` |
| `copy_missing_videos.py` | 把 merged repo 缺的 left/right camera videos 補進 annotated repo |
| `test_annotation_quality.py` | 標注品質驗證（PASS/FAIL，非零 exit code 代表有 FAIL）；`--repo-id` 可換 repo |

---

## Key Files

| 檔案 | 說明 |
|------|------|
| `SARM_Training_Lite.ipynb` | 正式訓練 Notebook（資料已備妥時用） |
| `SARM_Sanity_Test_Colab.ipynb` | 200-step 健檢 Notebook |
| `SARM_Predict_Visualization_Lite.ipynb` | 預測視覺化（影片 + 進度條） |
| `SARM_Training_Colab.ipynb` | 完整版（含 merge 6 repos + 標注流程） |
| `SARM_Predict_Visualization.ipynb` | 完整版視覺化 |
| `CLAUDE.md` | 開發守則 + LeRobot dataset 地雷 + 預測全 0 除錯 checklist |
| `doc/debug.md` | 已知錯誤排除手冊（Exx 條目） |
| `doc/guideline.html` | 手動標注完整教學（瀏覽器開啟） |
| `doc/spec.md` | Cell 設計規格 |

---

## 查看 HTML 文件（Live Server）

`doc/guideline.html` 是完整的手動標注教學，建議用 VS Code Live Server 開啟，
可以在修改後自動重新整理頁面。

1. VS Code → Extensions（`Ctrl+Shift+X`）→ 搜尋 **Live Server**（Ritwick Dey）→ Install
2. 對 `doc/guideline.html` 按右鍵 → **Open with Live Server**

不想裝 extension 也可直接用瀏覽器開啟：

```bash
xdg-open doc/guideline.html   # Linux
open doc/guideline.html        # macOS
```

---

## References

- [SARM Paper (arXiv)](https://arxiv.org/abs/2509.25358)
- [LeRobot SARM Docs](https://huggingface.co/docs/lerobot/en/sarm)
- [AllenAI Charging Dataset](https://huggingface.co/collections/allenai/molmoact2-bimanualyam-dataset)
- [lerobot-annotate Tool](https://github.com/huggingface/lerobot-annotate)
