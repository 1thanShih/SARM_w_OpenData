# SARM Training — Complete Colab Specification

## 使用方式

將以下各 Cell 的程式碼依序複製貼入 Google Colab Notebook 並執行。
每個 Cell 都有獨立的說明，並標示預期輸出。

**執行前確認**: Runtime → Change runtime type → **A100 GPU**

---

## Cell 0: GPU 確認

```python
# 確認已取得 A100 GPU
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)

import torch
assert torch.cuda.is_available(), "錯誤：沒有 GPU！請檢查 Runtime 設定"
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"\nGPU: {gpu_name}")
print(f"VRAM: {vram_gb:.1f} GB")

if vram_gb < 70:
    print("⚠️  警告：VRAM 不足 70GB，建議使用 A100 80GB")
else:
    print("✓ GPU 規格符合需求")
```

**預期輸出**: 顯示 `A100-SXM4-80GB`，VRAM 約 80.0 GB

---

## Cell 1: 使用者設定（必填）

```python
# ============================================================
# 請填入你的資訊
# ============================================================

# 你的 HuggingFace 使用者名稱
HF_USERNAME = "your-username"   # <-- 改成你的 HF username

# 你的 HuggingFace Write Token
# 取得方式: https://huggingface.co/settings/tokens → New token → Write 權限
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"   # <-- 貼上你的 token

# 標注後資料集的 repo 名稱（會自動建立）
ANNOTATED_REPO_ID = f"{HF_USERNAME}/furniture_bench_one_leg_annotated"

# 訓練好的 SARM 模型 repo 名稱
MODEL_REPO_ID = f"{HF_USERNAME}/sarm-furniture-one-leg"

# 原始資料集（FurnitureBench, LeRobot v3.0）
SOURCE_DATASET = "tailong-wu/furniture_bench_dataset_lerobot_v30"

# 要訓練的任務（0=one_leg, 1-8=其他家具任務，Cell 5b 會顯示對照表）
TASK_INDEX = 0

print(f"標注資料集將存入: {ANNOTATED_REPO_ID}")
print(f"SARM 模型將存入: {MODEL_REPO_ID}")
print(f"使用任務 index: {TASK_INDEX}")
```

---

## Cell 2: 安裝套件

```bash
%%bash
# 安裝 LeRobot（包含 SARM 依賴）
pip install -q "git+https://github.com/huggingface/lerobot.git#egg=lerobot[sarm]"

# 安裝 Flash Attention 2（讓 Qwen3-VL-30B 在 A100 上有效率運行）
pip install -q flash-attn --no-build-isolation

# 確認安裝
python -c "import lerobot; print(f'LeRobot version: {lerobot.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

**預期輸出**: 無錯誤，顯示 LeRobot 和 Transformers 版本號

> **注意**: 安裝約需 5–10 分鐘。若出現依賴衝突警告可忽略，只要最後確認行有正確輸出即可。

---

## Cell 3: 修補 Transformers 5.x Bug（CRITICAL — 必須執行）

LeRobot SARM 使用 CLIP 模型做特徵提取，但 Transformers 5.x 版將回傳型別從 `tensor` 改為
`BaseModelOutputWithPooling` 物件，導致後續 `.detach()` 呼叫失敗。此 Cell 自動修補此問題。

```python
import os, re

# 找到安裝位置
import lerobot
lerobot_root = os.path.dirname(lerobot.__file__)
processor_path = os.path.join(lerobot_root, "policies", "sarm", "processor_sarm.py")
print(f"修補目標: {processor_path}")
assert os.path.exists(processor_path), "找不到 processor_sarm.py，請確認 SARM 安裝成功"

with open(processor_path, 'r') as f:
    content = f.read()

patched = False

# 修補 image features
old_img = "embeddings = self.clip_model.get_image_features(**inputs).detach().cpu()"
new_img = (
    "output = self.clip_model.get_image_features(**inputs)\n"
    "        if not isinstance(output, torch.Tensor):\n"
    "            output = output.pooler_output\n"
    "        embeddings = output.detach().cpu()"
)
if old_img in content:
    content = content.replace(old_img, new_img)
    print("✓ 已修補 image features")
    patched = True
else:
    print("ℹ image features 不需修補（已是最新版或已修補）")

# 修補 text features
old_txt = "embeddings = self.clip_model.get_text_features(**inputs).detach().cpu()"
new_txt = (
    "output = self.clip_model.get_text_features(**inputs)\n"
    "        if not isinstance(output, torch.Tensor):\n"
    "            output = output.pooler_output\n"
    "        embeddings = output.detach().cpu()"
)
if old_txt in content:
    content = content.replace(old_txt, new_txt)
    print("✓ 已修補 text features")
    patched = True
else:
    print("ℹ text features 不需修補（已是最新版或已修補）")

with open(processor_path, 'w') as f:
    f.write(content)

if patched:
    print("\n修補完成。驗證修補結果:")
    import subprocess
    result = subprocess.run(['grep', '-n', 'pooler_output', processor_path],
                           capture_output=True, text=True)
    print(result.stdout if result.stdout else "（未找到 pooler_output — 請檢查原始碼是否有變動）")
else:
    print("\n✓ 程式碼已是最新版，無需修補")
```

**預期輸出**: 顯示已修補的行號，或確認無需修補

---

## Cell 4: HuggingFace 登入

```python
from huggingface_hub import login
login(token=HF_TOKEN)
print("✓ HuggingFace 登入成功")
```

---

## Cell 5: 資料集預覽（確認 Image Key 與子任務名稱）

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

print("載入資料集 metadata（不下載影片）...")
dataset = LeRobotDataset(SOURCE_DATASET)

print(f"\n資料集基本資訊:")
print(f"  Episodes 總數: {dataset.num_episodes}")
print(f"  Frames 總數: {dataset.num_frames:,}")
print(f"  FPS: {dataset.fps}")
print(f"  格式版本: {dataset.meta.info.get('codebase_version', 'unknown')}")

print(f"\n可用的 Feature Keys:")
for k in dataset.features:
    print(f"  {k}")

image_keys = [k for k in dataset.features if 'images' in k]
print(f"\n攝影機 Keys: {image_keys}")

print(f"\n任務描述:")
for idx, task in enumerate(dataset.meta.tasks.values()):
    print(f"  [{idx}] {task}")

# 取樣一個 frame 確認形狀
sample = dataset[0]
print(f"\n單一 Frame 形狀:")
for k, v in sample.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
```

**預期輸出**:
- Episodes 總數: 5100，任務數: 9
- 攝影機 Keys 包含 `observation.images.image` 和 `observation.images.wrist_image`
- `observation.state` 形狀為 `(8,)`（xyz + 四元數 + 夾爪）

---

## Cell 5b: 任務過濾（找出 one_leg 的 Episodes）

```python
import pandas as pd
from huggingface_hub import hf_hub_download
import json

print("下載 episodes 元資料（僅 metadata，無影片）...")

# 讀取資料集 parquet 確認 task_index 分布
# 用 LeRobot dataset 直接查 episodes
episode_df = pd.DataFrame([
    {'episode_index': ep, 'task_index': info.get('task_index', -1)}
    for ep, info in dataset.meta.episodes.items()
])

print(f"\n各 task_index 的 episodes 數量:")
task_counts = episode_df.groupby('task_index').size()
print(task_counts.to_string())

# 讀取 tasks.jsonl 取得任務名稱
try:
    tasks_path = hf_hub_download(
        repo_id=SOURCE_DATASET,
        filename="meta/tasks.jsonl",
        repo_type="dataset",
        token=HF_TOKEN,
    )
    with open(tasks_path) as f:
        tasks = [json.loads(l) for l in f]
    print(f"\n任務名稱對照:")
    for t in tasks:
        count = task_counts.get(t['task_index'], 0)
        marker = " ← 選擇此任務" if t['task_index'] == TASK_INDEX else ""
        print(f"  [{t['task_index']}] {t['task']}  ({count} episodes){marker}")
except Exception as e:
    print(f"無法取得任務名稱: {e}")
    print("請根據 task_index 數字自行對應 FurnitureBench 任務")

# 取得選定任務的所有 episode indices
TASK_EPISODES = sorted(
    episode_df[episode_df['task_index'] == TASK_INDEX]['episode_index'].tolist()
)
print(f"\n✓ task_index={TASK_INDEX} 共有 {len(TASK_EPISODES)} 個 episodes")
print(f"  Episode 範圍: {TASK_EPISODES[0]} ~ {TASK_EPISODES[-1]}")
```

**預期輸出**: 列出 9 個任務及其 episode 數量，task_index=0 應有約 567 個 episodes。

---

## Cell 6: 觀看範例 Episodes（選擇性，確認子任務名稱）

```python
# 下載並顯示 3 個 episodes 的前幾幀
# 用來確認布料折疊的實際動作，以便調整子任務名稱

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 從 TASK_EPISODES 取前 3 個來預覽
preview_eps = TASK_EPISODES[:3]
dataset_preview = LeRobotDataset(SOURCE_DATASET, episodes=preview_eps)

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
for row, ep_idx in enumerate(preview_eps):
    ep_frames = [i for i, s in enumerate(dataset_preview)
                 if s['episode_index'].item() == ep_idx]
    sample_indices = np.linspace(0, len(ep_frames)-1, 5, dtype=int)
    for col, frame_idx in enumerate(sample_indices):
        frame = dataset_preview[ep_frames[frame_idx]]
        img = frame['observation.images.image'].numpy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        axes[row][col].imshow(img.transpose(1, 2, 0) if img.ndim == 3 else img)
        axes[row][col].set_title(f"Ep {ep_idx}, t={frame_idx}")
        axes[row][col].axis('off')

plt.suptitle(f"Task {TASK_INDEX} Episodes 預覽 (用來確認子任務名稱)", fontsize=14)
plt.tight_layout()
plt.savefig('/content/episode_preview.png', dpi=100, bbox_inches='tight')
plt.show()
print("✓ 預覽圖已儲存至 /content/episode_preview.png")
```

> 觀察這 15 幀畫面後，決定適合的 Dense 子任務名稱。
> one_leg 預設建議:
> `"Pick up furniture leg from workspace,Orient leg vertically above table socket,Insert leg tip into table socket,Push leg down until fully seated"`
> 如果實際動作不同，請在下方 Cell 7 調整 `DENSE_SUBTASKS` 變數。

---

## Cell 7: 設定標注參數

```python
# ============================================================
# 根據 Cell 6 的觀察調整以下參數
# ============================================================

# 細粒度子任務名稱（用逗號分隔，順序需符合實際動作順序）
# 建議先用 10 個 episodes 測試，確認正確後再跑全部
# one_leg 任務：抓腿 → 對齊插槽 → 插入 → 壓緊
DENSE_SUBTASKS = "Pick up furniture leg from workspace,Orient leg vertically above table socket,Insert leg tip into table socket,Push leg down until fully seated"

# 要使用的攝影機
VIDEO_KEY = "observation.images.image"

# VLM 模型（預設使用 SARM 論文的推薦模型）
VLM_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# 先測試前 10 個 one_leg episodes 再決定是否跑全部
# TASK_EPISODES 由 Cell 5b 產生（task_index=TASK_INDEX 的所有 episode indices）
TEST_EPISODES = TASK_EPISODES[:10]   # 前 10 個做測試
RUN_ALL_EPISODES = False             # 確認子任務正確後改為 True

print(f"標注設定:")
print(f"  子任務: {DENSE_SUBTASKS}")
print(f"  攝影機: {VIDEO_KEY}")
print(f"  VLM: {VLM_MODEL}")
print(f"  測試 episodes: {TEST_EPISODES}")
print(f"  全量 episodes 共: {len(TASK_EPISODES)} 個")
print(f"  跑全部 episodes: {RUN_ALL_EPISODES}")
```

---

## Cell 8: 建立標注輸出 Repo

```python
from huggingface_hub import HfApi

api = HfApi(token=HF_TOKEN)

try:
    api.create_repo(
        repo_id=ANNOTATED_REPO_ID,
        repo_type="dataset",
        exist_ok=True
    )
    print(f"✓ Repo 建立/確認成功: https://huggingface.co/datasets/{ANNOTATED_REPO_ID}")
except Exception as e:
    print(f"✗ 建立 repo 失敗: {e}")
    raise
```

---

## Cell 9a: 測試標注（先跑 10 個 Episodes）

**建議先執行此步驟驗證子任務名稱正確，再跑全部 1,200 個 episodes。**

```bash
%%bash
# 設定環境變數（從 Python 傳入）
source /content/sarm_env.sh 2>/dev/null || true

python -c "
import os
print('HF_USERNAME:', os.environ.get('HF_USERNAME', '未設定'))
print('ANNOTATED_REPO_ID:', os.environ.get('ANNOTATED_REPO_ID', '未設定'))
"
```

```python
import subprocess, os

# 寫入環境變數供 bash 使用
env_vars = {
    'HF_USERNAME': HF_USERNAME,
    'HF_TOKEN': HF_TOKEN,
    'ANNOTATED_REPO_ID': ANNOTATED_REPO_ID,
    'SOURCE_DATASET': SOURCE_DATASET,
    'DENSE_SUBTASKS': DENSE_SUBTASKS,
    'VIDEO_KEY': VIDEO_KEY,
    'VLM_MODEL': VLM_MODEL,
}

env = os.environ.copy()
env.update(env_vars)

cmd = [
    "python", "-m", "lerobot.data_processing.sarm_annotations.subtask_annotation",
    "--repo-id", SOURCE_DATASET,
    "--output-repo-id", ANNOTATED_REPO_ID,
    "--dense-only",
    "--dense-subtasks", DENSE_SUBTASKS,
    "--video-key", VIDEO_KEY,
    "--model", VLM_MODEL,
    "--num-workers", "1",
    "--episodes", *[str(e) for e in TEST_EPISODES],
    "--num-visualizations", "5",
    "--output-dir", "/content/annotation_viz_test",
    "--push-to-hub",
]

print(f"執行測試標注（{len(TEST_EPISODES)} episodes, task_index={TASK_INDEX}）...")
print("命令:", " ".join(cmd))
print("預計時間: 5–15 分鐘\n")

result = subprocess.run(cmd, env=env, text=True)
if result.returncode == 0:
    print("\n✓ 測試標注完成！")
else:
    print(f"\n✗ 標注失敗，返回碼: {result.returncode}")
```

---

## Cell 9b: 驗證測試標注結果

```python
import glob
from IPython.display import Image as IPyImage, display

viz_files = sorted(glob.glob('/content/annotation_viz_test/*.png'))
if not viz_files:
    print("找不到視覺化圖片。請確認標注步驟成功執行。")
else:
    print(f"找到 {len(viz_files)} 張視覺化圖片:\n")
    for path in viz_files[:5]:
        print(f"  {os.path.basename(path)}")
        display(IPyImage(path))
```

> **關鍵確認**:
> - 每個 episode 的時間軸上應顯示 4 個顏色區段（對應 4 個子任務）
> - 子任務邊界應與影片中的動作轉換點吻合
> - 若標注看起來不合理（例如所有幀都是同一個 stage），請調整 `DENSE_SUBTASKS` 後重新執行

---

## Cell 10: 全量標注（確認測試正確後執行）

**確認 Cell 9b 的視覺化結果正確後，才執行此步驟。**
預計時間：3–6 小時（中途 Colab 斷線可重新執行，`--skip-existing` 會跳過已標注的 episodes）

```python
import subprocess, os

cmd = [
    "python", "-m", "lerobot.data_processing.sarm_annotations.subtask_annotation",
    "--repo-id", SOURCE_DATASET,
    "--output-repo-id", ANNOTATED_REPO_ID,
    "--dense-only",
    "--dense-subtasks", DENSE_SUBTASKS,
    "--video-key", VIDEO_KEY,
    "--model", VLM_MODEL,
    "--num-workers", "1",
    "--episodes", *[str(e) for e in TASK_EPISODES],   # 只標注 one_leg 任務
    "--skip-existing",          # 斷線後重跑時跳過已完成的 episodes
    "--num-visualizations", "5",
    "--output-dir", "/content/annotation_viz",
    "--push-to-hub",
]

print(f"開始全量標注（{len(TASK_EPISODES)} episodes, task_index={TASK_INDEX}）...")
print("⚠️  預計需要 1.5–3 小時，建議隔夜執行")
print("   若 Colab 斷線，重新執行此 Cell 即可從斷點繼續（--skip-existing）\n")
print("命令:", " ".join(cmd))

env = os.environ.copy()
env.update({
    'HF_TOKEN': HF_TOKEN,
    'HUGGING_FACE_HUB_TOKEN': HF_TOKEN,
})

result = subprocess.run(cmd, env=env, text=True)
if result.returncode == 0:
    print("\n✓ 全量標注完成！")
else:
    print(f"\n✗ 標注中斷（返回碼: {result.returncode}）")
    print("重新執行此 Cell 可從斷點繼續（已有 --skip-existing）")
```

---

## Cell 11: 驗證完整標注

```python
import glob
from IPython.display import Image as IPyImage, display

print("最終標注視覺化:")
viz_files = sorted(glob.glob('/content/annotation_viz/*.png'))
for path in viz_files[:5]:
    print(f"\n{os.path.basename(path)}:")
    display(IPyImage(path))

print(f"\n標注資料集位置: https://huggingface.co/datasets/{ANNOTATED_REPO_ID}")
```

---

## Cell 12: 訓練 SARM Reward Model

標注完成後，訓練 SARM。預計 45–90 分鐘。

```python
import subprocess, os

cmd = [
    "lerobot-train",
    f"--dataset.repo_id={ANNOTATED_REPO_ID}",
    "--policy.type=sarm",
    "--policy.annotation_mode=dense_only",
    f"--policy.image_key={VIDEO_KEY}",
    "--policy.state_key=observation.state",
    "--policy.n_obs_steps=8",
    "--policy.frame_gap=30",
    "--output_dir=outputs/train/sarm_furniture_one_leg",
    "--batch_size=64",          # A100 80GB 可支援 64（預設 32，加速訓練）
    "--steps=5000",
    f"--policy.repo_id={MODEL_REPO_ID}",
    "--wandb.enable=false",     # 若有 W&B 帳號可改為 true 並加 --wandb.project=sarm_folding
]

print("開始訓練 SARM Reward Model...")
print("預計時間: 45–90 分鐘\n")
print("命令:", " ".join(cmd))

env = os.environ.copy()
env.update({
    'HF_TOKEN': HF_TOKEN,
    'HUGGING_FACE_HUB_TOKEN': HF_TOKEN,
})

result = subprocess.run(cmd, env=env, text=True)
if result.returncode == 0:
    print(f"\n✓ 訓練完成！模型已上傳至: https://huggingface.co/{MODEL_REPO_ID}")
else:
    print(f"\n✗ 訓練失敗（返回碼: {result.returncode}）")
```

**預期輸出**: 訓練 loss 應隨步數下降。最後一行應顯示模型已 push 到 HuggingFace Hub。

---

## Cell 13: 儲存 Checkpoint 到 Google Drive（選擇性）

若 Colab session 結束前需要備份：

```python
from google.colab import drive
import shutil, os

drive.mount('/content/drive')

backup_path = f"/content/drive/MyDrive/sarm_checkpoints/sarm_furniture_one_leg"
os.makedirs(backup_path, exist_ok=True)

if os.path.exists('outputs/train/sarm_furniture_one_leg'):
    shutil.copytree(
        'outputs/train/sarm_furniture_one_leg',
        backup_path,
        dirs_exist_ok=True
    )
    print(f"✓ Checkpoint 已備份至 Google Drive: {backup_path}")
else:
    print("找不到 checkpoint 目錄。請確認訓練已完成。")
```

---

## Cell 14: 視覺化預測結果

```python
import subprocess, os

cmd = [
    "python", "-m", "lerobot.policies.sarm.compute_rabc_weights",
    "--dataset-repo-id", ANNOTATED_REPO_ID,
    "--reward-model-path", MODEL_REPO_ID,
    "--visualize-only",
    "--num-visualizations", "5",
    "--head-mode", "dense",
    "--output-dir", "/content/sarm_predictions_viz",
]

print("產生預測視覺化...")
env = os.environ.copy()
env.update({'HF_TOKEN': HF_TOKEN, 'HUGGING_FACE_HUB_TOKEN': HF_TOKEN})

result = subprocess.run(cmd, env=env, text=True)
if result.returncode != 0:
    print(f"視覺化失敗（返回碼: {result.returncode}）")
```

```python
import glob
from IPython.display import Image as IPyImage, display

print("SARM Reward Model 預測結果:")
pred_files = sorted(glob.glob('/content/sarm_predictions_viz/*.png'))
for path in pred_files[:5]:
    print(f"\n{os.path.basename(path)}:")
    display(IPyImage(path))
```

**預期輸出**: 每個 episode 的進度預測曲線應呈現**單調遞增趨勢** (0 → 1)，
並在子任務轉換處有明顯的斜率變化。

---

## Cell 15: 確認模型已上傳 HuggingFace Hub

```python
from huggingface_hub import HfApi

api = HfApi(token=HF_TOKEN)

try:
    model_info = api.model_info(MODEL_REPO_ID)
    print(f"✓ 模型已成功上傳！")
    print(f"  名稱: {model_info.id}")
    print(f"  最後更新: {model_info.last_modified}")
    print(f"  連結: https://huggingface.co/{MODEL_REPO_ID}")
except Exception as e:
    print(f"✗ 找不到模型: {e}")
    print(f"  請確認 Cell 12 的訓練步驟是否成功完成")
```

---

## 完整流程總結

| Cell | 步驟 | 估計時間 |
|------|------|---------|
| 0 | GPU 確認 | 1 分鐘 |
| 1 | 使用者設定 | 2 分鐘 |
| 2 | 安裝套件 | 5–10 分鐘 |
| 3 | 修補 Transformers Bug | 1 分鐘 |
| 4 | HuggingFace 登入 | 1 分鐘 |
| 5–6 | 資料集預覽 | 2–5 分鐘 |
| 7 | 設定標注參數 | 2 分鐘 |
| 8 | 建立 HF Repo | 1 分鐘 |
| 9a–9b | 測試標注 (10 episodes) | 5–15 分鐘 |
| 10–11 | 全量標注 (~567 episodes, one_leg) | **1.5–3 小時** |
| 12 | 訓練 SARM | **45–90 分鐘** |
| 13 | 備份 (選擇性) | 5 分鐘 |
| 14–15 | 視覺化 + 確認 | 5–10 分鐘 |

---

## 常見錯誤排除

### `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'detach'`
→ Cell 3 的 Bug 修補未執行或失敗。重新執行 Cell 3。

### `OutOfMemoryError` 在標注階段
→ 降低 batch size 或重啟 runtime 後只跑標注（不要同時載入其他模型）。

### 標注視覺化顯示所有幀都是同一個 stage
→ 子任務描述不夠精確。修改 Cell 7 的 `DENSE_SUBTASKS`，描述要更具體（例如: "Robot arm reaches toward leg and grasps it" 而非 "Pick up leg"）。

### 訓練 loss 不收斂
→ 增加訓練步數：`--steps=10000`，或確認標注資料集格式正確。

### `404 Not Found` 存取標注資料集
→ Cell 8 的 repo 建立失敗，或 HF Token 沒有 write 權限。確認 Token 設定正確。
