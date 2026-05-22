# SARM 訓練排錯手冊

## 使用方式

遇到報錯時，將完整錯誤訊息傳給 Claude，並說明是哪個 Cell 出錯。
Claude 解決後會在此文件新增/更新對應條目。

---

## 已知錯誤索引

| # | 錯誤關鍵字 | 發生階段 | 跳至 |
|---|-----------|---------|------|
| E01 | `AttributeError: 'BaseModelOutputWithPooling'` | Cell 9a / Cell 10 / Cell 12 | [E01](#e01) |
| E02 | `CUDA out of memory` | Cell 9a / Cell 10 | [E02](#e02) |
| E03 | `Repository Not Found` / `404` | Cell 8 / Cell 9a | [E03](#e03) |
| E04 | `ImportError: No module named 'lerobot'` | 任何 Cell | [E04](#e04) |
| E05 | `lerobot-train: command not found` | Cell 12 | [E05](#e05) |
| E06 | `KeyError: 'dense_subtask_names'` | Cell 12 | [E06](#e06) |
| E07 | 標注視覺化全是同一 stage | Cell 9b / Cell 11 | [E07](#e07) |
| E08 | 預測曲線雜亂無序（不遞增） | Cell 14 | [E08](#e08) |
| E09 | `AssertionError: codebase_version` | Cell 5 / Cell 12 | [E09](#e09) |
| E10 | Colab session 斷線 | Cell 10（標注）| [E10](#e10) |
| E12 | `FileNotFoundError: ... videos/.../*.mp4` | Cell 4b（合併） | [E12](#e12) |
| E13 | `TypeError: data type 'list<element: double>[pyarrow]' not understood` | Cell 4b（合併） | [E13](#e13) |
| E14 | 跳過合併但結尾找不到 `meta/info.json` | Cell 4b（合併） | [E14](#e14) |
| E15 | `TypeError: HfHubHTTPError.__init__() missing ... 'response'` | Cell 4b 驗證 / 後續 LeRobotDataset 載入 | [E15](#e15) |

---

## 詳細說明

### E01

**錯誤訊息**
```
AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'detach'
```

**發生時機**: Cell 9a、Cell 10（標注）或 Cell 12（訓練）時，CLIP 編碼圖片或文字

**根本原因**: Transformers 5.x 將 `CLIPModel.get_image_features()` 和 `get_text_features()` 的回傳型別從 `torch.Tensor` 改為 `BaseModelOutputWithPooling` 物件，但 LeRobot SARM 程式碼預期是 tensor。

**解決方法**: 重新執行 Cell 3（Bug 修補）
```python
# 確認修補是否成功，應看到 pooler_output 出現在輸出中
import subprocess
r = subprocess.run(
    ['grep', '-n', 'pooler_output',
     '/content/lerobot/src/lerobot/policies/sarm/processor_sarm.py'],
    capture_output=True, text=True
)
print(r.stdout)
```
若 grep 無輸出，表示修補失敗。請確認 Cell 3 的程式碼邏輯：
- `old_img` 字串是否與原始碼完全一致（包含空格）
- 若不一致，手動搜尋 `get_image_features` 並改成：
```python
output = self.clip_model.get_image_features(**inputs)
if not isinstance(output, torch.Tensor):
    output = output.pooler_output
embeddings = output.detach().cpu()
```

---

### E02

**錯誤訊息**
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate X GiB
```

**發生時機**: Cell 9a / Cell 10（VLM 標注，Qwen3-VL-30B 載入時）

**根本原因**: Qwen3-VL-30B 需要約 60 GB VRAM（bfloat16）。若 GPU 不是 A100 80GB 或有其他程式佔用記憶體，會 OOM。

**解決方法**:

1. 確認 GPU 類型（Cell 0 應顯示 A100 80GB）
2. 清除 GPU 記憶體後重試：
```python
import torch, gc
gc.collect()
torch.cuda.empty_cache()
print(f"可用 VRAM: {torch.cuda.memory_reserved(0)/1e9:.1f} GB")
```
3. 若仍 OOM，嘗試加入 `--dtype float16` 參數（比 bfloat16 節省少量記憶體）
4. 確認 `--num-workers 1`（多 worker 會各自載入完整模型）

---

### E03

**錯誤訊息**
```
huggingface_hub.errors.RepositoryNotFoundError: ... Repository Not Found
```
或
```
HTTPError: 404 Client Error
```

**發生時機**: Cell 8（確認 fork）、Cell 9a（標注推送）

**根本原因**: 資料集 Fork 尚未完成，或 HF_TOKEN 無寫入權限。

**解決方法**:
1. 確認已在 HuggingFace 網頁完成 Duplicate：  
   `https://huggingface.co/datasets/lerobot/high_quality_folding`  
   → 右上角 `[...]` → `Duplicate this dataset` → Name: `high_quality_folding_annotated`

2. 確認 HF_TOKEN 為 **Write** 類型（不是 Read-only）：
```python
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
user = api.whoami()
print(f"登入為: {user['name']}")
# 測試是否有寫入權限
try:
    api.dataset_info(ANNOTATED_REPO_ID)
    print("repo 存在")
except:
    print("repo 不存在，請先完成 Fork")
```

---

### E04

**錯誤訊息**
```
ImportError: No module named 'lerobot'
ModuleNotFoundError: No module named 'lerobot'
```

**發生時機**: 任何 import lerobot 的 Cell

**根本原因**: 
- Cell 2 的安裝未完成
- 或安裝後未 Restart Runtime

**解決方法**:
1. 確認 `/content/lerobot` 存在：
```python
import os
print(os.path.exists('/content/lerobot'))
```
2. 若不存在，重新執行 Cell 2
3. 若存在但仍報錯：**Runtime → Restart Runtime**，再從 Cell 3 繼續（不需重新安裝）
4. 若重啟後仍找不到模組，手動加入路徑：
```python
import sys
sys.path.insert(0, '/content/lerobot/src')
import lerobot
```

---

### E05

**錯誤訊息**
```
FileNotFoundError: [Errno 2] No such file or directory: 'lerobot-train'
```
或
```
/bin/bash: lerobot-train: command not found
```

**發生時機**: Cell 12（訓練）

**根本原因**: `lerobot-train` entry point 未正確安裝

**解決方法**:
```python
import subprocess, sys

# 方法 1：用完整路徑呼叫
result = subprocess.run(
    ['find', '/usr', '/content', '-name', 'lerobot-train', '-type', 'f'],
    capture_output=True, text=True
)
print("找到:", result.stdout)

# 方法 2：用 python -m 方式執行
# 將 Cell 12 的 lerobot_train 改為：
lerobot_train = [sys.executable, '-m', 'lerobot.scripts.lerobot_train']
```
在 Cell 12 中把 `lerobot_train = shutil.which('lerobot-train') or 'lerobot-train'` 改為：
```python
lerobot_train = [sys.executable, '-m', 'lerobot.scripts.lerobot_train']
# 然後 cmd 的第一個元素改為展開：
cmd = [*lerobot_train, '--dataset.repo_id=...', ...]
```

---

### E06

**錯誤訊息**
```
KeyError: 'dense_subtask_names'
KeyError: 'meta/temporal_proportions_dense.json'
```

**發生時機**: Cell 12（訓練），讀取 dense 標注時

**根本原因**: 標注資料集中缺少 dense 標注欄位，可能是：
- Cell 10（全量標注）未完成
- 標注推送失敗，HF Hub 上的資料集沒有標注欄位

**解決方法**:
1. 確認標注資料集包含必要欄位：
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import sys; sys.path.insert(0, '/content/lerobot/src')

ds = LeRobotDataset(ANNOTATED_REPO_ID)
print("Episodes 欄位:", list(ds.meta.episodes.columns) if hasattr(ds.meta, 'episodes') else "無法取得")

# 檢查 temporal_proportions 是否存在
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
files = api.list_repo_files(ANNOTATED_REPO_ID, repo_type="dataset")
for f in files:
    if 'temporal' in f or 'subtask' in f:
        print(f)
```
2. 若標注欄位缺失，重新執行 Cell 10（全量標注）
3. 確認推送成功（`--push-to-hub` 沒有報錯）

---

### E07

**問題**: 標注視覺化圖片中，所有幀都被標記為同一個 stage（顏色單一）

**發生時機**: Cell 9b / Cell 11（確認標注視覺化）

**根本原因**: 子任務描述過於模糊，Qwen3-VL-30B 無法區分不同動作階段

**解決方法**:
1. 讓描述更具體，加入視覺上可辨別的動作細節：

| 過於模糊 | 更具體的描述 |
|---------|------------|
| `"fold"` | `"Grab near side of cloth and fold toward center"` |
| `"pick up"` | `"Both robot hands grasp cloth edge and lift upward"` |
| `"finish"` | `"Place folded cloth flat and release gripper"` |

2. 修改 Cell 7 的 `DENSE_SUBTASKS` 後，**重新執行 Cell 9a**（只跑 10 episodes 測試）
3. 觀看 Cell 6 的 episode 預覽，根據實際動作描述每個階段

---

### E08

**問題**: Cell 14 的預測曲線雜亂，在同一 episode 內大幅震盪而非遞增

**發生時機**: Cell 14（預測視覺化）

**根本原因**: 
- SARM 訓練 steps 不足（loss 未收斂）
- 或標注品質差（subtask 邊界不準確）

**解決方法**:

1. 先確認訓練 loss 是否收斂。找到訓練日誌：
```python
import glob
logs = sorted(glob.glob('outputs/train/sarm_folding_dense/**/*.log', recursive=True))
for l in logs[-3:]:
    print(l)
```

2. 若 loss 仍在下降，增加訓練步數重訓：
```python
# 在 Cell 12 把 --steps=5000 改為 --steps=10000
```

3. 若 loss 已收斂但曲線仍不好，問題在標注品質。回到 Cell 7 改善子任務描述後重新標注。

---

### E09

**錯誤訊息**
```
AssertionError: codebase_version v2.0 not supported
ValueError: Dataset version mismatch
```

**發生時機**: Cell 5（dataset inspection）或 Cell 12（訓練）

**根本原因**: LeRobot 資料集格式版本不相容。SARM 需要 v3.0，但某些舊資料集是 v2.0。

**解決方法**: 
`lerobot/high_quality_folding` 已是 v3.0，若仍報此錯，確認：
1. `SOURCE_DATASET = 'lerobot/high_quality_folding'`（不是其他資料集）
2. LeRobot 安裝是最新版（重新執行 Cell 2）

---

### E10

**問題**: Colab session 斷線，標注中途中斷

**發生時機**: Cell 10（全量標注，3–6 小時）

**根本原因**: Colab Pro 最長 session 約 12–24 小時，但 VLM 標注耗時長

**解決方法**:
1. **直接重跑 Cell 10**，`--skip-existing` 會自動跳過已完成的 episodes
2. 斷線後需要重新執行的 Cells:
   - Cell 1（重設變數）
   - Cell 4（重新登入 HF）
   - Cell 10（繼續標注）
3. 確認已標注的 episodes 數量：
```python
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
# 列出 episodes 目錄的 parquet 數量
files = list(api.list_repo_files(ANNOTATED_REPO_ID, repo_type="dataset"))
ep_files = [f for f in files if f.startswith('episodes/') and f.endswith('.parquet')]
print(f"已標注 episodes 數: {len(ep_files)}")
```

---

### E11

**錯誤訊息**
```
TypeError: 'numpy.ndarray' object is not callable
```

**發生時機**: Cell 5（資料集預覽），執行 `dataset.meta.tasks.values()`

**根本原因**: LeRobot v3.0 的 `dataset.meta.tasks` 回傳型別不固定（可能是 dict、list 或 numpy array），不能直接呼叫 `.values()`。

**解決方法**: 改用型別判斷再迭代：
```python
tasks = dataset.meta.tasks
if hasattr(tasks, "items"):
    for idx, task in tasks.items():
        print(f"  [{idx}] {task}")
else:
    for idx, task in enumerate(tasks):
        print(f"  [{idx}] {task}")
```

同理，Cell 5b 的 `dataset.meta.episodes` 也需要判斷是 DataFrame 還是 dict：
```python
ep_meta = dataset.meta.episodes
if isinstance(ep_meta, pd.DataFrame):
    episode_df = ep_meta[["episode_index", "task_index"]].copy()
elif hasattr(ep_meta, "items"):
    episode_df = pd.DataFrame([
        {"episode_index": ep, "task_index": info.get("task_index", -1)}
        for ep, info in ep_meta.items()
    ])
```

---

### E12

**錯誤訊息**
```
FileNotFoundError: [Errno 2] No such file or directory:
 '/root/.cache/huggingface/lerobot/hub/datasets--allenai--19012026-charging-01/snapshots/<sha>/videos/observation.images.right/chunk-000/file-000.mp4'
```

**發生時機**: Cell 4b（六個 charging dataset 合併），在 `aggregate_datasets()` 進入 `Copy data and videos` 階段時

**根本原因**: `LeRobotDatasetMetadata(repo_id)` 內部呼叫 `snapshot_download(..., allow_patterns=[...meta...])`，只下載 `meta/*` 小檔。但 `aggregate_datasets()` 接著會直接從本地讀 `videos/*.mp4` 與 `data/*.parquet`（透過 `src_meta.root / DEFAULT_VIDEO_PATH`）。本地沒這些大檔 → 炸。

**解決方法（兩個關鍵點都要做對）**:

1. **`cache_dir` 必須是 `HF_LEROBOT_HUB_CACHE`**（= `~/.cache/huggingface/lerobot/hub`）。LeRobotDatasetMetadata 跟 aggregate 都認這個路徑，若用 HF 預設的 `~/.cache/huggingface/hub`，aggregate 找不到檔案。
2. **`revision` 必須跟 LeRobotDatasetMetadata 解析出的一樣**。LeRobotDatasetMetadata 內部用 `CODEBASE_VERSION = "v3.0"`，再透過 `get_safe_version` 解析到對應 commit SHA。若 snapshot_download 不指定 revision，預設下載 `main`，會落到「不同的 `snapshots/{sha}/` 子資料夾」，aggregate 還是找不到 mp4。

正確寫法：先建 metadata 拿 `meta.revision`，再用同一個 revision 完整下載：

```python
from huggingface_hub import snapshot_download
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE

for repo in SOURCE_REPO_IDS:
    meta = LeRobotDatasetMetadata(repo)         # 已 cache 過的 meta，快
    snapshot_download(
        repo_id=repo,
        repo_type='dataset',
        revision=meta.revision,                  # ★ 跟 metadata 同一 revision
        cache_dir=HF_LEROBOT_HUB_CACHE,
        token=HF_TOKEN,
    )

aggregate_datasets(
    repo_ids=SOURCE_REPO_IDS,
    aggr_repo_id=MERGED_DATASET,
    aggr_root=Path('/content/merged_dataset'),
)
```

**Symptom 辨識**：第一次嘗試 fix 後，錯誤路徑的 SHA（例如 `eed856cb...`）跟原本失敗時完全一樣 — 這就是 revision 不對齊的證據（snapshot_download 下到了 `main` 的 SHA，metadata 在另一個 SHA 找檔案）。

**磁碟估算**: 六個 source 合計約 24 GB（data + videos），加上 `/content/merged_dataset` 約 24 GB 的合併輸出，總共 ~50 GB。Colab A100 Pro（~166 GB 磁碟）足夠，Free tier 可能不夠。

**Cell 4b 已內建此修正**（snapshot_download 迴圈在 aggregate 之前），重跑 Cell 4b 即可。

---

### E13

**錯誤訊息**
```
TypeError: data type 'list<element: double>[pyarrow]' not understood
  File ".../pandas/core/dtypes/common.py", line 1645, in pandas_dtype
    npdtype = np.dtype(dtype)
```
傳遞鏈：`aggregate_datasets(...)` → `aggregate_data(...)` → `pd.read_parquet(src_path)` 內部某處 → `np.dtype(...)` 認不得 ArrowDtype。

**發生時機**: Cell 4b（六個 charging dataset 合併），進到 `Copy data and videos` 階段、E12 修好之後

**根本原因**: 較新的 `pyarrow`（約 15+）與 `pandas` 2.x 配對下，`pd.read_parquet` 預設把 list/array 欄位（LeRobot v3.0 dataset 的 `action`、`observation.state` 都是 14-dim float array）讀成 `pd.ArrowDtype('list<element:double>')` 擴展型別。LeRobot 的 `aggregate.py` 內部會把 column dtype 餵給 `np.dtype()`，這個 ArrowDtype 無法被 numpy 解析。

**解決方法（v4，雙保險 + diagnostic）**: monkey-patch `pd.read_parquet`：
1. 強制 `dtype_backend='numpy_nullable'`，從讀的源頭避開 ArrowDtype
2. 萬一還是有 ArrowDtype 欄位，**用 `list(df[col])` 強制 materialize 重建 Series**（不是 `astype(object)`——v3 的這個寫法產生「dtype 報告 object 但底層仍是 ArrowExtensionArray」的偽轉換，下游 `to_parquet` / `update_data_df` 仍然炸）
3. `df.index` 也可能是 ArrowDtype，要一併處理
4. 計算 patch 被呼叫次數，用於確認真的 fire 過

```python
import pandas as pd

_orig_read_parquet = pd.read_parquet
_calls = {'n': 0}

def _read_parquet_no_arrow(*args, **kwargs):
    _calls['n'] += 1
    kwargs.setdefault('dtype_backend', 'numpy_nullable')
    try:
        df = _orig_read_parquet(*args, **kwargs)
    except TypeError:
        kwargs.pop('dtype_backend', None)
        df = _orig_read_parquet(*args, **kwargs)
    for col in df.columns:
        if hasattr(df[col].dtype, 'pyarrow_dtype'):
            df[col] = list(df[col])  # 真 materialize，不是 astype(object)
    if hasattr(df.index.dtype, 'pyarrow_dtype'):
        df.index = pd.Index(list(df.index), name=df.index.name)
    return df

pd.read_parquet = _read_parquet_no_arrow
try:
    aggregate_datasets(repo_ids=..., aggr_repo_id=..., aggr_root=...)
finally:
    pd.read_parquet = _orig_read_parquet
    print(f'patch 共呼叫 {_calls["n"]} 次')   # 若為 0 表示沒生效
```

**v3 → v4 教訓**: 同一個錯誤訊息再次出現代表前一個假設（`astype(object)` 把 ArrowDtype 完全換掉）是錯的。每一次「再 patch 一層」前先預想：這個 fix 是不是只是改了個表象指標？**用 `list(df[col])` 是真的把值複製成 Python list，`astype(object)` 在 pandas 某些版本只是改 dtype 標籤**。

**v4 → v5 教訓**: v4 加了 diagnostic counter，從 traceback 看到錯誤在 `_orig_read_parquet(*args, **kwargs)` 行炸——**pd.read_parquet 本身就壞**，根本沒回到後處理。`dtype_backend='numpy_nullable'` 也救不了（fallback 拿掉它仍同錯）。最終解法是**完全繞過 `pd.read_parquet`**：

```python
import pandas as pd
import pyarrow.parquet as pq

_orig_read_parquet = pd.read_parquet

def _read_parquet_via_pyarrow(*args, **kwargs):
    path = args[0] if args else (kwargs.get('path') or kwargs.get('path_or_buf'))
    columns = kwargs.get('columns', None)
    table = pq.read_table(str(path), columns=columns)
    df = table.to_pandas()       # 預設 types_mapper=None → list 欄位變 object dtype
    # 保險網
    for col in df.columns:
        if hasattr(df[col].dtype, 'pyarrow_dtype'):
            df[col] = list(df[col])
    if hasattr(df.index.dtype, 'pyarrow_dtype'):
        df.index = pd.Index(list(df.index), name=df.index.name)
    return df

pd.read_parquet = _read_parquet_via_pyarrow
try:
    aggregate_datasets(...)
finally:
    pd.read_parquet = _orig_read_parquet
```

**為什麼 pyarrow 直讀可以**: `pq.read_table(path).to_pandas()` 預設 `types_mapper=None`，pyarrow 走的是傳統轉換——list 欄位 → object dtype（Python list），不會觸發 pandas 內部嘗試把 ArrowDtype 餵給 `np.dtype()` 的那條壞掉 code path。`pd.read_parquet` 內部會強制走 ArrowDtype 路徑（不論你怎麼設 dtype_backend），所以必須完全跳過它。

**通用 debug 原則**: traceback 第一個帶有「我寫的程式碼路徑」的行就是修正的著力點。v4 traceback 出現 `_orig_read_parquet(*args, **kwargs)` 在我的 patch 函式裡——那一行就是「無法繞過的內部錯誤」，必須換實作而不是 wrap。

**v5 → v6 教訓（最終突破）**: v5 展開的 traceback 把整個壞掉的鏈條完整顯示出來：

```
pyarrow/pandas_compat.py:783  _get_extension_dtypes(table, all_columns, types_mapper)
pyarrow/pandas_compat.py:862  pandas_dtype = _pandas_api.pandas_dtype(dtype)
pandas/.../common.py:1645     npdtype = np.dtype(dtype)
TypeError: data type 'list<element: double>[pyarrow]' not understood
```

**真正的根因**：不是 pandas 也不是 pyarrow 的 conversion bug，而是這個 dataset 的 **parquet 檔本身在 schema 裡嵌入了 pandas metadata**（一段 JSON 寫死 column dtype 是 `list<element: double>[pyarrow]`）。pyarrow 的 `to_pandas()` 預設 `ignore_metadata=False` 會去解析這段 metadata 並丟給 pandas_dtype → np.dtype → 炸。

不是所有 parquet 都中招——`tasks.parquet` 跟 `data/*.parquet` 都沒中（v5 diagnostic 確認讀過 16 次成功），只有 `meta/episodes/*.parquet` 帶這個壞 metadata。差別是寫入時的工具版本不同。

**v6 解法**: 用 `to_pandas(ignore_metadata=True)` 跳過解析；但這會丟掉 `tasks.parquet` 需要的 index 資訊，所以**先試 default，TypeError 時再 retry**：

```python
import pandas as pd
import pyarrow.parquet as pq

_orig_read_parquet = pd.read_parquet

def _read_parquet_via_pyarrow(*args, **kwargs):
    path = args[0] if args else (kwargs.get('path') or kwargs.get('path_or_buf'))
    columns = kwargs.get('columns', None)
    table = pq.read_table(str(path), columns=columns)
    try:
        df = table.to_pandas()                          # 保留 index（tasks.parquet 需要）
    except TypeError as e:
        if '[pyarrow]' in str(e) or 'not understood' in str(e):
            df = table.to_pandas(ignore_metadata=True)   # 跳過壞掉的 pandas metadata
        else:
            raise
    for col in df.columns:
        if hasattr(df[col].dtype, 'pyarrow_dtype'):
            df[col] = list(df[col])
    return df

pd.read_parquet = _read_parquet_via_pyarrow
```

**通用原則**: 同樣的錯誤訊息背後可能有不同的觸發點（讀 data parquet vs 讀 meta parquet）。v5 的 diagnostic counter 揭露了「前 16 次都成功，第 17 次才炸」這個關鍵資訊——沒有 diagnostic 永遠以為是同一個地方炸。**任何重試的 patch 都應該帶 instrumentation**（call count、首次出錯路徑等），不然就只是在黑盒裡盲改。

**為什麼不降版本**: pyarrow / pandas 版本是被 lerobot[sarm] 跟其他依賴共同決定的，硬降會打到別的元件。Monkey-patch 局部生效、用完即還原，最小副作用。

**Cell 4b 已內建此修正**，重跑 Cell 4b 即可。

---

### E14

**問題**: Hub 上有 `{user}/charging_bimanual_merged` repo 但裡面沒有 `meta/info.json`（先前失敗執行留下的 ghost repo）

**發生時機**: Cell 4b 重跑時，`api.dataset_info(MERGED_DATASET)` 成功 → 誤判 `merged_exists=True` → 跳過合併 → 最後 `LeRobotDatasetMetadata(MERGED_DATASET)` 找不到 meta files 而炸

**根本原因**: 「repo 存在於 Hub」不等於「repo 有完整資料」。任何之前到了 `api.create_repo(exist_ok=True)` 但 upload_folder 之前掛掉、或被使用者手動建立的空 repo，都會欺騙這個判斷。

**解決方法**: early-exit 判斷必須驗證 `meta/info.json` 存在（不能只看 repo 是否存在）：
```python
def _merged_is_complete():
    try:
        api.dataset_info(MERGED_DATASET)
    except Exception:
        return False
    try:
        api.hf_hub_download(MERGED_DATASET, 'meta/info.json',
                            repo_type='dataset', token=HF_TOKEN)
        return True
    except Exception:
        return False
```

**Cell 4b 已內建此修正**。如果想徹底重來，到 Hub 網頁刪掉 `{user}/charging_bimanual_merged` repo 即可。

---

### E15

**錯誤訊息**
```
FileNotFoundError: [Errno 2] No such file or directory: '.../meta/info.json'

During handling of the above exception, another exception occurred:

TypeError: HfHubHTTPError.__init__() missing 1 required keyword-only argument: 'response'
```

**發生時機**: Cell 4b 末尾驗證 `LeRobotDatasetMetadata(MERGED_DATASET)` 時，或任何後續 cell 第一次嘗試 load 新建的 merged dataset

**根本原因**: `LeRobotDatasetMetadata(repo_id)` 預設用 `revision="v3.0"`（CODEBASE_VERSION），它會在 Hub 上找這個 git tag。`api.upload_folder(...)` **不會自動建 tag**——剛 push 的 repo 只有 `main` branch，沒有 `v3.0` tag。

`get_safe_version` 找不到 tag 想丟 `RevisionNotFoundError`，但這個 error class 繼承自 `HfHubHTTPError`，其 `__init__` 要求 `response=` 關鍵字參數；某些版本的 lerobot/huggingface_hub 沒帶這個參數 → 丟 `TypeError`（這是 upstream bug，但治標不治本，真正缺的還是 tag）。

**解決方法**: `upload_folder` 後馬上建 `v3.0` tag：

```python
api.upload_folder(
    folder_path=str(aggr_root),
    repo_id=MERGED_DATASET,
    repo_type='dataset',
)
api.create_tag(
    repo_id=MERGED_DATASET, repo_type='dataset',
    tag='v3.0', exist_ok=True,
)
```

**緊急 workaround**（不重跑合併，到 Hub 網頁手動建）：
1. 開 `https://huggingface.co/datasets/{user}/charging_bimanual_merged`
2. Settings → Tags → New tag
3. Tag name: `v3.0`，From: `main`，Create

**驗證最好用 local root**（剛合併完 `/content/merged_dataset` 還在本地，不需要再從 Hub 拉）：

```python
local_root = Path('/content/merged_dataset')
if (local_root / 'meta' / 'info.json').exists():
    meta = LeRobotDatasetMetadata(MERGED_DATASET, root=local_root)
else:
    meta = LeRobotDatasetMetadata(MERGED_DATASET)  # 需要 tag
```

**Cell 4b 已內建此修正**。

---

## 回報新錯誤的格式

遇到不在上方列表的錯誤時，請提供：

```
【報錯】
Cell: Cell XX（XXX 步驟）
錯誤訊息（完整複製）:
  <error message here>

Traceback:
  <full traceback here>

已嘗試:
  - 步驟1
  - 步驟2
```

Claude 會診斷並在此文件新增對應條目（Exx）。
