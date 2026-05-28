# CLAUDE.md — SARM Training Project

## Colab Notebook Code Rules

當修改或新增 `SARM_Training_Colab.ipynb` 的任何 cell 程式碼時，**必須**在提供給使用者之前完成以下驗證：

### 1. 語法驗證（必做）
每個 cell 的程式碼必須先在本地用 Python 解析器檢查語法：
```bash
python3 -c "import ast; ast.parse(open('/tmp/cell_code.py').read()); print('syntax OK')"
```
或直接 `python3 -c "..."` 跑一遍不依賴外部服務的部分。

### 2. Notebook JSON 驗證（必做）
修改 `.ipynb` 後，確認 JSON 可正確 load：
```bash
python3 -c "import json; json.load(open('SARM_Training_Colab.ipynb')); print('JSON OK')"
```

### 3. 路徑與 API 不假設（必做）
- 不要 hardcode 套件安裝路徑（如 `/usr/local/lib/...`）— 改用 `glob` 或 `find` 動態搜尋
- 不要假設 CLI entry point 存在（如 `lerobot-train`）— 用 `shutil.which()` 加 fallback
- 不要假設 HuggingFace 資料集的 metadata 檔名（如 `meta/tasks.jsonl`）存在 — 用 try/except

### 4. subprocess 輸出規則
- 長時間執行的指令（>30 秒）**不得**使用 `capture_output=True`，必須讓輸出即時顯示
- 短暫的確認指令（版本查詢等）可以用 `capture_output=True`

### 5. f-string 與多行字串
- Cell source 內若有 f-string 含花括號，在 `json.dump` 前必須確認不會被 JSON 序列化破壞
- 避免在 `subprocess.run([sys.executable, '-c', ...])` 的字串裡使用多行隱式拼接

### 6. 進度回饋（必做）
**任何預估執行時間 >30 秒的 cell 都必須提供進度回饋**，讓使用者隨時看得到目前在做什麼、已經跑多久。

- Notebook 已內建 `progress_stage(title, steps, heartbeat_seconds)` context manager（定義於 Cell 1b）。新增或修改長時 cell 時請直接使用：
  ```python
  with progress_stage(
      '階段名稱（預計 X 分鐘）',
      steps=['第 1 小步驟', '第 2 小步驟', ...],
      heartbeat_seconds=30,
  ):
      # 長時邏輯：subprocess、aggregate_datasets、copytree、訓練...
      ...
  ```
- 三項必含：
  - **開頭**印階段標題與分點步驟清單（讓使用者知道接下來會發生什麼）
  - **執行中**背景 thread 每 30 秒印一次 `⏱  已執行 H:MM:SS` 心跳
  - **結尾**印 `✓ ... 完成，總耗時 H:MM:SS`（失敗時印 `✗` 與已執行時間）
- subprocess cell（annotation、training、viz）也要包，雖然底層工具自己會印進度——心跳的價值在於 Colab 斷線重連時仍能立即看到「還在跑」。
- 若新增的長時 cell 順序在 Cell 1b 之前，請改成在該 cell 內就近定義 helper，或調整 cell 順序。

## 除錯流程
遇到報錯時，更新 `doc/debug.md` 加入新的 Exx 條目，包含：
- 錯誤訊息
- 根本原因
- 解決方法（含可直接執行的程式碼）

### 預測式除錯（重要）
當一個 cell 連續報錯時，**不要**單純對著當下這個 traceback 修就完事。每修一層、回給使用者**之前**，必須先預想：

1. 我這次修了 A，會不會打到 B？（例：monkey-patch `pd.read_parquet` 解決 ArrowDtype list 問題，但也順手把 int 欄位也轉成 object，會打到後面 `Index.take()` 的 int-array 假設）
2. 同樣 cell 還有哪些別的 code path（write、metadata read、video concat...）沒被這次修正覆蓋到？
3. 若同一個錯訊息 SHA / 路徑 / 行號**完全一樣**，代表前一次的假設根本沒生效，要重新思考方向，不要只是「再 patch 一層」。

如果想得到 N 個可能失敗點，**全部列出來給使用者看**，再請使用者貼下一個錯訊息——比起連改三次反而省時間。

## LeRobot dataset 操作地雷（重要）

操作 LeRobot v3.0 dataset（讀、合併、aggregate、push）時，反覆踩過的坑：

### 1. Cache 路徑不是 HF 預設
- LeRobot 用 `HF_LEROBOT_HOME = ~/.cache/huggingface/lerobot/`，hub cache 是 `HF_LEROBOT_HUB_CACHE = HF_LEROBOT_HOME/hub`
- **不是** HF 標準的 `~/.cache/huggingface/hub/`
- `from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE` 拿來用，呼叫 `snapshot_download(cache_dir=HF_LEROBOT_HUB_CACHE, ...)`

### 2. Revision 不是 main
- `LeRobotDatasetMetadata` 內部 `self.revision = get_safe_version(repo_id, CODEBASE_VERSION="v3.0")`
- 解析到那個 dataset 的 `v3.0` tag 對應的 commit SHA
- 直接 `snapshot_download(repo_id)` 預設下載 `main` → 落在不同 snapshot 資料夾 → aggregate 找不到檔案
- 正確：先建 `meta = LeRobotDatasetMetadata(repo)` 拿 `meta.revision`，再 `snapshot_download(revision=meta.revision, ...)`

### 3. `LeRobotDatasetMetadata(repo)` 只下載 `meta/*`
- 用 `allow_patterns=[...meta...]` 過濾，**不**下載 videos / data parquet
- 後續 `aggregate_datasets()`、`LeRobotDataset(repo)` 會嘗試從本地讀大檔
- 需要的話自己呼叫 `snapshot_download` 把完整 repo 下下來（用對 cache_dir + revision）

### 4. ArrowDtype 不相容
- 新版 `pyarrow + pandas 2.x` 預設把 list 欄位（如 `action`、`observation.state`）讀成 `list<element:T>[pyarrow]` ArrowDtype
- LeRobot 的 `aggregate.py` 內部 `np.dtype(dtype)` 不認得 → `TypeError`
- 修法：monkey-patch `pd.read_parquet` 把 ArrowDtype**只限 list/large_list 類型**轉成 object dtype。**不要**轉 scalar（int/float）pyarrow 欄位，否則 `Index.take`、算術運算可能炸。
  ```python
  import pyarrow as pa
  if hasattr(dt, 'pyarrow_dtype') and (
      pa.types.is_list(dt.pyarrow_dtype) or pa.types.is_large_list(dt.pyarrow_dtype)
  ):
      df[col] = df[col].astype(object)
  ```

### 5. 「Hub 上 repo 存在」≠「repo 有完整資料」
- 不要只用 `api.dataset_info(repo)` 判斷 early-exit；之前失敗的執行可能留下空 repo
- 應該檢查 `meta/info.json` 是否存在且 `total_episodes > 0`：
  ```python
  try:
      api.hf_hub_download(repo, 'meta/info.json', repo_type='dataset')
      info = json.load(open(...))
      merged_ok = info.get('total_episodes', 0) > 0
  except Exception:
      merged_ok = False
  ```

### 6. 不要猜 LeRobot API，先讀 source
- 直接 `urllib.request` 抓 `https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/...` 確認簽名跟行為
- 重點檔案：`src/lerobot/datasets/aggregate.py`、`dataset_metadata.py`、`utils.py`、`utils/constants.py`
- 看完函式簽名、回傳型別、它怎麼讀本地路徑（`src_meta.root / DEFAULT_DATA_PATH.format(...)`）再決定怎麼修

### 7. SARM 標注必須寫進 `meta/episodes/*.parquet`，不是 `lerobot_annotations.json`
- SARM 訓練讀的是 `dataset.meta.episodes`（也就是 `meta/episodes/chunk-XXX/file-XXX.parquet`）的這幾欄：
  - `dense_subtask_names`、`dense_subtask_start_frames`、`dense_subtask_end_frames`（或 fallback 的 `subtask_names` / `subtask_start_frames` / `subtask_end_frames`）
- `meta/lerobot_annotations.json` 只是 lerobot-annotate UI 的工作檔，**訓練不會讀它**
- `processor_sarm.py::_load_episode_annotations` 拿不到 `dense_subtask_names`（NULL）就直接 `return None, None, None`；接著 `find_stage_and_tau(subtask_names=None, ...)` 直接走 `pass` 分支回傳 `(stage_idx=0, tau=0.0)` → dense target = 0.0
- 後果：未標注的 episode 在訓練時 dense target 全是 0；如果未標注佔大宗，整個 dense head 會塌成「永遠輸出 0」
- **預防驗證**：push 完 dataset 後，下載 `meta/episodes/chunk-000/file-000.parquet`，檢查 `dense_subtask_names` 欄位的 NULL 數量：
  ```python
  import pandas as pd
  df = pd.read_parquet('meta/episodes/chunk-000/file-000.parquet')
  null_cnt = df['dense_subtask_names'].isna().sum()
  assert null_cnt == 0, f'{null_cnt}/{len(df)} episodes 缺 dense_subtask_names → SARM 會塌'
  ```
- 已知踩過：`Lebruhbruh/SARM-opendata-annotated-fixed`（273 eps，只有 ep 0–7 有 dense 欄位，265 個是 NULL → 訓練 5000 steps 後預測全 0）
- 重 export 時要把 `lerobot_annotations.json` 的內容 materialize 到 `meta/episodes/*.parquet` 的 `dense_subtask_*` 欄；只搬 JSON 不夠

## SARM 「預測全 0」的除錯 checklist

看到 visualize 出來的 progress 曲線在所有 frame 都黏在 0，依序排查：

1. **dataset 端標注**：先跑上面 §7 的 NULL 檢查。`dense_subtask_names` 大量 NULL 是最常見也最容易漏的元兇——model config 與 `temporal_proportions_dense.json` 都會看起來完全正常（4 個 stage、proportions sum=1.0），但 per-episode 標注其實是空的。
2. **要看的 episode 有沒有被標注**：visualize cell 指定的 episode index，必須在「有 dense 標注」那批裡。否則就算 model 沒塌，那個 episode 訓練時的 target 也是 0，模型自然學會對它輸出 0。
3. **stage_probs 訊號**：跑 Cell 13z 之類的診斷 cell，看 `stage_probs`：
   - 全部接近 one-hot 在 stage 0（[1, 0, 0, 0]）→ stage head 也塌了，幾乎肯定是 §1 的問題
   - 分布有變化但 progress 仍是 0 → subtask (tau) head 塌了，可能 step 數不足或 loss weight 失衡
   - 4 個 stage 機率均等（~0.25）→ CLIP embedding 沒分辨力，檢查 transformers 5.x patch（`pooler_output` fallback）有沒有套上
4. **model config 對齊**：確認 `reward_model.config.dense_subtask_names`、`num_dense_stages`、`dense_temporal_proportions` 跟 dataset 的 `meta/temporal_proportions_dense.json` 一致；不一致代表 model 與 dataset 不是同一輪 annotation 出來的
