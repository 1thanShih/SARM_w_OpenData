"""
Recompute stats.json for Lebruhbruh/SARM-opendata-annotated-fixed.
- Recomputes all scalar/vector stats from the data parquet (585,425 frames, 273 eps)
- Adds missing subtask_index and task_index_high_level
- Preserves image stats (cannot recompute without decoding all video frames)
- Updates count to match actual frame count
"""

import os, json, numpy as np, pandas as pd
from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=HF_TOKEN)
REPO = "Lebruhbruh/SARM-opendata-annotated-fixed"

# ── 1. Load current stats.json (to preserve image stats) ─────────────────────
stats_path = hf_hub_download(REPO, "meta/stats.json", repo_type="dataset", token=HF_TOKEN)
with open(stats_path) as f:
    old_stats = json.load(f)

IMAGE_KEYS = {"observation.images.top", "observation.images.left", "observation.images.right"}

# ── 2. Load data parquet ──────────────────────────────────────────────────────
print("Downloading data parquet…")
data_path = hf_hub_download(REPO, "data/chunk-000/file-000.parquet", repo_type="dataset", token=HF_TOKEN)
df = pd.read_parquet(data_path)

import pyarrow as pa
# Convert ArrowDtype list columns to object (LeRobot compat)
for col in df.columns:
    dt = df[col].dtype
    if hasattr(dt, "pyarrow_dtype") and (
        pa.types.is_list(dt.pyarrow_dtype) or pa.types.is_large_list(dt.pyarrow_dtype)
    ):
        df[col] = df[col].astype(object)

print(f"Loaded {len(df):,} frames, columns: {df.columns.tolist()}")

N = len(df)
print(f"Episode range: {df['episode_index'].min()} – {df['episode_index'].max()}")
print(f"subtask_index unique: {sorted(df['subtask_index'].unique())}")

# ── 3. Helper to compute stats for one feature ────────────────────────────────
QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
QNAMES    = ["q01", "q10", "q50", "q90", "q99"]

def compute_feature_stats(series_or_2d):
    """
    series_or_2d: pd.Series of scalars, or pd.Series of lists/arrays (vector feature).
    Returns dict matching LeRobot stats.json schema (all values are lists).
    """
    if isinstance(series_or_2d.iloc[0], (list, np.ndarray)):
        mat = np.stack(series_or_2d.values)  # (N, D)
    else:
        mat = series_or_2d.values.reshape(-1, 1)  # (N, 1)

    result = {
        "min":   mat.min(axis=0).tolist(),
        "max":   mat.max(axis=0).tolist(),
        "mean":  mat.mean(axis=0).tolist(),
        "std":   mat.std(axis=0).tolist(),
        "count": [N],
    }
    for qname, q in zip(QNAMES, QUANTILES):
        result[qname] = np.quantile(mat, q, axis=0).tolist()
    return result

# ── 4. Columns to recompute (non-image) ──────────────────────────────────────
VECTOR_COLS  = ["action", "observation.state"]
SCALAR_COLS  = ["timestamp", "frame_index", "episode_index", "index",
                "task_index", "subtask_index", "task_index_high_level"]

new_stats = {}

print("\nComputing vector feature stats…")
for col in VECTOR_COLS:
    print(f"  {col}…", end=" ", flush=True)
    new_stats[col] = compute_feature_stats(df[col])
    print(f"shape={np.stack(df[col].values).shape}")

print("Computing scalar feature stats…")
for col in SCALAR_COLS:
    if col not in df.columns:
        print(f"  {col}: NOT IN PARQUET, skipping")
        continue
    print(f"  {col}…", end=" ", flush=True)
    new_stats[col] = compute_feature_stats(df[col])
    v = new_stats[col]
    print(f"min={v['min'][0]:.4g}  max={v['max'][0]:.4g}  mean={v['mean'][0]:.4g}")

# ── 5. Preserve image stats (update count only) ───────────────────────────────
print("\nPreserving image stats (updating count)…")
for k in IMAGE_KEYS:
    if k in old_stats:
        img_stat = dict(old_stats[k])
        img_stat["count"] = [N]
        new_stats[k] = img_stat
        print(f"  {k}: count updated to {N}")

# ── 6. Sanity checks ──────────────────────────────────────────────────────────
print("\n── Sanity checks ──")
print(f"subtask_index: min={new_stats['subtask_index']['min'][0]}  max={new_stats['subtask_index']['max'][0]}  "
      f"mean={new_stats['subtask_index']['mean'][0]:.4f}")
print(f"episode_index: min={new_stats['episode_index']['min'][0]}  max={new_stats['episode_index']['max'][0]}")
print(f"frame count:   {new_stats['timestamp']['count'][0]:,}")
assert new_stats['subtask_index']['min'][0] == 0, "subtask_index min should be 0"
assert new_stats['subtask_index']['max'][0] == 3, "subtask_index max should be 3"

# ── 7. Save and push ──────────────────────────────────────────────────────────
out_path = "/tmp/stats_recomputed.json"
with open(out_path, "w") as f:
    json.dump(new_stats, f, indent=2)
print(f"\nWrote {out_path}")

api.upload_file(
    path_or_fileobj=out_path,
    path_in_repo="meta/stats.json",
    repo_id=REPO,
    repo_type="dataset",
    commit_message="Recompute stats.json: 273 eps / 585,425 frames; add subtask_index & task_index_high_level",
)
print("✓ stats.json pushed to Hub")
