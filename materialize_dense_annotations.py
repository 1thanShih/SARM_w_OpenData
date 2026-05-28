"""
把 meta/lerobot_annotations.json 的 subtask 內容 materialize 進
meta/episodes/*.parquet 的 dense/sparse/plain subtask_* 欄位。

背景:
  Lebruhbruh/SARM-opendata-annotated-fixed 的 meta/episodes parquet 裡:
    - 8 個 episode (new index 0-7) 有 dense_subtask_* 欄位 (但值是錯的)
    - 265 個 episode (new index 8-272) 是 NULL
    - sparse / plain subtask_* 同樣 265/273 NULL
  → SARM 訓練時 find_stage_and_tau 對 265/273 episodes 回傳 (0, 0.0)
  → dense target ≡ 0 → model 塌成 0-generator

修法:
  從 lerobot_annotations.json (273 個 episode 完整) 重建這 9 個欄位:
    dense_subtask_names / start_times / end_times / start_frames / end_frames
    sparse_subtask_* (single 'task' stage 涵蓋全 episode)
    subtask_* (= sparse_*)

用法:
  HF_TOKEN=<write_token> python3 materialize_dense_annotations.py
  (預設 DRY_RUN=1 只本地修不 push；DRY_RUN=0 才會 push 回 Hub)
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download

REPO = "Lebruhbruh/SARM-opendata-annotated-fixed"
FPS = 30
WORK_DIR = Path("/tmp/sarm_materialize")
DRY_RUN = os.environ.get("DRY_RUN", "1") == "1"
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def build_subtask_columns(ann_subtasks: list[dict], ep_length: int) -> dict:
    """Build the 5 dense + 5 sparse + 5 plain columns from a list of
    {label, start, end} annotation segments.
    """
    if not ann_subtasks:
        names, starts, ends = [], [], []
    else:
        segs = sorted(ann_subtasks, key=lambda s: s["start"])
        names = [s["label"] for s in segs]
        starts = [float(s["start"]) for s in segs]
        ends = [float(s["end"]) for s in segs]

    start_frames = [int(round(s * FPS)) for s in starts]
    end_frames = [int(round(e * FPS)) for e in ends]
    # Clamp last frame to ep_length - 1
    if end_frames:
        end_frames[-1] = min(end_frames[-1], ep_length - 1)

    sparse_end_time = (ep_length - 1) / FPS
    sparse_end_frame = ep_length - 1

    return {
        # dense (the real annotations)
        "dense_subtask_names": names,
        "dense_subtask_start_times": [int(round(s)) for s in starts],
        "dense_subtask_end_times": [int(round(e)) for e in ends],
        "dense_subtask_start_frames": start_frames,
        "dense_subtask_end_frames": end_frames,
        # sparse (single 'task' stage spanning whole episode)
        "sparse_subtask_names": ["task"],
        "sparse_subtask_start_times": [0],
        "sparse_subtask_end_times": [int(round(sparse_end_time))],
        "sparse_subtask_start_frames": [0],
        "sparse_subtask_end_frames": [sparse_end_frame],
        # plain (= sparse, kept for backward compat)
        "subtask_names": ["task"],
        "subtask_start_times": [0],
        "subtask_end_times": [int(round(sparse_end_time))],
        "subtask_start_frames": [0],
        "subtask_end_frames": [sparse_end_frame],
    }


def main():
    if not DRY_RUN and not HF_TOKEN:
        sys.exit("ERROR: DRY_RUN=0 but HF_TOKEN is empty")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=HF_TOKEN or None)

    print(f"[1/6] Download meta/ from {REPO} ...")
    local_dir = Path(snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns=["meta/**"],
        local_dir=str(WORK_DIR / "snapshot"),
        token=HF_TOKEN or None,
    ))

    ann = json.loads((local_dir / "meta" / "lerobot_annotations.json").read_text())
    ann_eps = ann["episodes"]
    ann_keys_sorted = sorted(int(k) for k in ann_eps.keys())
    print(f"      annotations.json: {len(ann_keys_sorted)} episodes (original indices)")

    ep_files = sorted((local_dir / "meta" / "episodes").rglob("*.parquet"))
    print(f"[2/6] Found {len(ep_files)} episode parquet file(s)")

    print("[3/6] Build new columns from annotations.json ...")
    cumulative_new_idx = 0
    for ep_file in ep_files:
        df = pq.read_table(ep_file).to_pandas()
        print(f"      {ep_file.name}: {len(df)} rows")

        new_cols = {col: [] for col in [
            "dense_subtask_names", "dense_subtask_start_times", "dense_subtask_end_times",
            "dense_subtask_start_frames", "dense_subtask_end_frames",
            "sparse_subtask_names", "sparse_subtask_start_times", "sparse_subtask_end_times",
            "sparse_subtask_start_frames", "sparse_subtask_end_frames",
            "subtask_names", "subtask_start_times", "subtask_end_times",
            "subtask_start_frames", "subtask_end_frames",
        ]}

        for i, row in df.iterrows():
            new_ep_idx = int(row["episode_index"])
            ep_length = int(row["length"])
            # new_ep_idx in this file → position cumulative_new_idx + i in ann_keys_sorted
            global_pos = cumulative_new_idx + i
            if global_pos >= len(ann_keys_sorted):
                sys.exit(f"ERROR: parquet new ep {new_ep_idx} maps to position "
                         f"{global_pos} but annotations.json only has {len(ann_keys_sorted)} eps")
            orig_key = ann_keys_sorted[global_pos]
            ann_subtasks = ann_eps[str(orig_key)].get("subtasks", [])
            built = build_subtask_columns(ann_subtasks, ep_length)
            for col, val in built.items():
                new_cols[col].append(val)

        # Replace columns
        for col, vals in new_cols.items():
            df[col] = vals

        # Write back
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), ep_file)
        print(f"      ✓ rewrote {ep_file.name}")
        cumulative_new_idx += len(df)

    print(f"      total processed: {cumulative_new_idx} episodes "
          f"(annotations.json has {len(ann_keys_sorted)})")
    assert cumulative_new_idx == len(ann_keys_sorted), \
        f"Mismatch: parquet={cumulative_new_idx} vs annotations.json={len(ann_keys_sorted)}"

    print("[4/6] Local sanity check ...")
    for ep_file in ep_files:
        df = pq.read_table(ep_file).to_pandas()
        for col in ["dense_subtask_names", "sparse_subtask_names", "subtask_names"]:
            null_cnt = df[col].isna().sum() if df[col].dtype == object else 0
            # For list dtype it's harder; check whether each entry is empty list / None
            none_or_empty = sum(1 for v in df[col]
                                if v is None or (hasattr(v, "__len__") and len(v) == 0))
            assert none_or_empty == 0, f"{col} still has {none_or_empty} NULL/empty rows in {ep_file.name}"
            print(f"      ✓ {ep_file.name}::{col}: 0 NULL/empty")
        # Coverage check
        bad_cov = 0
        for _, row in df.iterrows():
            length = int(row["length"])
            end_f = list(row["dense_subtask_end_frames"])
            if end_f and end_f[-1] < length * 0.5:
                bad_cov += 1
        if bad_cov:
            print(f"      ⚠ {bad_cov} eps have dense_end < 50% of episode length (check annotations)")

    if DRY_RUN:
        print("\n[5/6] DRY_RUN=1: 跳過 push。檢查 /tmp/sarm_materialize/snapshot/meta/episodes/ 後\n"
              "      重跑 DRY_RUN=0 HF_TOKEN=<write_token> python3 materialize_dense_annotations.py")
        print(f"[6/6] Done (local only).")
        return

    print(f"[5/6] Push meta/episodes/ back to {REPO} ...")
    api.upload_folder(
        folder_path=str(local_dir / "meta" / "episodes"),
        path_in_repo="meta/episodes",
        repo_id=REPO,
        repo_type="dataset",
        commit_message="Materialize dense/sparse/plain subtask_* columns from lerobot_annotations.json",
    )
    print("      ✓ pushed")

    print("[6/6] Re-pin v3.0 tag to new HEAD ...")
    try:
        api.delete_tag(REPO, tag="v3.0", repo_type="dataset")
    except Exception as e:
        print(f"      (delete_tag warning: {type(e).__name__}: {e})")
    api.create_tag(REPO, tag="v3.0", repo_type="dataset", exist_ok=True)
    print("      ✓ v3.0 tag moved")

    print(f"\n✓ Done. Dataset: https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main()
