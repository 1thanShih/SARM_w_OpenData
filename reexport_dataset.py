"""
SARM Dataset Re-export Script
==============================
從 HGLLL/SARM-opendata-annotated-1 讀取標注資料，
- 排除有問題的 episodes（40, 247, 22, 6, 8, 55, 79, 111, 113, 134, 266）
- 修正 subtask_index 排列順序（按 stage 執行順序，非字母順序）
- Push 到你自己的 repo

使用方式：
    export HF_TOKEN=hf_xxxx
    export TARGET_REPO=Lebruhbruh/SARM-opendata-annotated-fixed   # 可改名
    python3 reexport_dataset.py
"""

import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, list_repo_files, snapshot_download

# ── 設定 ─────────────────────────────────────────────────────────────────────

SRC_REPO = "HGLLL/SARM-opendata-annotated-1"
TARGET_REPO = os.environ.get("TARGET_REPO", "Lebruhbruh/SARM-opendata-annotated-fixed")
HF_TOKEN = os.environ["HF_TOKEN"]
WORK_DIR = Path("/tmp/sarm_reexport")

# 排除的 episodes
EXCLUDED_EPISODES: set[int] = {
    40,   # labels 是 "1","2","3","4"（錯誤標注）
    247,  # 完全沒有 annotation
    22,   # stage 3 和 stage 4 重疊
    6,    # 影片中缺少 stage 4
    8,    # 影片中缺少 stage 2
    55,   # 影片中缺少 stage 2
    79,   # 影片中缺少 stage 2
    111,  # 影片中缺少 stage 2
    113,  # 影片中缺少 stage 2
    134,  # 影片中缺少 stage 2
    266,  # 影片中缺少 stage 2
}

# ── helpers ──────────────────────────────────────────────────────────────────

def _check():
    if not HF_TOKEN:
        raise SystemExit(
            "錯誤：請先設定 HF_TOKEN 環境變數\n"
            "  export HF_TOKEN=hf_xxxxxxxxxxxx"
        )
    print(f"SRC  : {SRC_REPO}")
    print(f"DST  : {TARGET_REPO}")
    print(f"排除 : {sorted(EXCLUDED_EPISODES)}")
    print()


def _build_subtask_map(episodes: dict) -> dict[str, int]:
    """按各 label 在所有 episode 中的中位出現順序排列，而非字母排序。"""
    label_ranks: dict[str, list[int]] = {}
    for ep_data in episodes.values():
        subtasks_sorted = sorted(ep_data.get("subtasks", []), key=lambda s: float(s.get("start", 0)))
        for rank, seg in enumerate(subtasks_sorted):
            label = seg.get("label", "")
            if label:
                label_ranks.setdefault(label, []).append(rank)

    def _median(label: str) -> float:
        ranks = label_ranks.get(label, [])
        return sorted(ranks)[len(ranks) // 2] if ranks else float("inf")

    all_labels = sorted(label_ranks, key=_median)
    return {label: idx for idx, label in enumerate(all_labels)}


def _assign_subtask_index(timestamps: list[float], subtasks: list[dict], subtask_map: dict[str, int]) -> list[int]:
    values = [-1] * len(timestamps)
    subtasks_sorted = sorted(subtasks, key=lambda s: float(s.get("start", 0)))
    for i, ts in enumerate(timestamps):
        for seg_i, seg in enumerate(subtasks_sorted):
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
            is_last = seg_i == len(subtasks_sorted) - 1
            if (start <= ts < end) or (is_last and ts <= end):
                values[i] = subtask_map.get(seg.get("label", ""), -1)
                break
    return values


def _recompute_temporal_proportions(episodes: dict, subtask_map: dict[str, int]) -> dict[str, float]:
    """從 annotation segments 計算各 stage 佔總時長的比例。"""
    stage_total: dict[str, float] = {}
    grand_total = 0.0
    for ep_data in episodes.values():
        for seg in ep_data.get("subtasks", []):
            label = seg.get("label", "")
            if label in subtask_map:
                dur = float(seg.get("end", 0)) - float(seg.get("start", 0))
                stage_total[label] = stage_total.get(label, 0.0) + dur
                grand_total += dur
    if grand_total == 0:
        return {}
    return {label: total / grand_total for label, total in stage_total.items()}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    _check()

    api = HfApi(token=HF_TOKEN)
    src_dir = WORK_DIR / "src"
    out_dir = WORK_DIR / "out"

    # ── 1. Download source dataset ──────────────────────────────────────────
    print("Step 1/7: 下載來源 dataset（meta + data，不含 videos）...")
    snapshot_download(
        SRC_REPO,
        repo_type="dataset",
        local_dir=src_dir,
        ignore_patterns=["videos/**"],
        token=HF_TOKEN,
    )
    print("  ✓ meta 與 data parquet 已下載完成")

    # ── 2. 複製到 output 目錄 ────────────────────────────────────────────────
    print("Step 2/7: 建立 output 目錄...")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(src_dir, out_dir)

    # ── 3. 過濾 lerobot_annotations.json ────────────────────────────────────
    print("Step 3/7: 過濾 annotations...")
    ann_path = out_dir / "meta" / "lerobot_annotations.json"
    with open(ann_path) as f:
        ann = json.load(f)

    before = len(ann["episodes"])
    ann["episodes"] = {k: v for k, v in ann["episodes"].items()
                       if int(k) not in EXCLUDED_EPISODES}
    after = len(ann["episodes"])
    print(f"  episodes: {before} → {after}（排除 {before - after} 個）")

    with open(ann_path, "w") as f:
        json.dump(ann, f, indent=2)

    # ── 4. 重建 subtask_index 映射 ──────────────────────────────────────────
    print("Step 4/7: 重建 subtask_index 映射（按 stage 順序）...")
    subtask_map = _build_subtask_map(ann["episodes"])
    for label, idx in subtask_map.items():
        print(f"  [{idx}] {label}")

    subtasks_df = pd.DataFrame(
        [{"subtask_index": idx} for _, idx in sorted(subtask_map.items(), key=lambda x: x[1])],
        index=pd.Index(
            [label for label, _ in sorted(subtask_map.items(), key=lambda x: x[1])],
            name="subtask",
        ),
    )
    subtasks_pq_path = out_dir / "meta" / "subtasks.parquet"
    pq.write_table(pa.Table.from_pandas(subtasks_df), subtasks_pq_path)
    print("  ✓ subtasks.parquet 已更新")

    # ── 5. 更新 data parquet ─────────────────────────────────────────────────
    print("Step 5/7: 更新 data parquet（過濾 episodes + 修正 subtask_index）...")
    data_files = sorted((out_dir / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError("找不到 data/*.parquet")

    total_neg = 0
    total_rows = 0
    for data_file in data_files:
        table = pq.read_table(data_file)
        df = table.to_pandas()

        # 過濾
        df = df[~df["episode_index"].isin(EXCLUDED_EPISODES)].copy()
        df["subtask_index"] = -1

        # 重新指派 subtask_index
        for ep_idx, grp in df.groupby("episode_index"):
            ep_str = str(int(ep_idx))
            if ep_str not in ann["episodes"]:
                continue
            subtasks = ann["episodes"][ep_str].get("subtasks", [])
            timestamps = grp.sort_values("timestamp")["timestamp"].tolist()
            new_idx = _assign_subtask_index(timestamps, subtasks, subtask_map)
            df.loc[grp.sort_values("timestamp").index, "subtask_index"] = new_idx

        neg = int((df["subtask_index"] < 0).sum())
        total_neg += neg
        total_rows += len(df)

        new_table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(new_table, data_file)

    print(f"  ✓ {total_rows:,} rows，subtask_index=-1: {total_neg} 個")

    # ── 6. 更新 meta/episodes parquet ───────────────────────────────────────
    print("Step 6/7: 更新 meta/episodes parquet 與 info.json...")
    ep_files = sorted((out_dir / "meta" / "episodes").rglob("*.parquet"))
    total_ep_rows = 0
    total_frames = 0
    for ep_file in ep_files:
        ep_table = pq.read_table(ep_file)
        ep_df = ep_table.to_pandas()
        ep_df = ep_df[~ep_df["episode_index"].isin(EXCLUDED_EPISODES)].copy()
        total_ep_rows += len(ep_df)
        if "length" in ep_df.columns:
            total_frames += int(ep_df["length"].sum())
        pq.write_table(pa.Table.from_pandas(ep_df, preserve_index=False), ep_file)
    print(f"  ✓ episodes: {total_ep_rows} 個，total_frames: {total_frames:,}")

    info_path = out_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["total_episodes"] = total_ep_rows
    if total_frames > 0:
        info["total_frames"] = total_frames
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # 更新 temporal_proportions
    tp = _recompute_temporal_proportions(ann["episodes"], subtask_map)
    tp_path = out_dir / "meta" / "temporal_proportions_dense.json"
    with open(tp_path, "w") as f:
        json.dump(tp, f, indent=2)
    print("  ✓ temporal_proportions_dense.json 已更新")

    # ── 7. Push to Hub ────────────────────────────────────────────────────────
    print(f"Step 7/7: 建立 {TARGET_REPO} 並 push meta + data...")
    api.create_repo(repo_id=TARGET_REPO, repo_type="dataset", exist_ok=True, private=False)

    # Push meta + data（不含 videos，稍後另外處理）
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=TARGET_REPO,
        repo_type="dataset",
        ignore_patterns=["videos/**"],
        commit_message=(
            "Re-export: fix subtask_index ordering, "
            f"exclude {sorted(EXCLUDED_EPISODES)}"
        ),
    )
    print("  ✓ meta + data 已 push")

    # 複製 videos（從 src repo 下載後上傳；videos 本身不需要修改）
    print("  複製 videos（下載 → 上傳，可能需要數分鐘）...")
    all_src_files = list(list_repo_files(SRC_REPO, repo_type="dataset", token=HF_TOKEN))
    video_files = [f for f in all_src_files if f.startswith("videos/")]
    print(f"  共 {len(video_files)} 個 video 檔案")

    for i, vf in enumerate(video_files, 1):
        local_path = hf_hub_download(SRC_REPO, vf, repo_type="dataset",
                                     local_dir=src_dir, token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=vf,
            repo_id=TARGET_REPO,
            repo_type="dataset",
        )
        print(f"  [{i}/{len(video_files)}] {vf}")

    print()
    print("=" * 60)
    print(f"✓ 完成！Dataset: https://huggingface.co/datasets/{TARGET_REPO}")
    print()
    print("下一步：執行驗證")
    print(f"  python3 test_annotation_quality.py --repo-id {TARGET_REPO}")
    print("=" * 60)


if __name__ == "__main__":
    main()
