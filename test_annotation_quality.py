"""
SARM Annotation Quality Tests
==============================
驗證 HGLLL/SARM-opendata-annotated-1 的標注資料是否符合 SARM Reward Model 訓練要求。

執行方式：
    python test_annotation_quality.py
    python test_annotation_quality.py --repo-id YOUR/REPO  # 改測其他 repo

輸出：每個 check 的 PASS/FAIL + 最後統計。非零 exit code 代表有 FAIL。
"""

import argparse
import json
import sys
from pathlib import Path

# ── 設定 ─────────────────────────────────────────────────────────────────────

DEFAULT_REPO_ID = "HGLLL/SARM-opendata-annotated-1"

EXPECTED_STAGES = [
    "Pick up the phone",
    "Flip the phone sideways",
    "Pick up the charging cable and plug it into the phone",
    "Turn on the power of the extension cord",
]
NUM_EXPECTED_STAGES = len(EXPECTED_STAGES)

# 已知影片中缺少特定 stage 的 episodes（使用者手動確認）
KNOWN_MISSING_STAGE_EPISODES = {
    6:   "no stage 4 (Turn on power)",
    8:   "no stage 2 (Flip phone)",
    55:  "no stage 2 (Flip phone)",
    79:  "no stage 2 (Flip phone)",
    111: "no stage 2 (Flip phone)",
    113: "no stage 2 (Flip phone)",
    134: "no stage 2 (Flip phone)",
    266: "no stage 2 (Flip phone)",
}

# 最短合理 stage 時長（秒）：低於此視為 suspicious
MIN_STAGE_DURATION_SEC = 1.0

# 最大允許的 segment gap（秒）：相鄰 subtask 之間不應有超過此值的空隙
MAX_GAP_SEC = 0.1

# ── helpers ──────────────────────────────────────────────────────────────────

_passed = 0
_failed = 0
_warnings = 0


def _result(ok: bool, name: str, msg: str = ""):
    global _passed, _failed
    icon = "PASS" if ok else "FAIL"
    print(f"  [{icon}] {name}" + (f": {msg}" if msg else ""))
    if ok:
        _passed += 1
    else:
        _failed += 1
    return ok


def _warn(name: str, msg: str = ""):
    global _warnings
    print(f"  [WARN] {name}" + (f": {msg}" if msg else ""))
    _warnings += 1


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── download helpers ──────────────────────────────────────────────────────────

def _download_json(repo_id: str, filename: str) -> dict:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id, filename, repo_type="dataset")
    with open(path) as f:
        return json.load(f)


def _download_parquet(repo_id: str, filename: str):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download(repo_id, filename, repo_type="dataset")
    return pd.read_parquet(path)


def _download_parquet_pyarrow(repo_id: str, filename: str):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    path = hf_hub_download(repo_id, filename, repo_type="dataset")
    return pq.read_table(path)


# ── Test Suite ────────────────────────────────────────────────────────────────

def test_meta_files_exist(repo_id: str):
    section("1. 必要 meta 檔案存在性")
    from huggingface_hub import list_repo_files
    files = set(list_repo_files(repo_id, repo_type="dataset"))

    required = [
        "meta/info.json",
        "meta/lerobot_annotations.json",
        "meta/subtasks.parquet",
        "meta/temporal_proportions_dense.json",
    ]
    for f in required:
        _result(f in files, f"檔案存在: {f}")

    # 確認有 data parquet
    data_files = [f for f in files if f.startswith("data/") and f.endswith(".parquet")]
    _result(len(data_files) > 0, "data/*.parquet 存在", f"找到 {len(data_files)} 個")

    return files


def test_info_json(repo_id: str) -> dict:
    section("2. meta/info.json 驗證")
    info = _download_json(repo_id, "meta/info.json")

    _result(info.get("codebase_version") == "v3.0", "codebase_version = v3.0",
            f"實際: {info.get('codebase_version')}")
    _result("fps" in info and info["fps"] > 0, "fps 存在且 > 0", f"fps={info.get('fps')}")

    features = info.get("features", {})
    _result("observation.state" in features, "features 含 observation.state")
    _result("action" in features, "features 含 action")

    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    _result(len(video_keys) > 0, "features 含至少一個 video key",
            f"找到: {video_keys}")

    # subtask_index 是否已被加入 features（export 後才會有）
    has_subtask_col = "subtask_index" in features
    if has_subtask_col:
        _result(True, "features 含 subtask_index（export 完成）")
    else:
        _warn("subtask_index 未在 info.json features 中",
              "可能尚未 export；data parquet 裡的欄位仍需另行確認")

    return info


def test_subtask_index_ordering(repo_id: str):
    """
    SARM 核心要求：subtask_index 必須按 stage 執行順序排列（0→1→2→3），
    而非按 label 字母順序。subtasks.parquet 的 index 決定 reward 計算方向。
    """
    section("3a. subtask_index 排列順序驗證（SARM 核心需求）")
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id, "meta/subtasks.parquet", repo_type="dataset")
        table = pq.read_table(path)
        df = table.to_pandas()
    except ImportError:
        _warn("pyarrow 未安裝，跳過 subtask_index 排列順序驗證")
        return

    # 取得 index 欄位（可能叫 subtask_index 或是 index）
    idx_col = "subtask_index" if "subtask_index" in df.columns else None
    if idx_col is None:
        _result(False, "subtasks.parquet 有 subtask_index 欄位")
        return

    # 確認每個 expected stage 的 subtask_index 按順序遞增
    label_to_idx = {}
    for label, row in df.iterrows():
        label_to_idx[label] = int(row[idx_col])

    print("  當前 label → subtask_index 映射:")
    for stage in EXPECTED_STAGES:
        idx = label_to_idx.get(stage, "NOT FOUND")
        print(f"    [{idx}] {stage}")

    # 正確的映射應是 stage 0→1→2→3（按照任務執行順序）
    indices_in_order = [label_to_idx.get(stage) for stage in EXPECTED_STAGES]
    is_sequential = (
        all(i is not None for i in indices_in_order) and
        indices_in_order == sorted(indices_in_order) and
        indices_in_order == list(range(len(EXPECTED_STAGES)))
    )

    _result(
        is_sequential,
        "subtask_index 為 0,1,2,3（按 stage 執行順序）",
        f"實際: {indices_in_order}（應為 {list(range(len(EXPECTED_STAGES)))}）"
        if not is_sequential else "",
    )

    if not is_sequential:
        print()
        print("  ⚠️  根本原因：subtasks.parquet 的 index 按 label 字母排序而非 stage 順序。")
        print("     SARM 用 subtask_index 計算 reward signal，若序列為 6→4→5→7，")
        print("     training 無法學到正確的 stage 進展方向。")
        print("  修正方式：re-export 時確保 subtask_index = stage 順序（0,1,2,3）")

    # 同時列出是否有非預期 label（ep40 的 '1','2','3','4'）
    extra_labels = [l for l in label_to_idx if l not in EXPECTED_STAGES]
    if extra_labels:
        _result(False, "subtasks.parquet 無非預期 label",
                f"多餘 label: {extra_labels}（來自 ep40 的錯誤標注）")
    else:
        _result(True, "subtasks.parquet 無非預期 label")


def test_subtasks_parquet(repo_id: str) -> dict:
    section("3. meta/subtasks.parquet 驗證")
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id, "meta/subtasks.parquet", repo_type="dataset")
        table = pq.read_table(path)
        df = table.to_pandas()
    except ImportError:
        _warn("pyarrow 未安裝，跳過 subtasks.parquet 驗證")
        return {}

    _result(len(df) > 0, "subtasks.parquet 有資料", f"{len(df)} rows")

    # subtask_index 可能是欄位或是 DataFrame index
    has_subtask_index = "subtask_index" in df.columns or df.index.name == "subtask_index"
    _result(has_subtask_index, "subtask_index 欄位存在（column 或 index）")
    has_subtask_label = "subtask" in df.columns or df.index.name == "subtask"
    _result(has_subtask_label, "subtask label 欄位存在（column 或 index）")

    if "subtask" in df.columns:
        actual_labels = set(df["subtask"].tolist())
        expected_set = set(EXPECTED_STAGES)
        missing_from_parquet = expected_set - actual_labels
        extra_in_parquet = actual_labels - expected_set
        _result(len(missing_from_parquet) == 0,
                "subtasks.parquet 包含所有 4 個 stage 名稱",
                f"缺少: {missing_from_parquet}" if missing_from_parquet else "")
        if extra_in_parquet:
            _warn("subtasks.parquet 有額外/異常 label", str(extra_in_parquet))

    return df.to_dict() if hasattr(df, "to_dict") else {}


def test_annotations(repo_id: str) -> dict:
    section("4. lerobot_annotations.json 深度驗證")
    ann = _download_json(repo_id, "meta/lerobot_annotations.json")

    _result(ann.get("version") == 1, "version == 1")
    episodes = ann.get("episodes", {})
    total = len(episodes)
    _result(total > 0, "episodes 不為空", f"{total} 個")

    # ── 4a: 每個 episode 都有 subtasks ──────────────────────
    no_subtask_eps = [int(k) for k, v in episodes.items()
                      if len(v.get("subtasks", [])) == 0]
    _result(len(no_subtask_eps) == 0, "沒有 episode 是 0 subtasks（完全沒標注）",
            f"問題 episodes: {no_subtask_eps}" if no_subtask_eps else "")

    # ── 4b: subtask 數量正確 ─────────────────────────────────
    wrong_count_eps = {int(k): len(v.get("subtasks", []))
                       for k, v in episodes.items()
                       if len(v.get("subtasks", [])) != NUM_EXPECTED_STAGES
                       and len(v.get("subtasks", [])) > 0}
    _result(len(wrong_count_eps) == 0,
            f"所有 episode 都有 {NUM_EXPECTED_STAGES} 個 subtasks",
            f"問題 episodes: {wrong_count_eps}" if wrong_count_eps else "")

    # ── 4c: label 名稱完全吻合 ─────────────────────────────
    label_mismatch_eps = {}
    for k, v in episodes.items():
        labels = [s["label"] for s in v.get("subtasks", [])]
        wrong = [l for l in labels if l not in EXPECTED_STAGES]
        if wrong:
            label_mismatch_eps[int(k)] = wrong
    _result(len(label_mismatch_eps) == 0,
            "所有 subtask label 完全吻合預期 stage 名稱",
            f"Label 不符的 episodes: {label_mismatch_eps}" if label_mismatch_eps else "")

    # ── 4d: subtask 順序吻合 EXPECTED_STAGES ────────────────
    wrong_order_eps = {}
    for k, v in episodes.items():
        labels = [s["label"] for s in v.get("subtasks", [])]
        if not labels:
            continue  # 空 episode 已由 4a 覆蓋
        if labels == EXPECTED_STAGES:
            continue
        if all(l in EXPECTED_STAGES for l in labels):
            wrong_order_eps[int(k)] = labels
    _result(len(wrong_order_eps) == 0,
            "所有 episode subtask 順序吻合 EXPECTED_STAGES",
            f"順序錯誤: {wrong_order_eps}" if wrong_order_eps else "")

    # ── 4e: segment 連續性（相鄰 end ≈ next start，負 gap 代表重疊）──────────
    gap_eps = {}
    overlap_eps = {}
    for k, v in episodes.items():
        subtasks = sorted(v.get("subtasks", []), key=lambda s: s["start"])
        gaps = []
        overlaps = []
        for i in range(len(subtasks) - 1):
            gap = subtasks[i + 1]["start"] - subtasks[i]["end"]
            if gap < -MAX_GAP_SEC:
                overlaps.append((i, round(gap, 4)))  # 負值 = 重疊
            elif gap > MAX_GAP_SEC:
                gaps.append((i, round(gap, 4)))
        if gaps:
            gap_eps[int(k)] = gaps
        if overlaps:
            overlap_eps[int(k)] = overlaps
    _result(len(gap_eps) == 0,
            f"所有 episode subtasks 無正向 gap（> {MAX_GAP_SEC}s）",
            f"有 gap 的 episodes (前 5): {dict(list(gap_eps.items())[:5])}" if gap_eps else "")
    _result(len(overlap_eps) == 0,
            "所有 episode subtasks 無重疊（負 gap）",
            f"重疊 episodes: {overlap_eps}" if overlap_eps else "")

    # ── 4f: 每個 segment 時長合理 ────────────────────────────
    suspicious_eps = {}
    for k, v in episodes.items():
        short = [(s["label"], round(s["end"] - s["start"], 3))
                 for s in v.get("subtasks", [])
                 if (s["end"] - s["start"]) < MIN_STAGE_DURATION_SEC]
        if short:
            suspicious_eps[int(k)] = short
    _result(len(suspicious_eps) == 0,
            f"所有 segment 時長 ≥ {MIN_STAGE_DURATION_SEC}s",
            f"過短 segment 的 episodes (前 5): {dict(list(suspicious_eps.items())[:5])}"
            if suspicious_eps else "")

    # ── 4g: 已知缺 stage 的 episodes 警告 ───────────────────
    section("4g. 已知缺失 stage 的 episodes（使用者手動確認）")
    print("  這些 episodes 在影片中缺少特定 stage，但 annotation 仍標注了對應時段。")
    print("  SARM 訓練時，這些 episodes 的對應 subtask_index 可能是噪音：")
    for ep_idx, note in KNOWN_MISSING_STAGE_EPISODES.items():
        ep_str = str(ep_idx)
        if ep_str in episodes:
            subtasks = episodes[ep_str]["subtasks"]
            labels = [s["label"] for s in subtasks]
            durations = {s["label"]: round(s["end"] - s["start"], 2) for s in subtasks}
            _warn(f"ep {ep_idx:3d} ({note})",
                  f"labels={labels}, durations={durations}")
        else:
            _warn(f"ep {ep_idx:3d} ({note})", "不在 annotations 中")

    return episodes


def test_stage_duration_distribution(episodes: dict):
    section("5. Stage 時長分佈分析")
    if not episodes:
        _warn("annotations 為空，跳過分佈分析")
        return

    from collections import defaultdict
    stage_durations = defaultdict(list)

    for v in episodes.values():
        for s in v.get("subtasks", []):
            if s["label"] in EXPECTED_STAGES:
                stage_durations[s["label"]].append(s["end"] - s["start"])

    for stage in EXPECTED_STAGES:
        durs = stage_durations[stage]
        if not durs:
            _result(False, f"Stage '{stage[:30]}...' 有資料", "完全沒有")
            continue
        mean_d = sum(durs) / len(durs)
        min_d = min(durs)
        max_d = max(durs)
        short_count = sum(1 for d in durs if d < MIN_STAGE_DURATION_SEC)
        print(f"  {stage[:50]}")
        print(f"    n={len(durs)}, mean={mean_d:.1f}s, min={min_d:.2f}s, max={max_d:.1f}s, "
              f"短於 {MIN_STAGE_DURATION_SEC}s: {short_count} 個")


def test_data_parquet(repo_id: str, files: set):
    section("6. data parquet 欄位驗證（抽查第一個 chunk）")
    data_files = sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))
    if not data_files:
        _result(False, "data parquet 存在", "找不到任何 data parquet")
        return

    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id, data_files[0], repo_type="dataset")
        table = pq.read_table(path)
        df = table.to_pandas()
    except ImportError:
        _warn("pyarrow 未安裝，跳過 data parquet 驗證")
        return
    except Exception as e:
        _result(False, f"讀取 {data_files[0]}", str(e))
        return

    print(f"  抽查: {data_files[0]}  ({len(df)} rows, columns: {list(df.columns)})")

    required_cols = ["episode_index", "timestamp", "action", "observation.state"]
    for col in required_cols:
        _result(col in df.columns, f"欄位 {col} 存在")

    # subtask_index 欄位
    if "subtask_index" in df.columns:
        neg = int((df["subtask_index"] < 0).sum())
        total = len(df)
        _result(neg == 0, "無 subtask_index = -1 的 frame",
                f"{neg}/{total} frames 沒有 subtask_index" if neg > 0 else "")

        # 每個 episode 的 subtask_index 應單調非遞減
        bad_mono_eps = []
        for ep_idx, grp in df.groupby("episode_index"):
            indices = grp.sort_values("timestamp")["subtask_index"].tolist()
            if any(indices[i] > indices[i + 1] for i in range(len(indices) - 1)):
                bad_mono_eps.append(int(ep_idx))
        _result(len(bad_mono_eps) == 0,
                "所有 episode subtask_index 單調非遞減",
                f"有回退的 episodes: {bad_mono_eps}" if bad_mono_eps else "")
    else:
        _warn("data parquet 無 subtask_index 欄位",
              "需先執行 lerobot-annotate 的 export 流程")

    # observation.state shape
    if "observation.state" in df.columns:
        sample = df["observation.state"].iloc[0]
        if hasattr(sample, "__len__"):
            _result(len(sample) == 14, "observation.state 維度 = 14",
                    f"實際: {len(sample)}")


def test_camera_coverage(repo_id: str, files: set):
    section("7a. 三 Camera Video 完整性驗證")
    from collections import Counter
    video_files = [f for f in files if f.startswith("videos/")]
    cams = Counter(f.split("/")[1] for f in video_files)

    EXPECTED_CAMERAS = [
        "observation.images.top",
        "observation.images.left",
        "observation.images.right",
    ]
    EXPECTED_COUNTS = {
        "observation.images.top": 18,
        "observation.images.left": 10,
        "observation.images.right": 12,
    }

    for cam in EXPECTED_CAMERAS:
        count = cams.get(cam, 0)
        expected = EXPECTED_COUNTS[cam]
        _result(count == expected, f"{cam.split('.')[-1]} camera: {count}/{expected} files",
                f"缺少 {expected - count} 個" if count < expected else "")


def test_episode_meta(repo_id: str, files: set):
    section("7. meta/episodes parquet 驗證")
    ep_files = sorted(f for f in files
                      if f.startswith("meta/episodes/") and f.endswith(".parquet"))
    if not ep_files:
        _result(False, "meta/episodes/*.parquet 存在", "找不到")
        return

    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id, ep_files[0], repo_type="dataset")
        table = pq.read_table(path)
        df = table.to_pandas()
    except ImportError:
        _warn("pyarrow 未安裝，跳過 episodes parquet 驗證")
        return

    _result("episode_index" in df.columns, "欄位 episode_index 存在")
    _result("length" in df.columns, "欄位 length 存在")
    _result(len(df) > 0, "episodes parquet 有資料", f"{len(df)} episodes")

    if "length" in df.columns:
        zero_len = int((df["length"] == 0).sum())
        _result(zero_len == 0, "沒有 length=0 的 episode",
                f"{zero_len} 個 length=0" if zero_len > 0 else "")


def test_temporal_proportions(repo_id: str):
    section("8. temporal_proportions_dense.json 驗證")
    tp = _download_json(repo_id, "meta/temporal_proportions_dense.json")

    _result(len(tp) == NUM_EXPECTED_STAGES,
            f"temporal_proportions 有 {NUM_EXPECTED_STAGES} 個 stage",
            f"實際: {len(tp)}")

    total = sum(tp.values())
    _result(abs(total - 1.0) < 0.01, "比例總和 ≈ 1.0", f"總和={total:.4f}")

    for stage in EXPECTED_STAGES:
        if stage in tp:
            prop = tp[stage]
            print(f"  {prop:.3%}  {stage}")
            if prop < 0.03:
                _warn(f"Stage 比例過低: '{stage[:40]}'", f"{prop:.3%}")
        else:
            _result(False, f"stage '{stage[:40]}' 在 temporal_proportions 中")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SARM Annotation Quality Tests")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    args = parser.parse_args()

    repo_id = args.repo_id
    print(f"\n{'='*60}")
    print(f"  SARM Annotation Quality Tests")
    print(f"  Dataset: {repo_id}")
    print(f"{'='*60}")

    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        print("ERROR: huggingface_hub 未安裝。請執行: pip install huggingface_hub")
        sys.exit(1)

    files = test_meta_files_exist(repo_id)
    test_info_json(repo_id)
    test_subtask_index_ordering(repo_id)
    test_subtasks_parquet(repo_id)
    episodes = test_annotations(repo_id)
    test_stage_duration_distribution(episodes)
    test_data_parquet(repo_id, files)
    test_camera_coverage(repo_id, files)
    test_episode_meta(repo_id, files)
    test_temporal_proportions(repo_id)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Summary: {_passed} PASS, {_failed} FAIL, {_warnings} WARN")
    print(f"{'='*60}\n")

    if _failed > 0:
        print("❌ 有 FAIL 項目，此 dataset 不符合 SARM 訓練要求，請先修正。")
        sys.exit(1)
    elif _warnings > 0:
        print("⚠️  所有 check PASS，但有 WARN 項目，建議確認後再訓練。")
    else:
        print("✅ 全部 PASS，dataset 符合 SARM 訓練要求。")


if __name__ == "__main__":
    main()
