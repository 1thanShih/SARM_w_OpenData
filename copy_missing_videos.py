"""
把 charging_bimanual_merged 缺少的 left/right camera videos
複製到 SARM-opendata-annotated-fixed。
"""

import os
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

SRC_REPO = "Lebruhbruh/charging_bimanual_merged"
DST_REPO = "Lebruhbruh/SARM-opendata-annotated-fixed"
HF_TOKEN = os.environ["HF_TOKEN"]

api = HfApi(token=HF_TOKEN)

src_videos = sorted(f for f in list_repo_files(SRC_REPO, repo_type="dataset") if f.startswith("videos/"))
dst_videos = set(f for f in list_repo_files(DST_REPO, repo_type="dataset") if f.startswith("videos/"))

missing = [f for f in src_videos if f not in dst_videos]
print(f"需要複製 {len(missing)} 個 video 檔\n")

for i, vf in enumerate(missing, 1):
    print(f"[{i}/{len(missing)}] {vf}")
    local = hf_hub_download(SRC_REPO, vf, repo_type="dataset", token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=vf,
        repo_id=DST_REPO,
        repo_type="dataset",
        commit_message=f"Add missing camera video: {vf}",
    )
    print(f"  ✓ uploaded")

print(f"\n✓ 全部完成！")
