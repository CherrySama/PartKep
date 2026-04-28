from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-9B",
    local_dir="models/Qwen3.5-9B",
    local_dir_use_symlinks=False,
    token=True
)
print("\n✅ 下载完成：models/Qwen3.5-9B")