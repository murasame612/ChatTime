from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="ChengsenWang/ChatTime-1-7B-Chat",
    local_dir="./ChatTime-1-7B-Chat",
    local_dir_use_symlinks=False
)
