from huggingface_hub import snapshot_download

repo_id = "deepdml/faster-whisper-large-v3-turbo-ct2"
local_dir = "faster-whisper-large-v3-turbo-ct2"
snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="model")