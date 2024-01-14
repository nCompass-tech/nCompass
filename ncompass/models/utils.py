from huggingface_hub import hf_hub_download

def download_from_hf(repo_id: str, filename: str) -> str :
    return hf_hub_download(repo_id, filename)
