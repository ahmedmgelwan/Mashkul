"""
رفع الموديل والتوكنايزر على Hugging Face Hub.

الاستخدام:
    hf auth login   # أول مرة بس
    python -m scripts.upload_to_hub --repo-id your-username/moshakeel
"""
import argparse

from huggingface_hub import create_repo, upload_file

from src.config import PathConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="ahmedmgelwan/Mashkul")
    args = parser.parse_args()

    create_repo(args.repo_id, exist_ok=True)

    for path, name_in_repo in [
        (PathConfig.model_path, "model.pt"),
        (PathConfig.tokenizer_path, "tokenizer.json"),
    ]:
        upload_file(path_or_fileobj=path, path_in_repo=name_in_repo, repo_id=args.repo_id)
        print(f"Uploaded {path} -> {args.repo_id}/{name_in_repo}")


if __name__ == "__main__":
    main()