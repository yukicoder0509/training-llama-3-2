"""Push a model saved by train.py to the Hugging Face Hub as one commit.

Runs outside the training job so the upload does not count against the 30-min Slurm limit.
It is network-bound only (no GPU, little CPU), so running it on the login node is fine:

    python push_model.py --model_dir ~/gpt2_models/<run_name> --repo_id <MODEL_NAME>
"""

import argparse
import os

from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True, help="Folder written by train.py's trainer.save_model()")
parser.add_argument("--repo_id", type=str, default=os.environ.get("MODEL_NAME"), help="Hub repo (default: $MODEL_NAME)")
parser.add_argument("--message", type=str, default=None, help="Commit message (default: 'Upload <model_dir name>')")
args = parser.parse_args()

model_dir = os.path.expanduser(args.model_dir)
assert args.repo_id, "pass --repo_id or set MODEL_NAME"
assert os.path.isfile(os.path.join(model_dir, "model.safetensors")), f"no model.safetensors in {model_dir}"

api = HfApi()
api.create_repo(args.repo_id, exist_ok=True)
commit = api.upload_folder(
    repo_id=args.repo_id,
    folder_path=model_dir,
    allow_patterns=["config.json", "generation_config.json", "model.safetensors"],  # what push_to_hub used to upload
    commit_message=args.message or f"Upload {os.path.basename(os.path.normpath(model_dir))}",
)
print(f"pushed {model_dir} -> {commit.commit_url}")
