import json
import time
import subprocess
from pathlib import Path
from artifact_committer import commit_success

def log(msg):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[trainer] {timestamp} - {msg}")

# ─────────────────────────────────────────────
# Curriculum Hook Scanner (GGUF merges)
# ─────────────────────────────────────────────
def scan_curriculum_hooks():
    hooks = []
    for path in Path(".").glob("*.curriculum.json"):
        try:
            with open(path) as f:
                hook = json.load(f)
            if hook.get("event") == "merge_complete" and hook["score"].get("merge_success"):
                hooks.append({
                    "model": hook["output_model"],
                    "meta": hook["meta"],
                    "score": hook["score"],
                    "hook_path": str(path)
                })
        except Exception as e:
            log(f"Failed to parse {path}: {e}")
    return hooks

def promote_model(hook):
    model_path = hook["model"]
    score = hook["score"]
    log(f"Promoting model: {model_path}")
    log(f"Score: {score}")

    if Path(model_path).exists():
        cmd = ["llama.cpp/build/bin/llama-main", "--model", model_path, "--prompt", "Hello"]
        log(f"Launching inference: {' '.join(cmd)}")
        try:
            subprocess.run(cmd)
        except Exception as e:
            log(f"Inference failed: {e}")
    else:
        log(f"Model file not found: {model_path}")

# ─────────────────────────────────────────────
# Merge Scoring Loop (tensor-level)
# ─────────────────────────────────────────────
def score_merge(path):
    try:
        with open(path) as f:
            data = json.load(f)
        score = 0
        delta = data["mean_delta"]
        max_delta = data["max_delta"]
        shape = data["shape"]

        if delta > 0.001: score += 1
        if max_delta > 0.01: score += 1
        if shape[0] >= 5120: score += 1
        return score, data
    except Exception as e:
        log(f"Failed to score {path}: {e}")
        return None, None

def promote_tensor_merge(path, score, data):
    log(f"Promoting tensor merge: {data['tag']} with score {score}")
    promoted_path = f"{data['tag']}.promoted.json"
    with open(promoted_path, "w") as f:
        json.dump(data, f, indent=2)
    log(f"Saved promotion to {promoted_path}")

    # Auto-generate a simple prompt file
    prompt_text = (
        f"Merge tensor '{data['tag']}' with scale {data.get('scale', 1.0)}. "
        f"Base: {data.get('base', 'unknown')}, Adapter: {data.get('lora', 'unknown')}."
    )
    prompt_path = Path(f"{data['tag']}.merge_prompt.txt")
    prompt_path.write_text(prompt_text)

    # Commit to GitHub
    commit_success(
        task_name=data['tag'],
        candidate_path=Path(promoted_path),
        prompt_path=prompt_path
    )

def replay_existing_artifacts():
    """
    Walk through existing artifacts in ~/child-ai-curriculum/merges,
    and optionally re-score or re-commit them.
    """
    repo_dir = Path.home() / "child-ai-curriculum" / "merges"
    if not repo_dir.exists():
        print("[replay] No merges directory found.")
        return

    for date_dir in repo_dir.iterdir():
        if not date_dir.is_dir():
            continue
        for task_dir in date_dir.iterdir():
            if not task_dir.is_dir() or task_dir.name == "failures":
                continue

            candidate_files = list(task_dir.glob("*_candidate.json"))
            prompt_files = list(task_dir.glob("*_prompt.txt"))

            if candidate_files and prompt_files:
                candidate_path = candidate_files[0]
                prompt_path = prompt_files[0]
                task_name = task_dir.name

                print(f"[replay] Found {task_name}, replaying commit...")

                # Optionally re‑score here (stubbed out)
                with open(candidate_path) as f:
                    data = json.load(f)
                    score = data.get("score", None)
                    print(f"[replay] Score = {score}")

                # Re‑commit to GitHub (idempotent if already committed)
                commit_success(task_name, candidate_path, prompt_path)



# ─────────────────────────────────────────────
# Main Trainer Loop
# ─────────────────────────────────────────────
def main():
    # Promote curriculum-based GGUF merges
    hooks = scan_curriculum_hooks()
    for hook in hooks:
        promote_model(hook)

    # Score and promote tensor-level merges
    for path in Path(".").glob("*.merge.json"):
        score, data = score_merge(path)
        if score is not None:
            log(f"{path.name} scored {score}")
            if score >= 2:
                promote_tensor_merge(path, score, data)

if __name__ == "__main__":
    # Normal training loop
    run_trainer_loop()

    # Replay existing artifacts for bootstrap
    replay_existing_artifacts()

