import json
import time
from pathlib import Path

def log(msg):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[scoring_loop] {timestamp} - {msg}")

def score_merge(path):
    try:
        with open(path) as f:
            data = json.load(f)
        score = 0
        delta = data["mean_delta"]
        max_delta = data["max_delta"]
        shape = data["shape"]

        # Scoring logic
        if delta > 0.001: score += 1
        if max_delta > 0.01: score += 1
        if shape[0] >= 5120: score += 1  # Large tensor
        return score, data
    except Exception as e:
        log(f"Failed to score {path}: {e}")
        return None, None

def promote(path, score, data):
    log(f"Promoting {data['tag']} with score {score}")
    # Example: rename or copy for inference
    promoted_path = f"{data['tag']}.promoted.json"
    with open(promoted_path, "w") as f:
        json.dump(data, f, indent=2)
    log(f"Saved promotion to {promoted_path}")

def main():
    for path in Path(".").glob("*.merge.json"):
        score, data = score_merge(path)
        if score is not None:
            log(f"{path.name} scored {score}")
            if score >= 2:
                promote(path, score, data)

if __name__ == "__main__":
    main()

