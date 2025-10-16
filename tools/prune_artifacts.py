#!/usr/bin/env python3
import os, sys, time, json, gzip, shutil
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(os.path.expanduser("~/child_env_runner"))
ART = BASE / "artifacts"
LOG_TRAINER = ART / "trainer_results.jsonl"
LOG_TASK = ART / "task_results.jsonl"

# Tunables via environment
TTL_DAYS = int(os.environ.get("PRUNE_TTL_DAYS", "14"))          # delete artifacts older than N days
MAX_LOG_MB = int(os.environ.get("PRUNE_MAX_LOG_MB", "256"))     # rotate when log exceeds MB
KEEP_PER_TASK = int(os.environ.get("PRUNE_KEEP_PER_TASK", "50"))# keep N most recent candidate_*.py per task (approx)

def human_ts():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def rotate_log(path: Path, max_mb: int):
    if not path.exists():
        return
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb <= max_mb:
        return
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    rot = path.with_name(f"{path.stem}-{ts}.jsonl")
    # Move and compress rotated file
    shutil.move(str(path), str(rot))
    with open(rot, "rb") as f_in, gzip.open(f"{rot}.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(rot)
    # Create fresh log
    path.touch()
    print(f"[{human_ts()}] Rotated {path.name} → {path.name}-{ts}.jsonl.gz")

def delete_old_artifacts(ttl_days: int):
    cutoff = datetime.utcnow() - timedelta(days=ttl_days)
    deleted = 0
    for p in ART.glob("candidate_*.py"):
        mtime = datetime.utcfromtimestamp(p.stat().st_mtime)
        if mtime < cutoff:
            p.unlink(missing_ok=True)
            deleted += 1
    print(f"[{human_ts()}] Deleted {deleted} old candidate files (> {ttl_days} days)")

def cap_per_task(limit: int):
    # Approximate per-task grouping by file mtime (no task id embedded)
    # Keep the newest N overall to bound growth; optional finer-grain later.
    files = sorted(ART.glob("candidate_*.py"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(files) <= limit:
        print(f"[{human_ts()}] candidate files within limit ({len(files)}/{limit})")
        return
    to_delete = files[limit:]
    for p in to_delete:
        p.unlink(missing_ok=True)
    print(f"[{human_ts()}] Capped candidate files to {limit}, deleted {len(to_delete)}")

def main():
    ART.mkdir(parents=True, exist_ok=True)
    rotate_log(LOG_TRAINER, MAX_LOG_MB)
    rotate_log(LOG_TASK, MAX_LOG_MB)
    delete_old_artifacts(TTL_DAYS)
    cap_per_task(KEEP_PER_TASK)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[{human_ts()}] Pruner error: {e}", file=sys.stderr)
        sys.exit(1)

