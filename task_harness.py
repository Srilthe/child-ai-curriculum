#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-

"""
task_harness.py

Loads a coding task, generates candidate solutions via model_driver,
evaluates them with pytest, logs results, and commits passing solutions
to the artifact repository with the original request embedded.
"""

import subprocess
import logging
from pathlib import Path
import sys
import tempfile
import shutil

from model_driver import generate_code
from artifact_committer import commit_success

# Configure logging
logging.basicConfig(
    filename="artifacts/logs/task_harness.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def run_pytest(candidate_path: Path, test_file: Path) -> bool:
    """Run pytest on the candidate against the given test file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_candidate = Path(tmpdir) / candidate_path.name
        shutil.copy(candidate_path, tmp_candidate)
        shutil.copy(test_file, Path(tmpdir) / test_file.name)

        result = subprocess.run(
            ["pytest", "-q", tmp_candidate.name],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )

        log_path = Path("artifacts/results") / f"{candidate_path.stem}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout + "\n" + result.stderr)

        return result.returncode == 0

def run_task(task_name: str, max_retries: int = 3):
    """Run a single task end-to-end."""
    task_dir = Path("tasks") / task_name
    prompt_path = task_dir / "prompt.txt"
    template_path = task_dir / "template.py"
    test_file = next(task_dir.glob("test_*.py"))

    prompt_text = prompt_path.read_text().strip()
    template_code = template_path.read_text()

    for attempt in range(1, max_retries + 1):
        logging.info("Task %s attempt %d", task_name, attempt)

        candidate_code = generate_code(prompt_text, template_code)
        candidate_dir = Path("artifacts/candidates")
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_dir / f"{task_name}_cand{attempt}.py"
        candidate_path.write_text(candidate_code)

        passed = run_pytest(candidate_path, test_file)

        if passed:
            logging.info("Task %s solved on attempt %d", task_name, attempt)
            commit_success(task_name, candidate_path, prompt_path)
            return True
        else:
            logging.info("Task %s failed on attempt %d", task_name, attempt)

    logging.warning("Task %s not solved after %d attempts", task_name, max_retries)
    return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python task_harness.py <task_name>")
        sys.exit(1)

    task_name = sys.argv[1]
    success = run_task(task_name)
    sys.exit(0 if success else 1)
