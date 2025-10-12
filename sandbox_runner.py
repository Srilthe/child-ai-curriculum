#!/usr/bin/env python3
import subprocess, tempfile, time, json, os, sys, uuid

ARTIFACTS_DIR = os.path.expanduser("~/child_env_runner/artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def run_snippet(code: str, timeout: int = 5):
    run_id = str(uuid.uuid4())[:8]
    ts = time.strftime("%Y%m%d-%H%M%S")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        proc = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        result = {
            "run_id": run_id,
            "timestamp": ts,
            "success": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "timeout": False
        }
    except subprocess.TimeoutExpired as e:
        result = {
            "run_id": run_id,
            "timestamp": ts,
            "success": False,
            "stdout": e.stdout or "",
            "stderr": "TimeoutExpired",
            "returncode": None,
            "timeout": True
        }
    finally:
        os.remove(fname)
    log_path = os.path.join(ARTIFACTS_DIR, "sandbox_runs.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    return result

if __name__ == "__main__":
    code = "print('Hello from the child sandbox!')"
    res = run_snippet(code)
    print(json.dumps(res, indent=2))
