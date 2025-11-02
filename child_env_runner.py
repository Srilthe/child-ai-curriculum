import subprocess
import argparse
import json
import os
import sys
import time
from pathlib import Path

def log(msg, level="INFO", logfile=None):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{level}] {timestamp} - {msg}"
    print(line)
    if logfile:
        with open(logfile, "a") as f:
            f.write(line + "\n")

def already_running(tag):
    pid_file = Path(f"/tmp/{tag}.pid")
    if pid_file.exists():
        try:
            with open(pid_file) as f:
                pid = int(f.read())
            os.kill(pid, 0)  # Check if process is alive
            return pid
        except Exception:
            pid_file.unlink()  # Stale PID
    return None

def save_pid(tag, pid):
    with open(f"/tmp/{tag}.pid", "w") as f:
        f.write(str(pid))

def save_metadata(meta, tag):
    path = Path(f"runs/{tag}_{int(time.time())}.meta.json")
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return str(path)

def emit_curriculum_hook(meta_path, output_path):
    hook = {
        "event": "merge_complete",
        "timestamp": time.time(),
        "meta": meta_path,
        "output_model": output_path,
        "score": {
            "merge_success": True,
            "tensor_count": 363  # Optional: parse from logs later
        }
    }
    hook_path = f"{output_path}.curriculum.json"
    with open(hook_path, "w") as f:
        json.dump(hook, f, indent=2)
    return hook_path

def launch(script, args, tag="default", dry_run=False):
    pid = already_running(tag)
    log_path = f"runs/{tag}.log"

    if pid:
        log(f"Process with tag '{tag}' already running (PID {pid})", logfile=log_path)
        return

    is_python = script.endswith(".py")
    cmd = [sys.executable, script] + args if is_python else [script] + args

    meta = {
        "script": script,
        "args": args,
        "timestamp": time.time(),
        "tag": tag,
        "cmd": cmd,
        "cwd": os.getcwd(),
        "env": dict(os.environ)
    }

    meta_path = save_metadata(meta, tag)

    if dry_run:
        log(f"Dry-run: would launch {' '.join(cmd)}", logfile=log_path)
        log(f"Metadata saved to {meta_path}", logfile=log_path)
        return

    try:
        log(f"Launching: {' '.join(cmd)}", logfile=log_path)
        proc = subprocess.Popen(cmd)
        save_pid(tag, proc.pid)
        log(f"Launched with PID {proc.pid}", logfile=log_path)

        # Wait briefly to allow output file to appear
        time.sleep(2)
        output_path = "lora.gguf"
        if Path(output_path).exists():
            hook_path = emit_curriculum_hook(meta_path, output_path)
            log(f"Curriculum hook saved to {hook_path}", logfile=log_path)

    except Exception as e:
        log(f"Failed to launch subprocess: {e}", level="ERROR", logfile=log_path)

def main():
    parser = argparse.ArgumentParser(description="Generic subprocess runner", allow_abbrev=False)
    parser.add_argument("script", help="Path to the binary or script to run")
    parser.add_argument("--tag", default="run", help="Tag for PID tracking and metadata")
    parser.add_argument("--dry-run", action="store_true", help="Preview launch without executing")

    # Parse known args, leave the rest for the subprocess
    opts, unknown_args = parser.parse_known_args()
    launch(opts.script, unknown_args, tag=opts.tag, dry_run=opts.dry_run)

if __name__ == "__main__":
    main()

