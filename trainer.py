#!/usr/bin/env python3
import os, sys, time, json, traceback, uuid, importlib.util, multiprocessing
from datetime import datetime

ARTIFACT_DIR = os.path.expanduser("~/child_env_runner/artifacts")
TASK_DIR = os.path.expanduser("~/child_env_runner/tasks")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TRAINER_RESULTS = os.path.join(ARTIFACT_DIR, "trainer_results.jsonl")
TASK_RESULTS = os.path.join(ARTIFACT_DIR, "task_results.jsonl")

# -------------------------------
# Candidate generator
# -------------------------------
def generate_candidate(task):
    tid = task["id"]

    # GPU task handling
    if task.get("type") == "gpu":
        return """import torch
def gpu_matrix_mul(A, B):
    A_gpu = torch.tensor(A, dtype=torch.float32).cuda()
    B_gpu = torch.tensor(B, dtype=torch.float32).cuda()
    result = torch.matmul(A_gpu, B_gpu)
    return result.cpu().tolist()"""

    # Starter pack examples
    if tid == "task1":
        return """def mean(lst): return sum(lst) / len(lst)"""
    elif tid == "task2":
        return """import pandas as pd
def load_and_sum(path):
    df = pd.read_csv(path)
    return int(df['value'].sum())"""
    elif tid == "task3":
        return """import matplotlib.pyplot as plt
def plot_series(data, outpath):
    plt.figure(); plt.plot(data); plt.savefig(outpath); return outpath"""
    elif tid == "task4":
        return """from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
def train_and_score(X, y):
    model = LinearRegression().fit(X, y)
    return float(r2_score(y, model.predict(X)))"""
    elif tid == "task5":
        return """def count_lines(path):
    with open(path) as f: return sum(1 for _ in f)"""

    # Fallback for tasks 6–215 (cycle categories)
    num = int(tid.replace("task", ""))
    mod = num % 10
    if mod == 0:
        return """def bubble_sort(lst):
    arr = lst[:]
    for i in range(len(arr)):
        for j in range(len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr"""
    elif mod == 1:
        return """def factorial(n): return 1 if n<=1 else n*factorial(n-1)"""
    elif mod == 2:
        return """import pandas as pd
def count_rows(path): return len(pd.read_csv(path))"""
    elif mod == 3:
        return """import matplotlib.pyplot as plt
def plot_hist(data, outpath):
    plt.hist(data); plt.savefig(outpath); return outpath"""
    elif mod == 4:
        return """def unique_word_count(path):
    with open(path) as f: return len(set(f.read().split()))"""
    elif mod == 5:
        return """import numpy as np
def zscore_normalize(arr):
    arr = np.array(arr, dtype=float)
    return (arr - arr.mean())/arr.std()"""
    elif mod == 6:
        return """from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def train_logreg(X,y):
    model = LogisticRegression(max_iter=200).fit(X,y)
    return float(accuracy_score(y, model.predict(X)))"""
    elif mod == 7:
        return """import json
def merge_json(path1,path2,outpath):
    with open(path1) as f1, open(path2) as f2:
        d1,d2=json.load(f1),json.load(f2)
    with open(outpath,'w') as f: json.dump({**d1,**d2},f)"""
    elif mod == 8:
        return """import requests
def fetch_json(url): return requests.get(url).json()"""
    elif mod == 9:
        return """def fib(n):
    a,b=0,1
    for _ in range(n): a,b=b,a+b
    return a"""

# -------------------------------
# Task selection (round robin)
# -------------------------------
_task_index = 0
def load_tasks():
    tasks = []
    for fname in sorted(os.listdir(TASK_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(TASK_DIR, fname)) as f:
                tasks.append(json.load(f))
    return tasks

def pick_next_task():
    global _task_index
    tasks = load_tasks()
    if not tasks:
        raise RuntimeError("No tasks found")
    task = tasks[_task_index % len(tasks)]
    _task_index += 1
    return task

# -------------------------------
# Sandbox execution with timeout + cleanup
# -------------------------------
def _execute_candidate(candidate_path, func_name, inputs, queue):
    try:
        spec = importlib.util.spec_from_file_location("candidate", candidate_path)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)

        if not hasattr(candidate, func_name):
            queue.put(("error", f"Function '{func_name}' not found"))
            return

        func = getattr(candidate, func_name)
        result = func(**inputs) if isinstance(inputs, dict) else func(inputs)
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))

def run_in_sandbox(task, code, timeout=5):
    run_id = uuid.uuid4().hex[:8]
    candidate_path = os.path.join(ARTIFACT_DIR, f"candidate_{run_id}.py")
    with open(candidate_path, "w") as f:
        f.write(code)

    result = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().strftime("%Y%m%d-%H%M%S"),
        "task_id": task["id"],
        "description": task.get("description", ""),
        "success": False,
        "stdout": "",
        "stderr": "",
        "returncode": 0,
        "timeout": False,
        "score": 0,
        "error": "",
    }

    # Determine function name
    func_name = task.get("function")
    if not func_name:
        words = task.get("description","").split()
        if "function" in words:
            idx = words.index("function")
            if idx+1 < len(words):
                func_name = words[idx+1].split("(")[0]

    # Run in subprocess with timeout
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_execute_candidate,
        args=(candidate_path, func_name, task.get("inputs", {}), queue)
    )
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        result["timeout"] = True
        result["error"] = "Execution timed out"
    else:
        try:
            status, payload = queue.get_nowait()
            if status == "ok":
                expected = task.get("expected_outputs")
                if expected is None or payload == expected:
                    result["success"] = True
                    result["score"] = 1
                else:
                    result["error"] = f"Output mismatch: got {payload}, expected {expected}"
            else:
                result["error"] = payload
        except Exception as e:
            result["error"] = f"No result returned: {e}"

    # Cleanup candidate file
    try:
        os.remove(candidate_path)
    except OSError:
        pass

    # Log results
    with open(TRAINER_RESULTS, "a") as f:
        f.write(json.dumps(result) + "\n")

    return result

# -------------------------------
# Logging helpers
# -------------------------------
def log_result(result):
    summary = {
        "task_id": result["task_id"],
        "score": result["score"],
        "timestamp": result["timestamp"],
        "success": result["success"],
        "error": result["error"][:200]  # truncate for readability
    }
    with open(TASK_RESULTS, "a") as f:
        f.write(json.dumps(summary) + "\n")

# -------------------------------
# One training cycle
# -------------------------------
def run_training_cycle():
    task = pick_next_task()
    code = generate_candidate(task)
    result = run_in_sandbox(task, code)
    log_result(result)
    print(f"[{datetime.utcnow()}] Completed {task['id']} → success={result['success']}, score={result['score']}")
    sys.stdout.flush()

# -------------------------------
# Main loop
# -------------------------------
if __name__ == "__main__":
    while True:
        try:
            run_training_cycle()
        except Exception as e:
            print(f"Trainer loop error: {e}", file=sys.stderr)
        time.sleep(5)

