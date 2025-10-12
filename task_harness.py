#!/usr/bin/env python3
import json, os, sys, importlib.util
from sandbox_runner import run_snippet

TASKS_DIR = os.path.expanduser("~/child_env_runner/tasks")
ARTIFACTS_DIR = os.path.expanduser("~/child_env_runner/artifacts")
os.makedirs(TASKS_DIR, exist_ok=True)

def load_task(task_id):
    path = os.path.join(TASKS_DIR, f"{task_id}.json")
    with open(path) as f:
        return json.load(f)

def score_task(task, code):
    # Run the snippet in the sandbox
    result = run_snippet(code)
    result["task_id"] = task["id"]
    result["description"] = task["description"]

    # If run failed, mark as fail
    if not result["success"]:
        result["score"] = 0
        return result

    # Dynamically import the function from the snippet
    # Expecting the snippet to define a function with the same name as task["function"]
    func_name = task.get("function", "solution")
    try:
        # Write code to temp file and import
        import tempfile, uuid
        fname = f"/tmp/{uuid.uuid4().hex}.py"
        with open(fname, "w") as f:
            f.write(code)
        spec = importlib.util.spec_from_file_location("candidate", fname)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        func = getattr(mod, func_name)
    except Exception as e:
        result["score"] = 0
        result["error"] = f"Import error: {e}"
        return result

    # Run tests
    passed = 0
    for test in task["tests"]:
        try:
            out = func(*test["input"]) if isinstance(test["input"], list) else func(test["input"])
            if out == test["expected"]:
                passed += 1
        except Exception as e:
            continue

    result["score"] = passed / len(task["tests"])
    return result

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python task_harness.py <task_id> <code_file>")
        sys.exit(1)

    task_id = sys.argv[1]
    code_file = sys.argv[2]

    with open(code_file) as f:
        code = f.read()

    task = load_task(task_id)
    result = score_task(task, code)

    log_path = os.path.join(ARTIFACTS_DIR, "task_results.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(json.dumps(result, indent=2))

