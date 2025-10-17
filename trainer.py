#!/usr/bin/env python3
import os, sys, time, json, traceback, uuid, importlib.util, multiprocessing, random
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")

ARTIFACT_DIR = os.path.expanduser("~/child_env_runner/artifacts")
TASK_DIR = os.path.expanduser("~/child_env_runner/tasks")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TRAINER_RESULTS = os.path.join(ARTIFACT_DIR, "trainer_results.jsonl")
TASK_RESULTS = os.path.join(ARTIFACT_DIR, "task_results.jsonl")
SUCCESSFUL_PIPELINES = os.path.join(ARTIFACT_DIR, "successful_pipelines.jsonl")

# -------------------------------
# Building blocks library
# -------------------------------
BUILDING_BLOCKS = {
    "load_csv": """import pandas as pd
def load_csv(path):
    return pd.read_csv(path)
""",
    "save_csv": """def save_csv(df, path):
    df.to_csv(path, index=False)
    return path
""",
    "normalize": """import numpy as np
def normalize(df, column):
    arr = df[column].to_numpy(dtype=float)
    norm = (arr - arr.mean()) / arr.std()
    df[column + "_norm"] = norm
    return df
""",
    "filter_rows": """def filter_rows(df, column, threshold):
    return df[df[column] > threshold]
""",
    "aggregate_mean": """def aggregate_mean(df, column):
    return float(df[column].mean())
""",
    "plot_histogram": """import matplotlib.pyplot as plt
def plot_histogram(df, column, outpath):
    plt.figure()
    df[column].hist()
    plt.savefig(outpath)
    return outpath
""",
    "plot_series": """import matplotlib.pyplot as plt
def plot_series(data, outpath):
    plt.figure()
    plt.plot(data)
    plt.savefig(outpath)
    return outpath
""",
    "fib": """def fib(n):
    a,b=0,1
    for _ in range(n): a,b=b,a+b
    return a
""",
    "factorial": """def factorial(n):
    return 1 if n<=1 else n*factorial(n-1)
""",
    "bubble_sort": """def bubble_sort(lst):
    arr = lst[:]
    for i in range(len(arr)):
        for j in range(len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
""",
    "tokenize": """def tokenize(sentence):
    return sentence.split()
""",
    "count_words": """def count_words(text):
    return len(text.split())
""",
    "train_logreg": """from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def train_logreg(X, y):
    model = LogisticRegression(max_iter=200).fit(X, y)
    return float(accuracy_score(y, model.predict(X)))
""",
    "train_linreg": """from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
def train_linreg(X, y):
    model = LinearRegression().fit(X, y)
    return float(r2_score(y, model.predict(X)))
""",
    "gpu_matrix_mul": """import torch
def gpu_matrix_mul(A, B):
    A_gpu = torch.tensor(A, dtype=torch.float32).cuda()
    B_gpu = torch.tensor(B, dtype=torch.float32).cuda()
    result = torch.matmul(A_gpu, B_gpu)
    return result.cpu().tolist()
"""
}

# -------------------------------
# Candidate generator
# -------------------------------
def legacy_generate_candidate(task):
    tid = task["id"]
    if task.get("type") == "gpu":
        return BUILDING_BLOCKS["gpu_matrix_mul"]

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

    num = int(tid.replace("task", ""))
    mod = num % 10
    if mod == 0: return BUILDING_BLOCKS["bubble_sort"]
    if mod == 1: return BUILDING_BLOCKS["factorial"]
    if mod == 2: return """import pandas as pd
def count_rows(path): return len(pd.read_csv(path))"""
    if mod == 3: return """import matplotlib.pyplot as plt
def plot_hist(data, outpath):
    plt.hist(data); plt.savefig(outpath); return outpath"""
    if mod == 4: return """def unique_word_count(path):
    with open(path) as f: return len(set(f.read().split()))"""
    if mod == 5: return """import numpy as np
def zscore_normalize(arr):
    arr = np.array(arr, dtype=float)
    return (arr - arr.mean())/arr.std()"""
    if mod == 6: return BUILDING_BLOCKS["train_logreg"]
    if mod == 7: return """import json
def merge_json(path1,path2,outpath):
    with open(path1) as f1, open(path2) as f2:
        d1,d2=json.load(f1),json.load(f2)
    with open(outpath,'w') as f: json.dump({**d1,**d2},f)"""
    if mod == 8: return """import requests
def fetch_json(url): return requests.get(url).json()"""
    if mod == 9: return BUILDING_BLOCKS["fib"]


# -------------------------------
# Curriculum memory
# -------------------------------
def remember_success(task, code):
    record = {
        "task_id": task["id"],
        "description": task.get("description", ""),
        "function": task.get("function", ""),
        "pipeline": task.get("pipeline", []),
        "code": code,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(SUCCESSFUL_PIPELINES, "a") as f:
        f.write(json.dumps(record) + "\n")

def recall_pipeline_with_pipeline(task, threshold=0.6):
    if not os.path.exists(SUCCESSFUL_PIPELINES):
        return None, None
    records, descs = [], []
    with open(SUCCESSFUL_PIPELINES) as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
            descs.append(rec["description"])
    if not records:
        return None, None
    vectorizer = TfidfVectorizer().fit(descs + [task.get("description","")])
    desc_vecs = vectorizer.transform(descs)
    task_vec = vectorizer.transform([task.get("description","")])

    sims = cosine_similarity(task_vec, desc_vecs)[0]
    best_idx = sims.argmax()
    if sims[best_idx] >= threshold:
        return records[best_idx]["code"], records[best_idx].get("pipeline", [])
    return None, None

# -------------------------------
# Mutation and crossover
# -------------------------------
def mutate_pipeline(pipeline, task):
    new_pipeline = pipeline[:]
    ops = ["add", "remove", "swap"]
    op = random.choice(ops)

    if op == "add":
        candidates = [b for b in BUILDING_BLOCKS if b not in new_pipeline]
        if candidates:
            new_pipeline.insert(random.randint(0, len(new_pipeline)), random.choice(candidates))
    elif op == "remove" and len(new_pipeline) > 1:
        new_pipeline.pop(random.randrange(len(new_pipeline)))
    elif op == "swap" and len(new_pipeline) > 1:
        i, j = random.sample(range(len(new_pipeline)), 2)
        new_pipeline[i], new_pipeline[j] = new_pipeline[j], new_pipeline[i]

    return new_pipeline

def crossover_pipelines(p1, p2):
    if not p1 or not p2:
        return p1 or p2
    cut1 = random.randint(1, len(p1))
    cut2 = random.randint(1, len(p2))
    return p1[:cut1] + p2[cut2:]

def random_pipeline(min_len=2, max_len=4):
    length = random.randint(min_len, max_len)
    return random.sample(list(BUILDING_BLOCKS.keys()), length)

# -------------------------------
# Evolutionary search
# -------------------------------
def evolutionary_search(task, pop_size=6, n_generations=3, elite_size=2):
    base_code, base_pipeline = recall_pipeline_with_pipeline(task)
    population = []
    if base_pipeline:
        population.append(base_pipeline)
    while len(population) < pop_size:
        population.append(random_pipeline())

    best_overall = None

    for gen in range(n_generations):
        scored = []
        for pipe in population:
            task["pipeline"] = pipe
            code = generate_candidate(task)
            result = run_in_sandbox(task, code)
            scored.append((result["score"], pipe, code, result))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_pipe, best_code, best_result = scored[0]
        if not best_overall or best_score > best_overall[0]:
            best_overall = (best_score, best_pipe, best_code, best_result)

        for _, pipe, code, result in scored:
            log_result(result)
            if result["success"]:
                remember_success(task, code)

        elites = [pipe for _, pipe, _, _ in scored[:elite_size]]
        next_pop = elites[:]
        while len(next_pop) < pop_size:
            if random.random() < 0.5 and len(elites) >= 2:
                p1, p2 = random.sample(elites, 2)
                child = crossover_pipelines(p1, p2)
            else:
                parent = random.choice(elites)
                child = mutate_pipeline(parent, task)
            next_pop.append(child)

        population = next_pop

    return best_overall[2], best_overall[3]

# -------------------------------
# Task selection
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

    func_name = task.get("function", "run_pipeline")

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

    try:
        os.remove(candidate_path)
    except OSError:
        pass

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
        "error": result["error"][:200]
    }
    with open(TASK_RESULTS, "a") as f:
        f.write(json.dumps(summary) + "\n")

# -------------------------------
# One training cycle
# -------------------------------
def run_training_cycle():
    task = pick_next_task()
    code, result = evolutionary_search(task)
    log_result(result)
    if result["success"]:
        remember_success(task, code)
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


def generate_candidate(task):
    pipeline = task.get("pipeline")
    func_name = task.get("function", "run_pipeline")

    if pipeline:
        code_parts = []
        for step in pipeline:
            if step in BUILDING_BLOCKS:
                code_parts.append(BUILDING_BLOCKS[step])

        driver_lines = [f"def {func_name}(**kwargs):", "    _last = None"]
        for step in pipeline:
            if step == "load_csv":
                driver_lines.append("    _last = load_csv(kwargs['path'])")
            elif step == "normalize":
                driver_lines.append("    _last = normalize(_last, kwargs['column'])")
            elif step == "filter_rows":
                driver_lines.append("    _last = filter_rows(_last, kwargs['column'], kwargs['threshold'])")
            elif step == "aggregate_mean":
                driver_lines.append("    _last = aggregate_mean(_last, kwargs['column'])")
            elif step == "plot_histogram":
                driver_lines.append("    _last = plot_histogram(_last, kwargs['column'], kwargs['outpath'])")
            elif step == "plot_series":
                driver_lines.append("    _last = plot_series(kwargs['data'], kwargs['outpath'])")
            elif step == "train_logreg":
                driver_lines.append("    _last = train_logreg(kwargs['X'], kwargs['y'])")
            elif step == "train_linreg":
                driver_lines.append("    _last = train_linreg(kwargs['X'], kwargs['y'])")
            elif step == "gpu_matrix_mul":
                driver_lines.append("    _last = gpu_matrix_mul(kwargs['A'], kwargs['B'])")
            else:
                driver_lines.append(f"    _last = {step}(**kwargs)")
        driver_lines.append("    return _last")
        code_parts.append("\n".join(driver_lines))
        return "\n".join(code_parts)

    legacy_code = legacy_generate_candidate(task)
    if f"def {func_name}" not in legacy_code:
        legacy_code += f"\n{func_name} = " + legacy_code.split("def ")[1].split("(")[0]
    return legacy_code

