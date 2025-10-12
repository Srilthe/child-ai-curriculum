#!/usr/bin/env python3
import os, json, time, uuid, importlib.util
from sandbox_runner import run_snippet
from task_harness import load_task, score_task

TASKS_DIR = os.path.expanduser("~/child_env_runner/tasks")
ARTIFACTS_DIR = os.path.expanduser("~/child_env_runner/artifacts")
os.makedirs(TASKS_DIR, exist_ok=True)

def generate_candidate(task):
    """Return candidate code for each known task."""
    if task["id"] == "task1":
        return """def mean(lst):
    return sum(lst) / len(lst)
"""
    elif task["id"] == "task2":
        return """import pandas as pd
def load_and_sum(path):
    df = pd.read_csv(path)
    return int(df['value'].sum())
"""
    elif task["id"] == "task3":
        return """import matplotlib.pyplot as plt
def plot_series(data, outpath):
    plt.figure()
    plt.plot(data)
    plt.savefig(outpath)
    return outpath
"""
    elif task["id"] == "task4":
        return """from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
def train_and_score(X, y):
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    return float(r2_score(y, preds))
"""
    elif task["id"] == "task5":
        return """def count_lines(path):
    with open(path, 'r') as f:
        return sum(1 for _ in f)
"""
    else:
        return "print('No solution yet')"

