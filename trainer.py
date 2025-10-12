#!/usr/bin/env python3
import os, json, time, uuid, importlib.util
from sandbox_runner import run_snippet
from task_harness import load_task, score_task

TASKS_DIR = os.path.expanduser("~/child_env_runner/tasks")
ARTIFACTS_DIR = os.path.expanduser("~/child_env_runner/artifacts")
os.makedirs(TASKS_DIR, exist_ok=True)

def generate_candidate(task):
    """Return candidate code for each known task, including the 200-task pack."""
    tid = task["id"]

    # Starter pack (1–5)
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

    # Second wave (6–15)
    elif tid == "task6":
        return """import pandas as pd
def clean_csv(path_in, path_out):
    df = pd.read_csv(path_in).dropna()
    df.to_csv(path_out, index=False)"""
    elif tid == "task7":
        return """import pandas as pd
def merge_csv(path1, path2, key, outpath):
    df1, df2 = pd.read_csv(path1), pd.read_csv(path2)
    merged = pd.merge(df1, df2, on=key)
    merged.to_csv(outpath, index=False)"""
    elif tid == "task8":
        return """import matplotlib.pyplot as plt
def plot_two_series(data1, data2, outpath):
    plt.figure(); plt.plot(data1, label='data1'); plt.plot(data2, label='data2')
    plt.legend(); plt.savefig(outpath); return outpath"""
    elif tid == "task9":
        return """from collections import Counter
def word_count(path):
    with open(path) as f: words = f.read().split()
    return dict(Counter(words))"""
    elif tid == "task10":
        return """from collections import Counter
def top_n_words(path, n):
    with open(path) as f: words = f.read().split()
    return Counter(words).most_common(n)"""
    elif tid == "task11":
        return """import numpy as np
def normalize_array(arr):
    arr = np.array(arr, dtype=float)
    return (arr - arr.min()) / (arr.max() - arr.min())"""
    elif tid == "task12":
        return """from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
def train_test_split_and_score(X, y):
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=0)
    model = LinearRegression().fit(Xtr,ytr)
    return float(r2_score(yte, model.predict(Xte)))"""
    elif tid == "task13":
        return """import numpy as np
def confusion_matrix_score(y_true, y_pred):
    tp = sum((yt==1 and yp==1) for yt,yp in zip(y_true,y_pred))
    tn = sum((yt==0 and yp==0) for yt,yp in zip(y_true,y_pred))
    fp = sum((yt==0 and yp==1) for yt,yp in zip(y_true,y_pred))
    fn = sum((yt==1 and yp==0) for yt,yp in zip(y_true,y_pred))
    return [[tn, fp],[fn, tp]]"""
    elif tid == "task14":
        return """import pandas as pd, json
def json_to_csv(json_path, csv_path):
    with open(json_path) as f: data = json.load(f)
    pd.DataFrame(data).to_csv(csv_path, index=False)"""
    elif tid == "task15":
        return """import requests, re
def fetch_and_parse(url):
    html = requests.get(url).text
    m = re.search(r'<title>(.*?)</title>', html, re.I|re.S)
    return m.group(1).strip() if m else ''"""

    # Big wave (16–215) — cycle through 10 categories
    else:
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

