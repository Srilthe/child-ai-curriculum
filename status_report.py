import os
from datetime import datetime

def latest_artifact():
    files = [f for f in os.listdir("artifacts") if f.startswith("candidate_")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join("artifacts", f)), reverse=True)
    return files[0] if files else "None"

print("Latest candidate:", latest_artifact())
print("Trainer active:", "trainer.py" in os.popen("ps aux").read())
print("Artifacts size:", os.popen("du -sh artifacts").read().strip())
print("Last updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

