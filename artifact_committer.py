from pathlib import Path
import subprocess
from datetime import datetime

def commit_success(task_name: str, candidate_path: Path, prompt_path: Path):
    """
    Wraps a passing candidate with its original request, organizes it into a dated subfolder,
    and commits to GitHub.
    """

    # Repo root (your GitHub repo)
    repo_dir = Path.home() / "child-ai-curriculum"

    # Organize by date and task
    date_str = datetime.now().strftime("%Y-%m-%d")
    dest_dir = repo_dir / "merges" / date_str / task_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Load prompt and candidate
    prompt_text = prompt_path.read_text().strip()
    candidate_code = Path(candidate_path).read_text()

    # Wrap into one self-contained file
    wrapped_code = (
        f'"""\nTask: {task_name}\nRequest: {prompt_text}\n"""\n\n{candidate_code}'
    )

    # Write wrapped solution into repo
    out_file = dest_dir / f"{task_name}_solution.py"
    out_file.write_text(wrapped_code)

    # Also copy the raw candidate and prompt for auditability
    raw_candidate_file = dest_dir / f"{task_name}_candidate.txt"
    raw_candidate_file.write_text(candidate_code)

    raw_prompt_file = dest_dir / f"{task_name}_prompt.txt"
    raw_prompt_file.write_text(prompt_text)

    # Stage and commit
    subprocess.run(["git", "-C", str(repo_dir), "add", str(dest_dir)], check=True)
    subprocess.run(
        ["git", "-C", str(repo_dir), "commit", "-m",
         f"[{date_str}] Add passing solution for {task_name}"],
        check=True
    )
    subprocess.run(["git", "-C", str(repo_dir), "push", "origin", "main"], check=True)

    print(f"[committer] Committed {out_file} and related files to GitHub")

