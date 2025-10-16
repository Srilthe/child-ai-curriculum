#!/bin/bash
set -e

REPO_DIR="/home/slider/child_env_runner"
cd "$REPO_DIR"

# Stage all changes
git add -A

# Commit with timestamp
COMMIT_MSG="Auto-push $(date '+%Y-%m-%d %H:%M:%S')"
git commit -m "$COMMIT_MSG" || echo "No changes to commit."

# Create a timestamped backup archive of critical files
BACKUP_FILE="trbak.$(date '+%s').tar.gz"
tar -czf "$BACKUP_FILE" trainer.py tasks artifacts 2>/dev/null || true

# Keep only the 10 most recent backups
ls -1tr trbak.*.tar.gz 2>/dev/null | head -n -10 | xargs -r rm -f

# Push to GitHub
git push origin main

