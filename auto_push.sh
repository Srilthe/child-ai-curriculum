#!/bin/bash
cd /home/slider/child_env_runner || exit 1

# Stage all changes
/usr/bin/git add -A

# Commit with timestamp (skip if nothing changed)
/usr/bin/git commit -m "Auto-push $(date +'%Y-%m-%d %H:%M:%S')" || true

# Push to remote
/usr/bin/git push origin main

