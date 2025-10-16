#!/bin/bash
set -euo pipefail
export PRUNE_TTL_DAYS=${PRUNE_TTL_DAYS:-14}
export PRUNE_MAX_LOG_MB=${PRUNE_MAX_LOG_MB:-256}
export PRUNE_KEEP_PER_TASK=${PRUNE_KEEP_PER_TASK:-500} # higher cap by default for rolling window
/home/slider/child_env/bin/python /home/slider/child_env_runner/tools/prune_artifacts.py

