#!/bin/bash
cd /home/slider/child_env_runner

# Always pull the latest code
/usr/bin/git pull origin main

# Start the trainer
exec /home/slider/child_env/bin/python /home/slider/child_env_runner/trainer.py

