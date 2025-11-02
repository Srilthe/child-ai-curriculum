#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_driver.py

Invokes llama.cpp with a local WizardCoder model to generate Python code.
"""

import os
import subprocess
import logging

MODEL_PATH = os.getenv("MODEL_PATH", "./models/wizardcoder-python-13b-v1.0.Q4_K_M.gguf")
LLAMA_BIN = "./llama.cpp/build/bin/main"


def generate_code(prompt: str, max_tokens: int = 256) -> str:
    """Call llama.cpp to generate Python code from a prompt."""
    full_prompt = f"### Task:\n{prompt}\n\n### Python solution:\n"
    try:
        result = subprocess.run(
            [LLAMA_BIN, "-m", MODEL_PATH, "-p", full_prompt, "-n", str(max_tokens)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error("Model invocation failed: %s", e.stderr)
        return ""
