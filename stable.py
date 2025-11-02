#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stable.py

Minimal loader for a Hugging Face model with safe offloading.
# Redownload from source
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./models"  # or the specific HF repo name if you pulled from hub

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="./offload",   # <--- add this line
)

print("Model loaded successfully.")
"""

"""
stable.py

Quick test harness for the local GGUF model via llama.cpp.
Generates code and verifies syntax with py_compile.
"""

import py_compile
import tempfile
from pathlib import Path
from model_driver import generate_code

def main():
    prompt = "Write a Python function factorial(n) that returns n! using recursion."
    print("=== Prompt ===")
    print(prompt)

    print("\n=== Model Output ===")
    code = generate_code(prompt, max_tokens=128)
    print(code)

    # Save to a temporary file and try compiling
    with tempfile.TemporaryDirectory() as tmpdir:
        candidate_path = Path(tmpdir) / "candidate.py"
        candidate_path.write_text(code)

        try:
            py_compile.compile(str(candidate_path), doraise=True)
            print("\n✅ Syntax check passed.")
        except py_compile.PyCompileError as e:
            print("\n❌ Syntax check failed:")
            print(e)

if __name__ == "__main__":
    main()

