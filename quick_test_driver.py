#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_test_driver.py

Simple harness to check that model_driver.generate_code works end-to-end.
"""

from model_driver import generate_code

def main():
    prompt = "Write a Python function factorial(n) that returns n! using recursion."
    print("=== Prompt ===")
    print(prompt)
    print("\n=== Model Output ===")
    code = generate_code(prompt, max_tokens=128)
    print(code)

if __name__ == "__main__":
    main()
