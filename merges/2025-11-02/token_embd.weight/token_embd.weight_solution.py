"""
Task: token_embd.weight
Request: Merge tensor 'token_embd.weight' with a random scale factor.
Ensure reproducibility and log diagnostics.
"""

import numpy as np

base = np.load("token_embd.weight_base.npy")
adapter = np.load("token_embd.weight_adapter.npy")
merged = base + 0.5 * adapter
np.save("token_embd.weight.merged.npy", merged)
print("Merge complete:", merged.shape)
