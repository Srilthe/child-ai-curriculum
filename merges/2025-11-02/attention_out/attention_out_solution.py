"""
Task: attention_out
Request: Merge tensor 'attention_out' with a random scale factor.
Ensure reproducibility and log diagnostics.
"""

import numpy as np

base = np.load("attention_out_base.npy")
adapter = np.load("attention_out_adapter.npy")
merged = base + 0.5 * adapter
np.save("attention_out.merged.npy", merged)
print("Merge complete:", merged.shape)
