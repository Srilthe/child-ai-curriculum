"""
Task: ffn_up_proj
Request: Merge tensor 'ffn_up_proj' with a random scale factor.
Ensure reproducibility and log diagnostics.
"""

import numpy as np

base = np.load("ffn_up_proj_base.npy")
adapter = np.load("ffn_up_proj_adapter.npy")
merged = base + 0.5 * adapter
np.save("ffn_up_proj.merged.npy", merged)
print("Merge complete:", merged.shape)
