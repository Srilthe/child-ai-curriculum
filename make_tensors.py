import numpy as np

# Base tensor: all ones
base = np.ones((4, 4), dtype=np.float32)
np.save("base_tensor.npy", base)

# LoRA adapter: small random deltas
lora = np.random.randn(4, 4).astype(np.float32) * 0.01
np.save("lora_adapter.npy", lora)

print("Saved base_tensor.npy and lora_adapter.npy")

