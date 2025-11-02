import numpy as np
import json
import time
from pathlib import Path

def merge_lora(base_tensor, lora_adapter, scale=1.0, tag="merge"):
    if base_tensor.shape != lora_adapter.shape:
        raise ValueError("Shape mismatch")

    result = base_tensor + scale * lora_adapter

    log = {
        "event": "tensor_merge",
        "timestamp": time.time(),
        "tag": tag,
        "shape": list(base_tensor.shape),
        "scale": scale,
        "mean_delta": float(np.mean(scale * lora_adapter)),
        "max_delta": float(np.max(scale * lora_adapter))
    }

    with open(f"{tag}.merge.json", "w") as f:
        json.dump(log, f, indent=2)

    return result

def load_tensor(path, fallback=None):
    if Path(path).exists():
        return np.load(path)
    elif fallback is not None:
        print(f"[adaptive_merge] Warning: {path} not found, using fallback.")
        return fallback
    else:
        raise FileNotFoundError(f"Missing tensor file: {path}")

if __name__ == "__main__":
    # Tensor name to merge
    tensor_name = "token_embd.weight"
    base_path = f"{tensor_name}.base.npy"
    lora_path = f"{tensor_name}.lora.npy"

    fallback_base = np.ones((4, 4), dtype=np.float32)
    fallback_lora = np.random.randn(4, 4).astype(np.float32) * 0.01

    base = load_tensor(base_path, fallback=fallback_base)
    lora = load_tensor(lora_path, fallback=fallback_lora)

    merged = merge_lora(base, lora, scale=0.5, tag=tensor_name)
    print(f"Merged tensor for {tensor_name}:\n", merged)

