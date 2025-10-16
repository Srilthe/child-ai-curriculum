#!/usr/bin/env python3
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    x = torch.randn((4096, 4096), device="cuda")
    y = torch.mm(x, x.T)
    print("MM done, y shape:", y.shape, "dtype:", y.dtype)

