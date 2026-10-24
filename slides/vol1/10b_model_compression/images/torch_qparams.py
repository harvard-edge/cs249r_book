# Lecture 10b: PyTorch reproduces Napkin Math 10.1 (range [-1, 3], UINT8).
# Run: python -W ignore torch_qparams.py   (torch 2.13.0, CPU)
import torch
from torch.ao.quantization.observer import MinMaxObserver

u8 = torch.quint8
x = torch.tensor([[-1.0, 1.0], [0.0, 3.0]])
obs = MinMaxObserver(dtype=u8)
obs(x)
s, z = obs.calculate_qparams()
q = torch.quantize_per_tensor(x, s.item(), z.item(), u8)
print(s.item(), z.item())
print(q.int_repr())
print(q.dequantize())
