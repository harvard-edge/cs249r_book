import torch, torchvision
from torchvision import models
print(f"torch {torch.__version__}, torchvision {torchvision.__version__}")
for name in ["mobilenet_v2", "resnet50", "vgg16"]:
    net = getattr(models, name)()          # architecture only, no weights
    n = sum(p.numel() for p in net.parameters())
    print(f"{name:13s} {n:>11,d} parameters")
