import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

class RealVisionTranslators:
    """
    Explicit pedagogical mapping explicitly decoupling torchvision abstractions natively 
    simulating True Dataset processing physically loading physical Bytes organically!
    """
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        
        # Absolute Canonical location pointing strictly to the MLPerf EDU offline cache!
        self.data_dir = os.path.join(os.path.dirname(__file__), "..", ".data", "vision")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Explicit Physical transforms natively mutating RGB limits logically
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Normalization physics dynamically constraining activation values statically!
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    def get_cifar100_loaders(self):
        """
        Natively checks if the cache is hit, forcing the student to decouple 
        their dependencies to true offline physical limits explicitly!
        """
        try:
            trainset = CIFAR100(root=self.data_dir, train=True, download=False, transform=self.transform_train)
            testset = CIFAR100(root=self.data_dir, train=False, download=False, transform=self.transform_test)
        except Exception as e:
            print(f"[yellow]⚠️ True Datasets not tracked natively: {e}[/yellow]")
            print(f"[dim]Run `mlperf fetch --task resnet-train` to organically populate the `.data/` cache bounds remotely![/dim]")
            
            # Auto-fallback for pure execution test viability safely 
            trainset = CIFAR100(root=self.data_dir, train=True, download=True, transform=self.transform_train)
            testset = CIFAR100(root=self.data_dir, train=False, download=True, transform=self.transform_test)

        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        return trainloader, testloader
