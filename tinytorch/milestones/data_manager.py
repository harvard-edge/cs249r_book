#!/usr/bin/env python3
"""
TinyTorch Dataset Manager
========================

Handles dataset downloading and preparation for milestone examples.
Students can focus on demonstrating their ML systems, not fighting with data logistics!

Supported Datasets:
- MNIST: Handwritten digits (28x28 grayscale)
- CIFAR-10: Natural images (32x32 RGB)  
- XOR: Synthetic non-linear problem
- Perceptron: Synthetic linearly separable data
"""

import os
import sys
import urllib.request
import tarfile
import pickle
import gzip
import numpy as np
from pathlib import Path

# Add project root for TinyTorch imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class DatasetManager:
    """Handles all dataset logistics for TinyTorch milestone examples."""
    
    def __init__(self, data_dir=None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent / "datasets"
        else:
            self.data_dir = Path(data_dir)
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)
        
    def download_with_progress(self, url, filename):
        """Download with visual progress bar."""
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                
                # Visual progress bar
                bar_length = 30
                filled = int(bar_length * percent / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                print(f"\r   [{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
        
        print(f"📥 Downloading {Path(filename).name}...")
        urllib.request.urlretrieve(url, filename, progress_hook)
        print("\n✅ Download complete!")
    
    def get_mnist(self):
        """Download and prepare MNIST dataset for MLP milestone."""
        mnist_dir = self.data_dir / "mnist"
        mnist_dir.mkdir(exist_ok=True)
        
        # MNIST URLs
        base_url = "http://yann.lecun.com/exdb/mnist/"
        files = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz", 
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        ]
        
        # Download if needed
        for filename in files:
            filepath = mnist_dir / filename
            if not filepath.exists():
                self.download_with_progress(base_url + filename, filepath)
        
        # Load and return data
        train_images = self._load_mnist_images(mnist_dir / files[0])
        train_labels = self._load_mnist_labels(mnist_dir / files[1])
        test_images = self._load_mnist_images(mnist_dir / files[2])
        test_labels = self._load_mnist_labels(mnist_dir / files[3])
        
        print(f"📊 MNIST loaded: {len(train_images)} training, {len(test_images)} test images")
        return (train_images, train_labels), (test_images, test_labels)
    
    def get_cifar10(self):
        """Download and prepare CIFAR-10 dataset for CNN milestone."""
        cifar_dir = self.data_dir / "cifar-10"
        cifar_dir.mkdir(exist_ok=True)
        
        # Check if already downloaded
        data_file = cifar_dir / "cifar-10-python.tar.gz"
        extracted_dir = cifar_dir / "cifar-10-batches-py"
        
        if not extracted_dir.exists():
            # Download if needed
            if not data_file.exists():
                url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
                self.download_with_progress(url, data_file)
            
            # Extract
            print("📦 Extracting CIFAR-10...")
            with tarfile.open(data_file, 'r:gz') as tar:
                tar.extractall(cifar_dir)
            print("✅ Extraction complete!")
        
        # Load data from pickle files
        train_data, train_labels = [], []
        for i in range(1, 6):
            batch_file = extracted_dir / f"data_batch_{i}"
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                train_data.append(batch[b'data'])
                train_labels.extend(batch[b'labels'])
        
        # Test data
        test_file = extracted_dir / "test_batch"
        with open(test_file, 'rb') as f:
            test_batch = pickle.load(f, encoding='bytes')
            test_data = test_batch[b'data']
            test_labels = test_batch[b'labels']
        
        # Reshape to proper image format
        train_data = np.vstack(train_data).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        test_data = test_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        train_labels = np.array(train_labels, dtype=np.int64)
        test_labels = np.array(test_labels, dtype=np.int64)
        
        print(f"📊 CIFAR-10 loaded: {len(train_data)} training, {len(test_data)} test images")
        return (train_data, train_labels), (test_data, test_labels)
    
    def get_xor_data(self, num_samples=1000):
        """Generate XOR problem data for non-linear milestone."""
        print("🧮 Generating XOR problem data...")
        
        # Create XOR dataset
        np.random.seed(42)  # Reproducible
        X = np.random.randint(0, 2, (num_samples, 2)).astype(np.float32)
        # XOR: output 1 when inputs differ, 0 when same
        y = (X[:, 0].astype(int) != X[:, 1].astype(int)).astype(np.int64)
        
        # Add some noise to make it more realistic
        X += np.random.normal(0, 0.1, X.shape)
        
        print(f"📊 XOR data generated: {num_samples} samples")
        print("   Classes: [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0")
        return X, y
    
    def get_perceptron_data(self, num_samples=1000):
        """Generate linearly separable data for perceptron milestone."""
        print("📏 Generating linearly separable data...")
        
        np.random.seed(42)
        
        # Create two clusters
        cluster1 = np.random.normal([2, 2], 0.5, (num_samples//2, 2))
        cluster2 = np.random.normal([-2, -2], 0.5, (num_samples//2, 2))
        
        X = np.vstack([cluster1, cluster2]).astype(np.float32)
        y = np.hstack([np.ones(num_samples//2), np.zeros(num_samples//2)]).astype(np.int64)
        
        # Shuffle
        indices = np.random.permutation(num_samples)
        X, y = X[indices], y[indices]
        
        print(f"📊 Perceptron data generated: {num_samples} linearly separable samples")
        return X, y
    
    def _load_mnist_images(self, filepath):
        """Load MNIST image file."""
        with gzip.open(filepath, 'rb') as f:
            # Skip header
            f.read(16)
            # Read images
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(-1, 28, 28).astype(np.float32) / 255.0
    
    def _load_mnist_labels(self, filepath):
        """Load MNIST label file."""
        with gzip.open(filepath, 'rb') as f:
            # Skip header  
            f.read(8)
            # Read labels
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

def main():
    """Test dataset manager functionality."""
    print("🧪 Testing TinyTorch Dataset Manager")
    print("=" * 50)
    
    manager = DatasetManager()
    
    # Test each dataset
    print("\n1. Testing Perceptron Data:")
    X, y = manager.get_perceptron_data(100)
    print(f"   Shape: X={X.shape}, y={y.shape}")
    
    print("\n2. Testing XOR Data:")
    X, y = manager.get_xor_data(100)
    print(f"   Shape: X={X.shape}, y={y.shape}")
    
    print("\n3. Testing MNIST (this may take a moment):")
    try:
        (train_X, train_y), (test_X, test_y) = manager.get_mnist()
        print(f"   Shape: train_X={train_X.shape}, test_X={test_X.shape}")
    except Exception as e:
        print(f"   MNIST download failed: {e}")
    
    print("\n4. Testing CIFAR-10 (this may take a moment):")
    try:
        (train_X, train_y), (test_X, test_y) = manager.get_cifar10()
        print(f"   Shape: train_X={train_X.shape}, test_X={test_X.shape}")
    except Exception as e:
        print(f"   CIFAR-10 download failed: {e}")
    
    print("\n✅ Dataset Manager test complete!")

if __name__ == "__main__":
    main()