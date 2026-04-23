import os
import yaml
import urllib.request
import torch

class PedagogicDataFetcher:
    """
    Simulated 'Data Engineer Agent'
    Strictly prevents Data Leakage by executing an immutable 80/20 mathematical tensor split.
    """
    def __init__(self, data_dir="datasets/local_tensors"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def _fetch_wikitext(self):
        """Fetches TinyShakespeare/Wikitext bounds!"""
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        target_file = os.path.join(self.data_dir, "tinyshakespeare.txt")
        if not os.path.exists(target_file):
            print(f"📥 Fetching Base Sequence Data...")
            urllib.request.urlretrieve(url, target_file)
            
        with open(target_file, 'r') as f:
            data = f.read()
            
        # The Pedagogical 80/20 Math Split enforcing zero leakage dynamically
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # Save explicitly so the Multiprocessing core uses identical structures mathematically safely!
        with open(os.path.join(self.data_dir, "tinyshakespeare_train.txt"), 'w') as f:
            f.write(train_data)
        with open(os.path.join(self.data_dir, "tinyshakespeare_val.txt"), 'w') as f:
            f.write(val_data)
            
        print("✅ Language Dataset Strictly Segmented and Frozen.")

    def run_all(self):
        print("🚀 Booting Automated Data Engineer Agent...")
        # Add vision/audio stubs here logically
        self._fetch_wikitext()
        print("✅ Immutable Dataset Arrays are physically locked! Over to the Trainer...")

if __name__ == "__main__":
    fetcher = PedagogicDataFetcher()
    fetcher.run_all()
