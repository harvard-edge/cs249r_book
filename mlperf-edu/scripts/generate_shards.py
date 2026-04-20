import os
import numpy as np
import torch
from pathlib import Path

def generate_cifar10_shard(output_path, num_samples=1000):
    """
    Generates a 1,000-image shard of CIFAR-10.
    """
    print(f"🏗️ Generating CIFAR-10 shard ({num_samples} samples)...")
    try:
        from torchvision import datasets
    except ImportError:
        print("❌ Error: torchvision required for shard generation.")
        return

    # Download full dataset temporarily
    ds = datasets.CIFAR10(root="/tmp/cifar10", train=True, download=True)
    
    # Extract subset
    indices = np.random.choice(len(ds), num_samples, replace=False)
    data = ds.data[indices]
    targets = np.array(ds.targets)[indices]
    
    np.savez(output_path, data=data, targets=targets)
    print(f"✅ CIFAR-10 shard saved to {output_path} ({os.path.getsize(output_path)/1024**2:.2f} MB)")

def generate_speech_commands_shard(output_path, num_samples=1000):
    """
    Generates a 1,000-sample shard of Speech Commands (Mel-spectrograms).
    """
    print(f"🏗️ Generating Speech Commands shard ({num_samples} samples)...")
    try:
        import torchaudio
        from torchaudio.datasets import SPEECHCOMMANDS
    except ImportError:
        print("❌ Error: torchaudio required for shard generation.")
        return

    # Download full dataset temporarily
    ds = SPEECHCOMMANDS(root="/tmp/speech", download=True)
    
    # We want Mel-spectrograms (40 bins x 101 time steps)
    # This matches the standard KWS architecture expected in the labs
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=40, n_fft=480, hop_length=160
    )
    
    all_specs = []
    all_labels = []
    
    # Map labels to integers
    labels = sorted(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown'])
    label_to_idx = {l: i for i, l in enumerate(labels)}
    
    count = 0
    for waveform, sample_rate, label, speaker_id, utterance_number in ds:
        if count >= num_samples:
            break
            
        # Ensure 1 second (16000 samples)
        if waveform.shape[1] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
        else:
            waveform = waveform[:, :16000]
            
        spec = mel_spectrogram(waveform).squeeze().numpy()
        # Log scaling for spectrograms
        spec = np.log(spec + 1e-9)
        
        target = label_to_idx.get(label, label_to_idx['unknown'])
        
        all_specs.append(spec)
        all_labels.append(target)
        count += 1
        if count % 100 == 0:
            print(f"  Processed {count}/{num_samples}...")

    np.savez(output_path, data=np.array(all_specs), targets=np.array(all_labels))
    print(f"✅ Speech Commands shard saved to {output_path} ({os.path.getsize(output_path)/1024**2:.2f} MB)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["cifar10", "speech"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    if args.type == "cifar10":
        generate_cifar10_shard(args.out)
    else:
        generate_speech_commands_shard(args.out)
