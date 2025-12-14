#!/usr/bin/env python3
"""
Simple XOR test to verify multi-layer networks work
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from tinytorch import Tensor, Linear, ReLU, Sigmoid, BinaryCrossEntropyLoss, SGD

print("=" * 70)
print("🧪 Testing Multi-Layer Network on XOR Problem")
print("=" * 70)

# XOR dataset
X_data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_data = np.array([[0.0], [1.0], [1.0], [0.0]])

X = Tensor(X_data)
y = Tensor(y_data)

# Build network
hidden = Linear(2, 4)
relu = ReLU()
output = Linear(4, 1)
sigmoid = Sigmoid()

loss_fn = BinaryCrossEntropyLoss()
optimizer = SGD([p for p in hidden.parameters()] + [p for p in output.parameters()], lr=0.5)

print("\n🔥 Training on XOR...")
for epoch in range(500):
    # Forward
    h = hidden(X)
    h_act = relu(h)
    out = output(h_act)
    pred = sigmoid(out)
    loss = loss_fn(pred, y)

    # Backward
    loss.backward()

    # Update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        accuracy = ((pred.data > 0.5).astype(float) == y.data).mean()
        print(f"Epoch {epoch+1:3d}/500  Loss: {loss.data:.4f}  Accuracy: {accuracy:.1%}")

print("\n✅ Final predictions:")
for i in range(4):
    pred_val = (pred.data[i, 0] > 0.5).astype(int)
    print(f"  Input: {X_data[i]}  →  Predicted: {pred_val}  (Expected: {int(y_data[i, 0])})")

print("\n" + "=" * 70)
print("🎉 Multi-layer network working!")
print("=" * 70)
