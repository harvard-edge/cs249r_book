#!/usr/bin/env python3
"""
Thorough XOR test to verify multi-layer networks work correctly.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from tinytorch import Tensor, Linear, ReLU, Sigmoid, BinaryCrossEntropyLoss, SGD

print("=" * 70)
print("🧪 THOROUGH XOR TEST - Verifying Multi-Layer Networks")
print("=" * 70)

# Pure XOR dataset (no noise)
X_data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
y_data = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

print("\n📋 XOR Truth Table:")
print("  (0,0) → 0")
print("  (0,1) → 1")
print("  (1,0) → 1")
print("  (1,1) → 0")

X = Tensor(X_data)
y = Tensor(y_data)

# Build network with better architecture
hidden_size = 8  # Increased from 4
hidden = Linear(2, hidden_size)
relu = ReLU()
output = Linear(hidden_size, 1)
sigmoid = Sigmoid()

loss_fn = BinaryCrossEntropyLoss()

# Try different learning rates
for lr in [1.0, 0.5, 0.1]:
    print(f"\n{'='*70}")
    print(f"🔥 Training with learning rate: {lr}")
    print('='*70)

    # Reset network
    hidden = Linear(2, hidden_size)
    output = Linear(hidden_size, 1)

    optimizer = SGD([p for p in hidden.parameters()] + [p for p in output.parameters()], lr=lr)

    epochs = 1000
    for epoch in range(epochs):
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

        if (epoch + 1) % 200 == 0:
            accuracy = ((pred.data > 0.5).astype(float) == y.data).mean()
            print(f"Epoch {epoch+1:4d}/{epochs}  Loss: {loss.data:.4f}  Accuracy: {accuracy:.1%}")

    # Final evaluation
    print("\n✅ Final Predictions:")
    final_accuracy = ((pred.data > 0.5).astype(float) == y.data).mean()
    for i in range(4):
        x_in = X_data[i]
        y_true = int(y_data[i, 0])
        y_pred_prob = pred.data[i, 0]
        y_pred = int(y_pred_prob > 0.5)
        status = "✅" if y_pred == y_true else "❌"
        print(f"  Input: {x_in}  →  Pred: {y_pred} (prob: {y_pred_prob:.3f})  True: {y_true}  {status}")

    print(f"\n📊 Final Accuracy: {final_accuracy:.1%}")
    print(f"📊 Final Loss: {loss.data:.4f}")

    if final_accuracy >= 0.95:
        print("🎉 SUCCESS! XOR is properly solved!")
        break
    else:
        print("⚠️  Not perfect yet, trying different learning rate...")

print("\n" + "=" * 70)
print("🏁 XOR Testing Complete")
print("=" * 70)
