#!/usr/bin/env python3
"""
Original 1986 XOR Solution - Rumelhart, Hinton, Williams
Testing the MINIMAL architecture that solved the XOR crisis.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from tinytorch import Tensor, Linear, Sigmoid, BinaryCrossEntropyLoss, SGD

print("=" * 70)
print("🏛️  ORIGINAL 1986 XOR SOLUTION")
print("Rumelhart, Hinton, Williams - 'Learning representations by back-propagating errors'")
print("=" * 70)

# Pure XOR
X_data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
y_data = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

X = Tensor(X_data)
y = Tensor(y_data)

print("\n🏗️  Architecture (1986 style):")
print("  Input: 2 neurons")
print("  Hidden: 2 neurons (MINIMAL!)")
print("  Output: 1 neuron")
print("  Activation: Sigmoid (ReLU didn't exist yet!)")
print("  Total params: 9 (2×2 weights + 2 bias + 2×1 weights + 1 bias)")

# Original architecture: 2-2-1 with Sigmoid
hidden = Linear(2, 2)  # Only 2 hidden neurons!
sigmoid_hidden = Sigmoid()
output = Linear(2, 1)
sigmoid_output = Sigmoid()

loss_fn = BinaryCrossEntropyLoss()
optimizer = SGD([p for p in hidden.parameters()] + [p for p in output.parameters()], lr=1.0)

print("\n🔥 Training with original 1986 architecture...")
epochs = 2000  # May need more epochs with only 2 hidden units

for epoch in range(epochs):
    # Forward (all sigmoid, like 1986!)
    h = hidden(X)
    h_act = sigmoid_hidden(h)  # Sigmoid in hidden layer
    out = output(h_act)
    pred = sigmoid_output(out)  # Sigmoid in output layer
    loss = loss_fn(pred, y)

    # Backward
    loss.backward()

    # Update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 400 == 0:
        accuracy = ((pred.data > 0.5).astype(float) == y.data).mean()
        print(f"Epoch {epoch+1:4d}/{epochs}  Loss: {loss.data:.4f}  Accuracy: {accuracy:.1%}")

# Final evaluation
print("\n✅ Final Results:")
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

if final_accuracy == 1.0:
    print("\n🎉 SUCCESS! XOR solved with MINIMAL 1986 architecture!")
    print("   This is exactly what ended the AI Winter!")
else:
    print(f"\n⚠️  Accuracy: {final_accuracy:.1%} - may need more training")

# Show what the hidden units learned
print("\n🧠 What the 2 hidden neurons learned:")
print("   (Examining activation patterns)")
h_activations = sigmoid_hidden(hidden(X)).data
print(f"\n   Hidden unit activations for each input:")
for i, x_in in enumerate(X_data):
    print(f"   {x_in}: h1={h_activations[i,0]:.3f}, h2={h_activations[i,1]:.3f}")

print("\n" + "=" * 70)
print("💡 Historical Note:")
print("   This 2-2-1 architecture ended the 17-year AI Winter!")
print("   Proved that backprop + hidden layers solve 'impossible' problems")
print("=" * 70)
