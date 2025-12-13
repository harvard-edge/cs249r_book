"""
Checkpoint 2: Intelligence (After Module 3 - Activations)
Question: "Can I add nonlinearity - the key to neural network intelligence?"
"""

import numpy as np
import pytest

def test_checkpoint_02_intelligence():
    """
    Checkpoint 2: Intelligence

    Validates that students can apply activation functions to create nonlinear
    transformations - the key breakthrough that enables neural networks to
    learn complex patterns and exhibit intelligence.
    """
    print("\n🧠 Checkpoint 2: Intelligence")
    print("=" * 50)

    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax
    except ImportError as e:
        pytest.fail(f"❌ Cannot import required classes - complete Modules 2-3 first: {e}")

    # Test 1: ReLU - Sparsity and efficiency
    print("⚡ Testing ReLU activation...")
    data = Tensor([-2, -1, 0, 1, 2])
    relu = ReLU()
    relu_output = relu(data)

    expected_relu = np.array([0, 0, 0, 1, 2])
    assert np.array_equal(relu_output.data, expected_relu), f"ReLU failed: expected {expected_relu}, got {relu_output.data}"
    print(f"✅ ReLU creates sparsity: {data.data} → {relu_output.data}")

    # Test 2: Sigmoid - Probability outputs
    print("📊 Testing Sigmoid activation...")
    sigmoid = Sigmoid()
    sigmoid_output = sigmoid(data)

    # Sigmoid should output values between 0 and 1
    assert np.all(sigmoid_output.data >= 0) and np.all(sigmoid_output.data <= 1), "Sigmoid outputs should be in [0,1]"
    print(f"✅ Sigmoid creates probabilities: {data.data} → {sigmoid_output.data}")

    # Test 3: Tanh - Zero-centered outputs
    print("🎯 Testing Tanh activation...")
    tanh = Tanh()
    tanh_output = tanh(data)

    # Tanh should output values between -1 and 1
    assert np.all(tanh_output.data >= -1) and np.all(tanh_output.data <= 1), "Tanh outputs should be in [-1,1]"
    assert abs(tanh_output.data[2]) < 1e-6, "Tanh(0) should be approximately 0"
    print(f"✅ Tanh is zero-centered: {data.data} → {tanh_output.data}")

    # Test 4: Softmax - Probability distributions
    print("🎲 Testing Softmax activation...")
    logits = Tensor([1.0, 2.0, 3.0])
    softmax = Softmax()
    softmax_output = softmax(logits)

    # Softmax should sum to 1 and all values should be positive
    assert abs(np.sum(softmax_output.data) - 1.0) < 1e-6, f"Softmax should sum to 1, got {np.sum(softmax_output.data)}"
    assert np.all(softmax_output.data > 0), "All softmax outputs should be positive"
    print(f"✅ Softmax creates distribution: {logits.data} → {softmax_output.data} (sum={np.sum(softmax_output.data):.3f})")

    # Test 5: Chaining activations (nonlinear intelligence)
    print("🔗 Testing chained nonlinear transformations...")
    complex_data = Tensor([-3, -1, 0, 1, 3])

    # Apply multiple transformations: data → ReLU → Sigmoid
    step1 = relu(complex_data)  # Remove negative values
    intelligent_output = sigmoid(step1)  # Convert to probabilities

    assert np.all(intelligent_output.data >= 0) and np.all(intelligent_output.data <= 1), "Chained output should be valid probabilities"
    print(f"✅ Chained intelligence: {complex_data.data} → ReLU → Sigmoid → {intelligent_output.data}")

    # Test 6: Batch processing (multiple samples)
    print("📦 Testing batch processing...")
    batch_data = Tensor([[-1, 0, 1], [2, -2, 0]])  # 2 samples, 3 features each
    batch_output = relu(batch_data)

    expected_batch = np.array([[0, 0, 1], [2, 0, 0]])
    assert np.array_equal(batch_output.data, expected_batch), f"Batch ReLU failed: expected {expected_batch}, got {batch_output.data}"
    print(f"✅ Batch processing: {batch_data.shape} → {batch_output.shape}")

    print("\n🎉 Intelligence Complete!")
    print("📝 You can now add nonlinearity - the key to neural network intelligence")
    print("🔧 Built capabilities: ReLU, Sigmoid, Tanh, Softmax, chained activations, batch processing")
    print("🧠 Breakthrough: Networks can now learn complex, nonlinear patterns!")
    print("🎯 Next: Build learnable neural network components")

if __name__ == "__main__":
    test_checkpoint_02_intelligence()
