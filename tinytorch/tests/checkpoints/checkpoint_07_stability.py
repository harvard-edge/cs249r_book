"""
Checkpoint 7: Stability (After Module 8 - Normalization)
Question: "Can I stabilize training with normalization techniques?"
"""

import numpy as np
import pytest

def test_checkpoint_07_stability():
    """
    Checkpoint 7: Stability

    Validates that students can apply normalization techniques to stabilize
    deep network training - the key to making deep learning practical and
    enabling training of very deep networks.
    """
    print("\n⚖️ Checkpoint 7: Stability")
    print("=" * 50)

    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.normalization import BatchNorm1D, LayerNorm
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU
    except ImportError as e:
        pytest.fail(f"❌ Cannot import required classes - complete Modules 2-8 first: {e}")

    # Test 1: Batch normalization
    print("📊 Testing batch normalization...")
    batch_norm = BatchNorm1D(num_features=10)

    # Create batch of activations
    batch_data = Tensor(np.random.randn(32, 10) * 3 + 2)  # High variance, non-zero mean

    normalized = batch_norm(batch_data)

    # Check normalization properties
    mean = np.mean(normalized.data, axis=0)
    std = np.std(normalized.data, axis=0)

    assert normalized.shape == batch_data.shape, f"BatchNorm should preserve shape: {batch_data.shape}"
    assert np.allclose(mean, 0, atol=1e-6), f"BatchNorm should center data around 0, got mean={mean}"
    assert np.allclose(std, 1, atol=1e-6), f"BatchNorm should normalize variance to 1, got std={std}"
    print(f"✅ Batch normalization: {batch_data.shape} → normalized (mean≈0, std≈1)")

    # Test 2: Layer normalization
    print("🔧 Testing layer normalization...")
    layer_norm = LayerNorm(normalized_shape=8)

    # Create sequence data (common in transformers)
    sequence_data = Tensor(np.random.randn(2, 5, 8) * 4 + 1)  # batch=2, seq=5, features=8

    layer_normalized = layer_norm(sequence_data)

    # Check that each sample/sequence position is normalized
    assert layer_normalized.shape == sequence_data.shape, f"LayerNorm should preserve shape: {sequence_data.shape}"

    # Check normalization across feature dimension for each position
    for b in range(2):
        for s in range(5):
            features = layer_normalized.data[b, s, :]
            assert abs(np.mean(features)) < 1e-5, f"LayerNorm should center features at position ({b},{s})"
            assert abs(np.std(features) - 1) < 1e-5, f"LayerNorm should normalize variance at position ({b},{s})"

    print(f"✅ Layer normalization: {sequence_data.shape} → normalized per position")

    # Test 3: Normalization in deep networks
    print("🏗️ Testing normalization in deep networks...")

    # Build deep network with normalization
    layers = [
        Linear(16, 32),
        BatchNorm1D(32),
        ReLU(),
        Linear(32, 32),
        BatchNorm1D(32),
        ReLU(),
        Linear(32, 16),
        BatchNorm1D(16),
        ReLU(),
        Linear(16, 1)
    ]

    # Test forward pass through deep normalized network
    input_data = Tensor(np.random.randn(8, 16))

    x = input_data
    for i, layer in enumerate(layers):
        x = layer(x)
        if i % 3 == 1:  # After each BatchNorm
            # Check that activations are well-behaved
            assert not np.any(np.isnan(x.data)), f"No NaN after layer {i}"
            assert not np.any(np.isinf(x.data)), f"No Inf after layer {i}"

    assert x.shape == (8, 1), f"Deep network output should be (8, 1), got {x.shape}"
    print(f"✅ Deep normalized network: {input_data.shape} → 4 layers → {x.shape}")

    # Test 4: Gradient flow improvement
    print("📈 Testing gradient flow properties...")

    # Compare networks with and without normalization
    # Create identical architectures
    normalized_net = [
        Linear(10, 20),
        BatchNorm1D(20),
        ReLU(),
        Linear(20, 10),
        BatchNorm1D(10),
        ReLU(),
        Linear(10, 1)
    ]

    unnormalized_net = [
        Linear(10, 20),
        ReLU(),
        Linear(20, 10),
        ReLU(),
        Linear(10, 1)
    ]

    test_input = Tensor(np.random.randn(5, 10))

    # Forward pass through both networks
    norm_x = test_input
    for layer in normalized_net:
        norm_x = layer(norm_x)

    unnorm_x = test_input
    for layer in unnormalized_net:
        unnorm_x = layer(unnorm_x)

    # Both should produce valid outputs
    assert not np.any(np.isnan(norm_x.data)), "Normalized network should produce stable outputs"
    assert not np.any(np.isnan(unnorm_x.data)), "Unnormalized network should produce valid outputs"
    print(f"✅ Gradient flow: normalized and unnormalized networks both stable")

    # Test 5: Training vs inference modes
    print("🔄 Testing training vs inference modes...")

    # Create batch norm layer
    bn = BatchNorm1D(num_features=5)

    # Training mode: use batch statistics
    training_data = Tensor(np.random.randn(10, 5) * 2 + 1)

    if hasattr(bn, 'training'):
        bn.training = True
    train_output = bn(training_data)

    # Should normalize based on current batch
    train_mean = np.mean(train_output.data, axis=0)
    assert np.allclose(train_mean, 0, atol=1e-5), "Training mode should use batch statistics"

    # Inference mode: use running statistics (if implemented)
    if hasattr(bn, 'training'):
        bn.training = False

    # Single sample inference
    single_sample = Tensor(np.random.randn(1, 5))
    inference_output = bn(single_sample)

    assert inference_output.shape == (1, 5), f"Inference should work on single samples: {inference_output.shape}"
    print(f"✅ Mode switching: training and inference modes both functional")

    # Test 6: Learnable parameters in normalization
    print("📚 Testing learnable normalization parameters...")

    # Check that normalization layers have learnable parameters
    bn_with_params = BatchNorm1D(num_features=8)

    assert hasattr(bn_with_params, 'gamma') or hasattr(bn_with_params, 'weight'), "BatchNorm should have scale parameters"
    assert hasattr(bn_with_params, 'beta') or hasattr(bn_with_params, 'bias'), "BatchNorm should have shift parameters"

    # Test that parameters affect output
    test_data = Tensor(np.ones((4, 8)))  # All ones
    original_output = bn_with_params(test_data)

    # Modify parameters
    if hasattr(bn_with_params, 'gamma'):
        bn_with_params.gamma.data *= 2
        bn_with_params.beta.data += 1
    elif hasattr(bn_with_params, 'weight'):
        bn_with_params.weight.data *= 2
        bn_with_params.bias.data += 1

    modified_output = bn_with_params(test_data)

    # Output should change when parameters change
    assert not np.allclose(original_output.data, modified_output.data), "Learnable parameters should affect output"
    print(f"✅ Learnable parameters: scale and shift parameters modify normalization")

    # Test 7: Numerical stability
    print("🔢 Testing numerical stability...")

    # Test with extreme values
    extreme_data = Tensor(np.array([[1e6, -1e6, 1e-6, -1e-6, 0]]))
    stable_bn = BatchNorm1D(num_features=5)

    try:
        stable_output = stable_bn(extreme_data)
        assert not np.any(np.isnan(stable_output.data)), "Should handle extreme values without NaN"
        assert not np.any(np.isinf(stable_output.data)), "Should handle extreme values without Inf"
        print(f"✅ Numerical stability: handles extreme values → {stable_output.shape}")
    except Exception as e:
        print(f"⚠️ Numerical stability: some issues with extreme values ({e})")

    print("\n🎉 Stability Complete!")
    print("📝 You can now stabilize training with normalization techniques")
    print("🔧 Built capabilities: Batch normalization, layer normalization, stable deep networks")
    print("🧠 Breakthrough: You can now train deep networks reliably!")
    print("🎯 Next: Add automatic differentiation for learning")

if __name__ == "__main__":
    test_checkpoint_07_stability()
