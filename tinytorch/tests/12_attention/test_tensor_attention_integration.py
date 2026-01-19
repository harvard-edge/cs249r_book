"""
Integration Tests - Tensor and Attention

Tests cross-module interfaces and compatibility between Tensor and Attention modules.
Focuses on integration, not re-testing individual module functionality.
"""

import pytest
import numpy as np
from test_utils import setup_integration_test

# Ensure proper setup before importing
setup_integration_test()

# Import ONLY from TinyTorch package
from tinytorch.core.tensor import Tensor
from tinytorch.core.attention import scaled_dot_product_attention

# Try to import optional attention components (may not exist yet)
try:
    from tinytorch.core.attention import (
        SelfAttention,
        create_causal_mask,
        create_padding_mask,
        create_bidirectional_mask
    )
    HAS_ADVANCED_ATTENTION = True
except ImportError:
    HAS_ADVANCED_ATTENTION = False
    SelfAttention = None
    create_causal_mask = None
    create_padding_mask = None
    create_bidirectional_mask = None

# Skip this entire module if advanced attention is not available
pytestmark = pytest.mark.skipif(
    not HAS_ADVANCED_ATTENTION,
    reason="Advanced attention components not yet implemented"
)


class TestTensorAttentionInterface:
    """Test interface compatibility between Tensor and Attention modules."""

    def test_attention_accepts_tensor_data(self):
        """Test that attention functions accept Tensor.data input."""
        # Create Tensors
        seq_len, d_model = 4, 8
        Q = Tensor(np.random.randn(seq_len, d_model))
        K = Tensor(np.random.randn(seq_len, d_model))
        V = Tensor(np.random.randn(seq_len, d_model))

        # Test interface: attention should accept tensor.data
        output, weights = scaled_dot_product_attention(Q.data, K.data, V.data)

        # Verify interface compatibility (not functionality)
        assert isinstance(output, np.ndarray), "Attention should return numpy array compatible with Tensor"
        assert isinstance(weights, np.ndarray), "Attention weights should be numpy array"
        assert output.shape[0] == Q.shape[0], "Interface should preserve sequence dimension"
        assert output.shape[1] == V.shape[1], "Interface should preserve value dimension"

    def test_self_attention_tensor_interface(self):
        """Test SelfAttention class interface with Tensor objects."""
        d_model = 16
        seq_len = 6

        # Create SelfAttention and Tensor
        self_attn = SelfAttention(d_model)
        x = Tensor(np.random.randn(seq_len, d_model))

        # Test interface: SelfAttention should work with tensor.data
        output, weights = self_attn(x.data)

        # Verify interface compatibility
        assert isinstance(output, np.ndarray), "SelfAttention should return numpy arrays"
        assert isinstance(weights, np.ndarray), "SelfAttention should return numpy weights"
        assert output.shape == x.data.shape, "SelfAttention should preserve input shape"

        # Test that output can be converted back to Tensor
        result_tensor = Tensor(output)
        assert isinstance(result_tensor, Tensor), "Attention output should be convertible to Tensor"

    def test_attention_output_tensor_compatibility(self):
        """Test that attention outputs are compatible with Tensor creation."""
        seq_len, d_model = 5, 12

        # Create input tensors
        x = Tensor(np.random.randn(seq_len, d_model))

        # Apply attention
        self_attn = SelfAttention(d_model)
        output, weights = self_attn(x.data)

        # Test output compatibility with Tensor
        output_tensor = Tensor(output)
        weights_tensor = Tensor(weights)

        # Verify Tensor creation works
        assert isinstance(output_tensor, Tensor), "Attention output should create valid Tensor"
        assert isinstance(weights_tensor, Tensor), "Attention weights should create valid Tensor"
        assert output_tensor.shape == (seq_len, d_model), "Output Tensor should have correct shape"
        assert weights_tensor.shape == (seq_len, seq_len), "Weights Tensor should have correct shape"

    def test_masked_attention_tensor_interface(self):
        """Test that masking utilities work with Tensor-compatible data types."""
        seq_len = 6

        # Test mask creation (should create arrays compatible with Tensor)
        causal_mask = create_causal_mask(seq_len)
        padding_mask = create_padding_mask([seq_len, seq_len-2], seq_len)
        bidirectional_mask = create_bidirectional_mask(seq_len)

        # Test that masks can be used with Tensor data
        x = Tensor(np.random.randn(seq_len, 8))

        # Test interface: masks should work with tensor.data
        output, _ = scaled_dot_product_attention(x.data, x.data, x.data, causal_mask)

        # Verify interface compatibility
        assert isinstance(output, np.ndarray), "Masked attention should return numpy array"
        assert output.shape == x.data.shape, "Masked attention should preserve shape"

        # Test mask types are compatible
        assert causal_mask.dtype in [np.float32, np.float64, np.int32, np.int64], "Causal mask should have numeric dtype"
        assert padding_mask.dtype in [np.float32, np.float64, np.int32, np.int64], "Padding mask should have numeric dtype"


class TestAttentionTensorDataTypes:
    """Test data type compatibility between Tensor and Attention."""

    def test_float32_tensor_compatibility(self):
        """Test attention with float32 Tensor data."""
        seq_len, d_model = 3, 6

        # Create float32 tensors
        x_f32 = Tensor(np.random.randn(seq_len, d_model).astype(np.float32))

        # Test attention interface
        self_attn = SelfAttention(d_model)
        output, weights = self_attn(x_f32.data)

        # Verify dtype preservation in interface
        assert output.dtype == np.float32, "Attention should preserve float32 from Tensor"
        assert weights.dtype == np.float32, "Attention weights should be float32"

    def test_float64_tensor_compatibility(self):
        """Test attention with float64 Tensor data."""
        seq_len, d_model = 3, 6

        # Create float64 tensors
        x_f64 = Tensor(np.random.randn(seq_len, d_model).astype(np.float64))

        # Test attention interface
        self_attn = SelfAttention(d_model)
        output, weights = self_attn(x_f64.data)

        # Verify dtype preservation in interface
        assert output.dtype == np.float64, "Attention should preserve float64 from Tensor"
        assert weights.dtype == np.float64, "Attention weights should be float64"

    def test_batched_tensor_interface(self):
        """Test attention interface with batched Tensor data."""
        batch_size, seq_len, d_model = 2, 4, 8

        # Create batched tensor
        x_batch = Tensor(np.random.randn(batch_size, seq_len, d_model))

        # Test batched attention interface
        output, weights = scaled_dot_product_attention(x_batch.data, x_batch.data, x_batch.data)

        # Verify batched interface compatibility
        assert output.shape == x_batch.data.shape, "Batched attention should preserve tensor shape"
        assert weights.shape == (batch_size, seq_len, seq_len), "Batched weights should have correct shape"

        # Test that batched output can create Tensors
        output_tensor = Tensor(output)
        assert output_tensor.shape == x_batch.shape, "Batched output should create valid Tensor"


class TestAttentionTensorSystemIntegration:
    """Test system-level integration scenarios with Tensor and Attention."""

    def test_tensor_attention_tensor_roundtrip(self):
        """Test Tensor → Attention → Tensor roundtrip compatibility."""
        seq_len, d_model = 5, 10

        # Start with Tensor
        input_tensor = Tensor(np.random.randn(seq_len, d_model))

        # Apply attention (using tensor.data)
        self_attn = SelfAttention(d_model)
        attention_output, _ = self_attn(input_tensor.data)

        # Convert back to Tensor
        output_tensor = Tensor(attention_output)

        # Verify complete roundtrip works
        assert isinstance(output_tensor, Tensor), "Roundtrip should produce valid Tensor"
        assert output_tensor.shape == input_tensor.shape, "Roundtrip should preserve shape"
        assert output_tensor.dtype == input_tensor.dtype, "Roundtrip should preserve dtype"

    def test_multiple_attention_operations_with_tensors(self):
        """Test multiple attention operations in sequence with Tensor interface."""
        seq_len, d_model = 4, 8

        # Create initial tensor
        x = Tensor(np.random.randn(seq_len, d_model))
        current_data = x.data

        # Apply multiple attention operations
        attn1 = SelfAttention(d_model)
        attn2 = SelfAttention(d_model)
        attn3 = SelfAttention(d_model)

        # Chain operations
        out1, _ = attn1(current_data)
        out2, _ = attn2(out1)
        out3, _ = attn3(out2)

        # Test final conversion to Tensor
        final_tensor = Tensor(out3)

        # Verify chained operations preserve interface compatibility
        assert isinstance(final_tensor, Tensor), "Chained attention should produce valid Tensor"
        assert final_tensor.shape == x.shape, "Chained attention should preserve shape"

    def test_attention_error_handling_with_tensors(self):
        """Test that attention properly handles edge cases with Tensor data."""
        # Test empty tensor compatibility
        empty_tensor = Tensor(np.array([]).reshape(0, 4))

        # Attention should handle empty data gracefully (interface test)
        try:
            self_attn = SelfAttention(4)
            # This might fail, but it should fail gracefully with clear error
            output, weights = self_attn(empty_tensor.data)
        except (ValueError, IndexError) as e:
            # Expected behavior - should fail with clear error message
            assert isinstance(e, (ValueError, IndexError)), "Should fail gracefully with empty data"

        # Test single sequence element
        single_seq = Tensor(np.random.randn(1, 8))
        self_attn = SelfAttention(8)
        output, weights = self_attn(single_seq.data)

        # Should handle single sequence
        assert output.shape == (1, 8), "Should handle single sequence"
        assert weights.shape == (1, 1), "Should produce 1x1 attention weights"


if __name__ == "__main__":
    pytest.main([__file__])
