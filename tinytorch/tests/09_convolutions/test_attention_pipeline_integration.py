"""
Integration Tests - Attention Pipeline

Tests cross-module pipeline interfaces and compatibility.
Focuses on how attention integrates with other TinyTorch modules to build complete workflows.
"""

import pytest
import numpy as np
from test_utils import setup_integration_test

# Ensure proper setup before importing
setup_integration_test()

# Import ONLY from TinyTorch package
from tinytorch.core.tensor import Tensor
from tinytorch.core.attention import scaled_dot_product_attention
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Softmax
from tinytorch.core.layers import Sequential

# Try to import optional attention components (may not exist yet)
try:
    from tinytorch.core.attention import SelfAttention, create_causal_mask
    HAS_SELF_ATTENTION = True
except ImportError:
    HAS_SELF_ATTENTION = False
    SelfAttention = None
    create_causal_mask = None

# Skip this entire module if SelfAttention is not available
pytestmark = pytest.mark.skipif(
    not HAS_SELF_ATTENTION,
    reason="SelfAttention and create_causal_mask not yet implemented"
)


class TestAttentionDensePipelineInterface:
    """Test interface compatibility between Attention and Dense modules."""

    def test_attention_output_to_dense_input(self):
        """Test that attention output can be used as Dense layer input."""
        seq_len, d_model = 6, 16

        # Create attention and dense components
        self_attn = SelfAttention(d_model)
        dense = Linear(input_size=d_model, output_size=10)

        # Create input
        x = Tensor(np.random.randn(seq_len, d_model))

        # Test pipeline interface: Attention → Dense
        attn_output, _ = self_attn(x.data)

        # Test that attention output can feed into dense layer
        for i in range(seq_len):
            pos_input = Tensor(attn_output[i:i+1])  # Single position
            dense_output = dense(pos_input)

            # Verify interface compatibility
            assert isinstance(dense_output, Tensor), "Dense should accept attention output as Tensor"
            assert dense_output.shape == (1, 10), "Dense should process attention output correctly"

    def test_attention_sequential_compatibility(self):
        """Test that attention can be integrated into Sequential pipelines."""
        d_model = 8

        # Test if we can build: Tensor → Dense → Attention-style processing
        input_tensor = Tensor(np.random.randn(4, 6))

        # Step 1: Dense layer to project to d_model
        projection = Linear(input_size=6, output_size=d_model)
        projected = projection(input_tensor)

        # Step 2: Attention processing (simulating attention in pipeline)
        self_attn = SelfAttention(d_model)
        attn_output, _ = self_attn(projected.data)

        # Step 3: Back to Dense layer
        output_projection = Linear(input_size=d_model, output_size=3)
        final_outputs = []
        for i in range(4):
            pos_input = Tensor(attn_output[i:i+1])
            pos_output = output_projection(pos_input)
            final_outputs.append(pos_output.data)

        final_result = np.concatenate(final_outputs, axis=0)

        # Verify pipeline interface works
        assert final_result.shape == (4, 3), "Complete pipeline should work"
        assert not np.any(np.isnan(final_result)), "Pipeline should produce valid outputs"

    def test_attention_with_activation_integration(self):
        """Test attention integration with activation functions."""
        seq_len, d_model = 5, 12

        # Create components
        self_attn = SelfAttention(d_model)
        relu = ReLU()
        dense = Linear(input_size=d_model, output_size=d_model)

        # Test pipeline: Input → Attention → Activation → Dense
        x = Tensor(np.random.randn(seq_len, d_model))

        # Attention step
        attn_output, _ = self_attn(x.data)

        # Process each position through activation and dense
        for i in range(seq_len):
            # Attention → Tensor → Activation → Dense pipeline
            pos_tensor = Tensor(attn_output[i:i+1])
            activated = relu(pos_tensor)
            dense_output = dense(activated)

            # Verify cross-module interface
            assert isinstance(activated, Tensor), "Activation should work with attention output"
            assert isinstance(dense_output, Tensor), "Dense should work after activation"
            assert dense_output.shape == (1, d_model), "Pipeline should preserve expected shapes"


class TestAttentionMultiModuleWorkflows:
    """Test attention in multi-module workflows and architectures."""

    def test_encoder_decoder_interface_pattern(self):
        """Test encoder-decoder pattern using multiple TinyTorch modules."""
        src_len, tgt_len, d_model = 6, 4, 16

        # Source processing (encoder-style)
        src = Tensor(np.random.randn(src_len, d_model))
        src_projection = Linear(input_size=d_model, output_size=d_model)
        src_projected = src_projection(src)

        encoder_attn = SelfAttention(d_model)
        encoded, _ = encoder_attn(src_projected.data)

        # Target processing (decoder-style)
        tgt = Tensor(np.random.randn(tgt_len, d_model))
        tgt_projection = Linear(input_size=d_model, output_size=d_model)
        tgt_projected = tgt_projection(tgt)

        # Cross-attention interface test
        cross_output, _ = scaled_dot_product_attention(
            tgt_projected.data,  # Queries from target
            encoded,            # Keys from encoder
            encoded             # Values from encoder
        )

        # Final processing
        output_projection = Linear(input_size=d_model, output_size=10)
        final_outputs = []
        for i in range(tgt_len):
            pos_input = Tensor(cross_output[i:i+1])
            pos_output = output_projection(pos_input)
            final_outputs.append(pos_output.data)

        final_result = np.concatenate(final_outputs, axis=0)

        # Verify multi-module workflow
        assert final_result.shape == (tgt_len, 10), "Encoder-decoder workflow should work"
        assert not np.any(np.isnan(final_result)), "Multi-module workflow should be stable"

    def test_multi_layer_attention_with_residuals(self):
        """Test multi-layer attention with residual connections using multiple modules."""
        seq_len, d_model = 8, 20
        num_layers = 3

        # Initial processing
        x = Tensor(np.random.randn(seq_len, d_model))
        embedding_projection = Linear(input_size=d_model, output_size=d_model)
        current_repr = embedding_projection(x).data

        # Multi-layer processing with residuals
        for layer in range(num_layers):
            # Self-attention
            attn = SelfAttention(d_model)
            attn_output, _ = attn(current_repr)

            # Feedforward network (using Dense layers)
            ff_network = Sequential([
                Linear(input_size=d_model, output_size=d_model * 2),
                ReLU(),
                Linear(input_size=d_model * 2, output_size=d_model)
            ])

            # Process each position through feedforward
            ff_outputs = []
            for i in range(seq_len):
                pos_input = Tensor(attn_output[i:i+1])
                pos_output = ff_network(pos_input)
                ff_outputs.append(pos_output.data)

            ff_result = np.concatenate(ff_outputs, axis=0)

            # Residual connection (attention + feedforward)
            current_repr = attn_output + ff_result

        # Verify multi-layer integration
        assert current_repr.shape == (seq_len, d_model), "Multi-layer should preserve shape"
        assert not np.any(np.isnan(current_repr)), "Multi-layer integration should be stable"

    def test_attention_classification_pipeline(self):
        """Test attention in classification pipeline with multiple modules."""
        seq_len, d_model, num_classes = 10, 24, 5

        # Input processing
        sentence = Tensor(np.random.randn(seq_len, d_model))
        input_projection = Linear(input_size=d_model, output_size=d_model)
        projected_input = input_projection(sentence)

        # Attention processing
        self_attn = SelfAttention(d_model)
        attended_seq, _ = self_attn(projected_input.data)

        # Global pooling (sequence → single representation)
        pooled_repr = np.mean(attended_seq, axis=0, keepdims=True)

        # Classification head (using Sequential)
        classifier = Sequential([
            Linear(input_size=d_model, output_size=d_model // 2),
            ReLU(),
            Linear(input_size=d_model // 2, output_size=num_classes)
        ])

        # Final classification
        pooled_tensor = Tensor(pooled_repr)
        class_scores = classifier(pooled_tensor)

        # Verify classification pipeline
        assert class_scores.shape == (1, num_classes), "Classification pipeline should work"
        assert isinstance(class_scores, Tensor), "Pipeline should produce Tensor output"


class TestAttentionDataFlowCompatibility:
    """Test data flow compatibility between attention and other modules."""

    def test_shape_preservation_across_modules(self):
        """Test that shapes flow correctly between attention and other modules."""
        batch_configs = [
            (4, 8),    # Small sequence
            (16, 32),  # Medium sequence
            (8, 64),   # Large model dimension
        ]

        for seq_len, d_model in batch_configs:
            # Input
            x = Tensor(np.random.randn(seq_len, d_model))

            # Processing pipeline
            input_proj = Linear(input_size=d_model, output_size=d_model)
            projected = input_proj(x)

            attn = SelfAttention(d_model)
            attn_out, _ = attn(projected.data)

            output_proj = Linear(input_size=d_model, output_size=d_model // 2)

            # Test shape flow
            for i in range(seq_len):
                pos_tensor = Tensor(attn_out[i:i+1])
                final_out = output_proj(pos_tensor)

                # Verify shape compatibility
                assert final_out.shape == (1, d_model // 2), f"Shape flow failed for config {(seq_len, d_model)}"

    def test_dtype_preservation_across_modules(self):
        """Test that data types are preserved across attention and other modules."""
        seq_len, d_model = 6, 16

        # Test float32 flow
        x_f32 = Tensor(np.random.randn(seq_len, d_model).astype(np.float32))

        dense_f32 = Linear(input_size=d_model, output_size=d_model)
        projected_f32 = dense_f32(x_f32)

        attn_f32 = SelfAttention(d_model)
        attn_out_f32, _ = attn_f32(projected_f32.data)

        # Verify dtype flow
        assert projected_f32.dtype == np.float32, "Dense should preserve float32"
        assert attn_out_f32.dtype == np.float32, "Attention should preserve float32"

        # Test conversion back to Tensor
        result_tensor_f32 = Tensor(attn_out_f32)
        assert result_tensor_f32.dtype == np.float32, "Tensor creation should preserve float32"

    def test_error_handling_across_modules(self):
        """Test error handling when modules are incompatibly connected."""
        # Test dimension mismatch between attention and dense
        seq_len = 4
        attn_dim = 8
        dense_dim = 16  # Intentional mismatch

        x = Tensor(np.random.randn(seq_len, attn_dim))
        attn = SelfAttention(attn_dim)
        attn_out, _ = attn(x.data)

        # This should fail gracefully
        incompatible_dense = Linear(input_size=dense_dim, output_size=10)

        try:
            pos_tensor = Tensor(attn_out[0:1])  # Shape (1, 8)
            result = incompatible_dense(pos_tensor)  # Expects (1, 16)
            assert False, "Should have failed with dimension mismatch"
        except (ValueError, AssertionError, TypeError) as e:
            # Expected behavior - should fail with clear error
            assert isinstance(e, (ValueError, AssertionError, TypeError)), "Should fail gracefully with incompatible dimensions"


class TestAttentionSystemLevelIntegration:
    """Test system-level integration scenarios."""

    def test_complete_transformer_block_simulation(self):
        """Test simulation of complete transformer block using TinyTorch modules."""
        seq_len, d_model = 8, 32

        # Input
        x = Tensor(np.random.randn(seq_len, d_model))

        # Transformer block simulation
        # 1. Self-attention
        self_attn = SelfAttention(d_model)
        attn_out, _ = self_attn(x.data)

        # 2. Residual connection (attention + input)
        attn_residual = attn_out + x.data

        # 3. Feedforward network
        ff_net = Sequential([
            Linear(input_size=d_model, output_size=d_model * 4),
            ReLU(),
            Linear(input_size=d_model * 4, output_size=d_model)
        ])

        # Process each position through feedforward
        ff_outputs = []
        for i in range(seq_len):
            pos_input = Tensor(attn_residual[i:i+1])
            pos_output = ff_net(pos_input)
            ff_outputs.append(pos_output.data)

        ff_result = np.concatenate(ff_outputs, axis=0)

        # 4. Second residual connection
        final_output = attn_residual + ff_result

        # Verify complete transformer block simulation
        assert final_output.shape == (seq_len, d_model), "Transformer block should preserve shape"
        assert not np.any(np.isnan(final_output)), "Transformer block should be stable"

        # Test that output can be used for next layer
        next_attn = SelfAttention(d_model)
        next_out, _ = next_attn(final_output)
        assert next_out.shape == (seq_len, d_model), "Should be stackable"

    def test_modular_component_replacement(self):
        """Test that attention components can be replaced modularly."""
        seq_len, d_model = 6, 16

        x = Tensor(np.random.randn(seq_len, d_model))

        # Pipeline with different attention configurations
        attention_variants = [
            SelfAttention(d_model),
            SelfAttention(d_model),  # Different instance
            SelfAttention(d_model),  # Another instance
        ]

        dense_postprocess = Linear(input_size=d_model, output_size=8)

        # Test that all variants work in same pipeline
        for i, attn_variant in enumerate(attention_variants):
            attn_out, _ = attn_variant(x.data)

            # Process first position
            pos_tensor = Tensor(attn_out[0:1])
            result = dense_postprocess(pos_tensor)

            # Verify modular replacement works
            assert result.shape == (1, 8), f"Attention variant {i} should work in pipeline"
            assert isinstance(result, Tensor), f"Attention variant {i} should produce Tensor output"


if __name__ == "__main__":
    pytest.main([__file__])
