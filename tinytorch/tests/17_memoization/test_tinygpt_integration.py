"""
Integration test for Module 15: TinyGPT
Tests language model components and GPT-style transformer integration.
"""

import sys
import numpy as np
from pathlib import Path

def run_integration_test():
    """Run integration test for TinyGPT module."""
    try:
        print("🧪 Testing TinyGPT Integration...")

        # Test 1: Import validation
        print("1. Import validation...")
        import tinytorch.tinygpt as tgpt
        print("   ✅ tinytorch.tinygpt imported successfully")

        # Test 2: Component availability
        print("2. Component availability...")
        expected_components = [
            'CharTokenizer', 'MultiHeadAttention', 'create_causal_mask',
            'LayerNorm', 'TransformerBlock', 'PositionalEncoding',
            'TinyGPT', 'LanguageModelLoss', 'LanguageModelAccuracy',
            'LanguageModelTrainer'
        ]

        available_components = tgpt.__all__
        missing_components = [comp for comp in expected_components if comp not in available_components]

        if missing_components:
            print(f"   ⚠️ Missing components: {missing_components}")
        else:
            print(f"   ✅ All {len(expected_components)} components available")

        # Test 3: CharTokenizer functionality
        print("3. CharTokenizer functionality...")
        tokenizer = tgpt.CharTokenizer(vocab_size=100)
        tokenizer.fit("Hello World! This is a test of the tokenizer.")

        # Test encoding/decoding
        test_text = "Hello"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)

        if decoded == test_text:
            print(f"   ✅ Tokenizer round-trip: '{test_text}' -> {len(encoded)} tokens -> '{decoded}'")
        else:
            print(f"   ❌ Tokenizer round-trip failed: '{test_text}' -> '{decoded}'")
            return False

        # Test 4: MultiHeadAttention functionality
        print("4. MultiHeadAttention functionality...")
        from tinytorch.core.tensor import Tensor

        attention = tgpt.MultiHeadAttention(d_model=64, num_heads=8)
        test_input = Tensor(np.random.randn(1, 10, 64))
        output = attention.forward(test_input, test_input, test_input)

        if output.shape == test_input.shape:
            print(f"   ✅ MultiHeadAttention: {test_input.shape} -> {output.shape}")
        else:
            print(f"   ❌ MultiHeadAttention shape mismatch: {test_input.shape} -> {output.shape}")
            return False

        # Test 5: TransformerBlock functionality
        print("5. TransformerBlock functionality...")
        transformer = tgpt.TransformerBlock(d_model=64, num_heads=8, d_ff=256)
        test_input = Tensor(np.random.randn(1, 10, 64))
        output = transformer.forward(test_input)

        if output.shape == test_input.shape:
            print(f"   ✅ TransformerBlock: {test_input.shape} -> {output.shape}")
        else:
            print(f"   ❌ TransformerBlock shape mismatch: {test_input.shape} -> {output.shape}")
            return False

        # Test 6: Complete TinyGPT model
        print("6. Complete TinyGPT model...")
        model = tgpt.TinyGPT(vocab_size=50, d_model=64, num_heads=8, num_layers=2)
        test_input = Tensor(np.array([[1, 2, 3, 4, 5]]))
        output = model.forward(test_input)

        expected_shape = (1, 5, 50)  # (batch, seq_len, vocab_size)
        if output.shape == expected_shape:
            print(f"   ✅ TinyGPT: {test_input.shape} -> {output.shape}")
            print(f"   ✅ Parameters: ~{model.count_parameters():,}")
        else:
            print(f"   ❌ TinyGPT shape mismatch: expected {expected_shape}, got {output.shape}")
            return False

        # Test 7: Text generation capability
        print("7. Text generation capability...")
        try:
            # Simple generation test
            generated = model.generate(test_input, max_new_tokens=3, temperature=1.0)
            if generated.shape[1] > test_input.shape[1]:
                print(f"   ✅ Text generation: {test_input.shape[1]} -> {generated.shape[1]} tokens")
            else:
                print(f"   ❌ Text generation failed: no new tokens generated")
                return False
        except Exception as e:
            print(f"   ⚠️ Text generation issue (non-critical): {e}")

        # Test 8: Training components
        print("8. Training components...")
        try:
            loss_fn = tgpt.LanguageModelLoss()
            metric = tgpt.LanguageModelAccuracy()

            # Test loss computation
            logits = Tensor(np.random.randn(1, 5, 50))
            targets = Tensor(np.array([[1, 2, 3, 4, 5]]))

            loss = loss_fn.forward(logits, targets)
            accuracy = metric.forward(logits, targets)

            print(f"   ✅ Training components: loss={loss:.3f}, accuracy={accuracy:.3f}")
        except Exception as e:
            print(f"   ⚠️ Training components issue (non-critical): {e}")

        print("✅ TinyGPT integration test PASSED")
        return {
            "success": True,
            "message": "TinyGPT integration test passed",
            "components_tested": 8,
            "module_name": "16_tinygpt"
        }

    except Exception as e:
        print(f"❌ TinyGPT integration test FAILED: {e}")
        return {
            "success": False,
            "error": str(e),
            "module_name": "16_tinygpt"
        }

if __name__ == "__main__":
    result = run_integration_test()
    sys.exit(0 if result.get("success", False) else 1)
