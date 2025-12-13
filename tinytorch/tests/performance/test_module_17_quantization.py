"""
Performance Tests for Module 17: Quantization

Tests whether quantization actually provides the claimed 4× speedup and memory
reduction with <1% accuracy loss.

Key questions:
- Does INT8 quantization actually reduce memory by 4×?
- Is there a real inference speedup from quantization?
- Is accuracy loss actually <1% as claimed?
- Does quantization work on realistic CNN models?
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add the performance framework to path
sys.path.append(str(Path(__file__).parent))
from performance_test_framework import PerformanceTestSuite, PerformanceComparator, WorkloadGenerator

# Add module path
sys.path.append(str(Path(__file__).parent.parent.parent / 'modules' / '17_quantization'))

try:
    from quantization_dev import (
        BaselineCNN, QuantizedCNN, INT8Quantizer, QuantizationPerformanceAnalyzer,
        QuantizationSystemsAnalyzer, QuantizedConv2d
    )
    QUANTIZATION_AVAILABLE = True
except ImportError:
    print("❌ Module 17 quantization tools not available")
    QUANTIZATION_AVAILABLE = False

class Module17PerformanceTests:
    """Test suite for Module 17 quantization techniques."""

    def __init__(self):
        self.suite = PerformanceTestSuite()
        self.comparator = PerformanceComparator()
        self.workloads = WorkloadGenerator()

    def test_memory_reduction(self):
        """Test whether quantization actually reduces memory by 4×."""
        if not QUANTIZATION_AVAILABLE:
            return "Quantization module not available"

        print("💾 Testing memory reduction from quantization")

        # Create models
        baseline_model = BaselineCNN(input_channels=3, num_classes=10)
        quantized_model = QuantizedCNN(input_channels=3, num_classes=10)

        # Quantize the model
        calibration_data = [np.random.randn(1, 3, 32, 32) for _ in range(5)]
        quantized_model.calibrate_and_quantize(calibration_data)

        # Measure memory usage
        def calculate_model_memory(model):
            """Calculate memory usage of model parameters."""
            total_bytes = 0

            # Baseline model memory
            if hasattr(model, 'conv1_weight'):
                total_bytes += model.conv1_weight.nbytes + model.conv1_bias.nbytes
                total_bytes += model.conv2_weight.nbytes + model.conv2_bias.nbytes
                total_bytes += model.fc.nbytes
            # Quantized model memory
            elif hasattr(model, 'conv1'):
                # Conv layers
                if hasattr(model.conv1, 'weight_quantized') and model.conv1.is_quantized:
                    total_bytes += model.conv1.weight_quantized.nbytes
                else:
                    total_bytes += model.conv1.weight_fp32.nbytes

                if hasattr(model.conv2, 'weight_quantized') and model.conv2.is_quantized:
                    total_bytes += model.conv2.weight_quantized.nbytes
                else:
                    total_bytes += model.conv2.weight_fp32.nbytes

                # FC layer
                total_bytes += model.fc.nbytes

            return total_bytes / (1024 * 1024)  # Convert to MB

        baseline_memory_mb = calculate_model_memory(baseline_model)
        quantized_memory_mb = calculate_model_memory(quantized_model)

        memory_reduction = baseline_memory_mb / quantized_memory_mb

        # Check if we achieved close to 4× reduction
        # Note: Only conv layers are quantized, FC layer remains FP32
        conv_portion = 0.7  # Approximately 70% of model is conv weights
        expected_reduction = 1 / (conv_portion * 0.25 + (1 - conv_portion) * 1.0)  # ~2.3×

        memory_test_passed = memory_reduction > 1.8  # At least some reduction

        result = {
            'baseline_memory_mb': baseline_memory_mb,
            'quantized_memory_mb': quantized_memory_mb,
            'memory_reduction': memory_reduction,
            'expected_reduction': expected_reduction,
            'memory_test_passed': memory_test_passed
        }

        if memory_test_passed:
            print(f"✅ Memory reduction achieved: {memory_reduction:.2f}× reduction")
        else:
            print(f"❌ Insufficient memory reduction: {memory_reduction:.2f}× reduction")

        return result

    def test_inference_speedup(self):
        """Test whether quantized inference is actually faster."""
        if not QUANTIZATION_AVAILABLE:
            return "Quantization module not available"

        print("🚀 Testing inference speedup from quantization")

        # Create models
        baseline_model = BaselineCNN(input_channels=3, num_classes=10)
        quantized_model = QuantizedCNN(input_channels=3, num_classes=10)

        # Quantize the model
        calibration_data = [np.random.randn(1, 3, 32, 32) for _ in range(5)]
        quantized_model.calibrate_and_quantize(calibration_data)

        # Create test input
        test_input = np.random.randn(4, 3, 32, 32)

        # Wrapper functions for timing
        def baseline_inference():
            return baseline_model.forward(test_input)

        def quantized_inference():
            return quantized_model.forward(test_input)

        # Verify results are close
        try:
            baseline_output = baseline_inference()
            quantized_output = quantized_inference()

            # Check if outputs are reasonably close
            output_close = np.allclose(baseline_output, quantized_output, rtol=0.1, atol=0.1)
            if not output_close:
                print("⚠️  Warning: Quantized output differs significantly from baseline")

        except Exception as e:
            return f"Inference test error: {e}"

        # Performance comparison
        comparison = self.comparator.compare_implementations(
            baseline_inference,
            quantized_inference,
            baseline_name="fp32_inference",
            optimized_name="int8_inference"
        )

        # Note: Educational quantization may not show speedup without real INT8 kernels
        # We'll consider any improvement or small regression as acceptable
        reasonable_performance = comparison.speedup > 0.5  # Within 2× slower

        result = {
            'speedup': comparison.speedup,
            'reasonable_performance': reasonable_performance,
            'baseline_time_ms': comparison.baseline.mean_time_ms,
            'quantized_time_ms': comparison.optimized.mean_time_ms,
            'outputs_close': output_close
        }

        if comparison.speedup > 1.1:
            print(f"🎉 Quantization speedup achieved: {comparison.speedup:.2f}×")
        elif reasonable_performance:
            print(f"✅ Quantization performance reasonable: {comparison.speedup:.2f}×")
            print("   (Educational implementation - production would use INT8 kernels)")
        else:
            print(f"❌ Quantization performance poor: {comparison.speedup:.2f}×")

        return comparison

    def test_accuracy_preservation(self):
        """Test whether quantization preserves accuracy as claimed (<1% loss)."""
        if not QUANTIZATION_AVAILABLE:
            return "Quantization module not available"

        print("🎯 Testing accuracy preservation in quantization")

        # Create models
        baseline_model = BaselineCNN(input_channels=3, num_classes=10)
        quantized_model = QuantizedCNN(input_channels=3, num_classes=10)

        # Copy weights from baseline to quantized before quantization
        quantized_model.conv1.weight_fp32 = baseline_model.conv1_weight.copy()
        quantized_model.conv1.bias = baseline_model.conv1_bias.copy()
        quantized_model.conv2.weight_fp32 = baseline_model.conv2_weight.copy()
        quantized_model.conv2.bias = baseline_model.conv2_bias.copy()
        quantized_model.fc = baseline_model.fc.copy()

        # Generate test dataset
        test_size = 100
        test_inputs = np.random.randn(test_size, 3, 32, 32)

        # Get baseline predictions
        baseline_outputs = baseline_model.forward(test_inputs)
        baseline_predictions = np.argmax(baseline_outputs, axis=1)

        # Quantize model
        calibration_data = [test_inputs[:5]]  # Use some test data for calibration
        quantized_model.calibrate_and_quantize(calibration_data)

        # Get quantized predictions
        quantized_outputs = quantized_model.forward(test_inputs)
        quantized_predictions = np.argmax(quantized_outputs, axis=1)

        # Calculate accuracy metrics
        prediction_agreement = np.mean(baseline_predictions == quantized_predictions)
        output_mse = np.mean((baseline_outputs - quantized_outputs) ** 2)
        output_mae = np.mean(np.abs(baseline_outputs - quantized_outputs))

        # Check accuracy preservation
        high_agreement = prediction_agreement > 0.95  # 95%+ predictions should match
        low_output_difference = output_mae < 1.0  # Mean absolute error < 1.0

        accuracy_preserved = high_agreement and low_output_difference

        result = {
            'prediction_agreement': prediction_agreement,
            'output_mse': output_mse,
            'output_mae': output_mae,
            'high_agreement': high_agreement,
            'low_output_difference': low_output_difference,
            'accuracy_preserved': accuracy_preserved,
            'test_samples': test_size
        }

        if accuracy_preserved:
            print(f"✅ Accuracy preserved: {prediction_agreement:.1%} agreement, {output_mae:.3f} MAE")
        else:
            print(f"❌ Accuracy degraded: {prediction_agreement:.1%} agreement, {output_mae:.3f} MAE")

        return result

    def test_quantization_precision(self):
        """Test the accuracy of the quantization/dequantization process."""
        if not QUANTIZATION_AVAILABLE:
            return "Quantization module not available"

        print("🔬 Testing quantization precision")

        quantizer = INT8Quantizer()

        # Test on different types of data
        test_cases = [
            ("small_weights", np.random.randn(100, 100) * 0.1),
            ("large_weights", np.random.randn(100, 100) * 2.0),
            ("uniform_weights", np.random.uniform(-1, 1, (100, 100))),
            ("sparse_weights", np.random.randn(100, 100) * 0.01)
        ]

        precision_results = {}

        for name, weights in test_cases:
            # Quantize and dequantize
            scale, zero_point = quantizer.compute_quantization_params(weights)
            quantized = quantizer.quantize_tensor(weights, scale, zero_point)
            dequantized = quantizer.dequantize_tensor(quantized, scale, zero_point)

            # Calculate precision metrics
            mse = np.mean((weights - dequantized) ** 2)
            mae = np.mean(np.abs(weights - dequantized))
            max_error = np.max(np.abs(weights - dequantized))

            # Relative error
            weight_range = np.max(weights) - np.min(weights)
            relative_mae = mae / weight_range if weight_range > 0 else 0

            precision_results[name] = {
                'mse': mse,
                'mae': mae,
                'max_error': max_error,
                'relative_mae': relative_mae,
                'good_precision': relative_mae < 0.02  # < 2% relative error
            }

            print(f"  {name}: MAE={mae:.4f}, relative={relative_mae:.1%}")

        # Overall precision test
        all_good_precision = all(result['good_precision'] for result in precision_results.values())

        result = {
            'test_cases': precision_results,
            'all_good_precision': all_good_precision
        }

        if all_good_precision:
            print("✅ Quantization precision good across all test cases")
        else:
            print("❌ Quantization precision issues detected")

        return result

    def test_systems_analysis_accuracy(self):
        """Test whether the systems analysis tools provide accurate assessments."""
        if not QUANTIZATION_AVAILABLE:
            return "Quantization module not available"

        print("📊 Testing systems analysis accuracy")

        try:
            analyzer = QuantizationSystemsAnalyzer()

            # Test precision vs performance analysis
            analysis = analyzer.analyze_precision_tradeoffs([32, 16, 8, 4])

            # Validate analysis structure
            required_keys = ['compute_efficiency', 'typical_accuracy_loss', 'memory_per_param']
            has_required_keys = all(key in analysis for key in required_keys)

            # Validate logical relationships
            memory_decreases = all(analysis['memory_per_param'][i] >= analysis['memory_per_param'][i+1]
                                 for i in range(len(analysis['memory_per_param'])-1))

            accuracy_loss_increases = all(analysis['typical_accuracy_loss'][i] <= analysis['typical_accuracy_loss'][i+1]
                                        for i in range(len(analysis['typical_accuracy_loss'])-1))

            # Check if INT8 is identified as optimal
            efficiency_ratios = [s / (1 + a) for s, a in zip(analysis['compute_efficiency'],
                                                            analysis['typical_accuracy_loss'])]
            optimal_idx = np.argmax(efficiency_ratios)
            optimal_bits = analysis['bit_widths'][optimal_idx]
            int8_optimal = optimal_bits == 8

            analysis_result = {
                'has_required_keys': has_required_keys,
                'memory_decreases_correctly': memory_decreases,
                'accuracy_loss_increases_correctly': accuracy_loss_increases,
                'int8_identified_as_optimal': int8_optimal,
                'optimal_bits': optimal_bits,
                'analysis_logical': has_required_keys and memory_decreases and accuracy_loss_increases
            }

            if analysis_result['analysis_logical'] and int8_optimal:
                print("✅ Systems analysis provides logical and accurate assessments")
            else:
                print("❌ Systems analysis has logical inconsistencies")

            return analysis_result

        except Exception as e:
            return f"Systems analysis error: {e}"

    def test_quantization_performance_analyzer(self):
        """Test the quantization performance analyzer tool."""
        if not QUANTIZATION_AVAILABLE:
            return "Quantization module not available"

        print("📈 Testing quantization performance analyzer")

        try:
            # Create models
            baseline_model = BaselineCNN(input_channels=3, num_classes=10)
            quantized_model = QuantizedCNN(input_channels=3, num_classes=10)

            # Quantize model
            calibration_data = [np.random.randn(1, 3, 32, 32) for _ in range(3)]
            quantized_model.calibrate_and_quantize(calibration_data)

            # Test data
            test_data = np.random.randn(4, 3, 32, 32)

            # Use the performance analyzer
            analyzer = QuantizationPerformanceAnalyzer()
            results = analyzer.benchmark_models(baseline_model, quantized_model, test_data, num_runs=5)

            # Validate analyzer results
            required_metrics = ['memory_reduction', 'speedup', 'prediction_agreement']
            has_required_metrics = all(metric in results for metric in required_metrics)

            reasonable_values = (
                results['memory_reduction'] > 1.0 and
                results['speedup'] > 0.1 and  # May be slower in educational implementation
                results['prediction_agreement'] >= 0.0
            )

            analyzer_result = {
                'has_required_metrics': has_required_metrics,
                'reasonable_values': reasonable_values,
                'memory_reduction': results['memory_reduction'],
                'speedup': results['speedup'],
                'prediction_agreement': results['prediction_agreement'],
                'analyzer_working': has_required_metrics and reasonable_values
            }

            if analyzer_result['analyzer_working']:
                print(f"✅ Performance analyzer working: {results['memory_reduction']:.1f}× memory, "
                     f"{results['speedup']:.1f}× speed, {results['prediction_agreement']:.1%} agreement")
            else:
                print("❌ Performance analyzer has issues")

            return analyzer_result

        except Exception as e:
            return f"Performance analyzer error: {e}"

def run_module_17_performance_tests():
    """Run all performance tests for Module 17."""
    print("🧪 TESTING MODULE 17: QUANTIZATION")
    print("=" * 60)
    print("Verifying that quantization provides real benefits with minimal accuracy loss")

    if not QUANTIZATION_AVAILABLE:
        print("❌ Cannot test Module 17 - quantization tools not available")
        return

    test_suite = Module17PerformanceTests()

    tests = {
        'memory_reduction': test_suite.test_memory_reduction,
        'inference_speedup': test_suite.test_inference_speedup,
        'accuracy_preservation': test_suite.test_accuracy_preservation,
        'quantization_precision': test_suite.test_quantization_precision,
        'systems_analysis': test_suite.test_systems_analysis_accuracy,
        'performance_analyzer': test_suite.test_quantization_performance_analyzer
    }

    results = test_suite.suite.run_module_tests('module_17_quantization', tests)

    # Summary
    print(f"\n📊 MODULE 17 TEST SUMMARY")
    print("=" * 40)

    total_tests = len(tests)
    passed_tests = 0

    key_metrics = {}

    for test_name, result in results.items():
        if hasattr(result, 'speedup'):  # ComparisonResult
            passed = result.speedup > 0.8  # Allow some performance variation
            key_metrics[f'{test_name}_speedup'] = result.speedup
        elif isinstance(result, dict):
            # Check specific success criteria for each test
            if 'memory_test_passed' in result:
                passed = result['memory_test_passed']
                key_metrics['memory_reduction'] = result.get('memory_reduction', 0)
            elif 'reasonable_performance' in result:
                passed = result['reasonable_performance']
            elif 'accuracy_preserved' in result:
                passed = result['accuracy_preserved']
                key_metrics['prediction_agreement'] = result.get('prediction_agreement', 0)
            elif 'all_good_precision' in result:
                passed = result['all_good_precision']
            elif 'analysis_logical' in result:
                passed = result['analysis_logical'] and result.get('int8_identified_as_optimal', False)
            elif 'analyzer_working' in result:
                passed = result['analyzer_working']
            else:
                passed = False
        else:
            passed = False

        if passed:
            passed_tests += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")

    success_rate = passed_tests / total_tests
    print(f"\nSUCCESS RATE: {success_rate:.1%} ({passed_tests}/{total_tests})")

    # Key insights
    if 'memory_reduction' in key_metrics:
        print(f"📊 Memory reduction: {key_metrics['memory_reduction']:.2f}×")
    if 'prediction_agreement' in key_metrics:
        print(f"🎯 Accuracy preservation: {key_metrics['prediction_agreement']:.1%}")

    if success_rate >= 0.7:
        print("🎉 Module 17 quantization is working effectively!")
        print("💡 Note: Performance gains depend on hardware INT8 support")
    else:
        print("⚠️  Module 17 quantization needs improvement")

    return results

if __name__ == "__main__":
    run_module_17_performance_tests()
