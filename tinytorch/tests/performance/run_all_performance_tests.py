"""
Comprehensive Performance Validation for TinyTorch Optimization Modules

This script runs all performance tests across modules 15-20 and generates
a complete validation report with actual measurements.

The goal is to provide honest, scientific assessment of whether each
optimization module actually delivers the claimed benefits.
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import traceback

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import all test modules
try:
    from test_module_15_profiling import run_module_15_performance_tests
    from test_module_16_acceleration import run_module_16_performance_tests
    from test_module_17_quantization import run_module_17_performance_tests
    from test_module_19_caching import run_module_19_performance_tests
    from test_module_20_benchmarking import run_module_20_performance_tests
    from performance_test_framework import PerformanceTestSuite
except ImportError as e:
    print(f"❌ Error importing test modules: {e}")
    sys.exit(1)

class TinyTorchPerformanceValidator:
    """
    Comprehensive validator for TinyTorch optimization modules.

    Runs scientific performance tests across all optimization modules
    and generates detailed reports with actual measurements.
    """

    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.test_suite = PerformanceTestSuite("validation_results")

    def run_all_tests(self):
        """Run performance tests for all optimization modules."""
        print("🧪 TINYTORCH OPTIMIZATION MODULES - PERFORMANCE VALIDATION")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("This validation tests whether optimization modules actually deliver")
        print("their claimed performance improvements with real measurements.")
        print()

        # Define all test modules
        test_modules = [
            ("Module 15: Profiling", run_module_15_performance_tests),
            ("Module 16: Acceleration", run_module_16_performance_tests),
            ("Module 17: Quantization", run_module_17_performance_tests),
            ("Module 19: KV Caching", run_module_19_performance_tests),
            ("Module 20: Benchmarking", run_module_20_benchmarking_tests)
        ]

        # Run each test module
        for module_name, test_function in test_modules:
            print(f"\n{'='*80}")
            print(f"TESTING {module_name.upper()}")
            print('='*80)

            try:
                module_start = time.time()
                results = test_function()
                module_duration = time.time() - module_start

                self.results[module_name] = {
                    'results': results,
                    'duration_seconds': module_duration,
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat()
                }

                print(f"\n✅ {module_name} testing completed in {module_duration:.1f}s")

            except Exception as e:
                error_info = {
                    'status': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat()
                }
                self.results[module_name] = error_info

                print(f"\n❌ {module_name} testing failed: {e}")
                print("Continuing with other modules...")

        total_duration = time.time() - self.start_time
        print(f"\n🏁 All tests completed in {total_duration:.1f}s")

        return self.results

    def analyze_results(self):
        """Analyze results across all modules and generate insights."""
        print(f"\n📊 COMPREHENSIVE ANALYSIS")
        print("=" * 60)

        analysis = {
            'overall_summary': {},
            'module_assessments': {},
            'key_insights': [],
            'recommendations': []
        }

        # Analyze each module
        modules_tested = 0
        modules_successful = 0
        total_speedups = []

        for module_name, module_data in self.results.items():
            if module_data.get('status') == 'error':
                analysis['module_assessments'][module_name] = {
                    'status': 'failed',
                    'assessment': 'Module could not be tested due to errors',
                    'error': module_data.get('error', 'Unknown error')
                }
                continue

            modules_tested += 1
            module_results = module_data.get('results', {})

            # Analyze module performance
            module_analysis = self._analyze_module_performance(module_name, module_results)
            analysis['module_assessments'][module_name] = module_analysis

            if module_analysis.get('overall_success', False):
                modules_successful += 1

            # Collect speedup data
            speedups = module_analysis.get('speedups', [])
            total_speedups.extend(speedups)

        # Overall summary
        success_rate = modules_successful / modules_tested if modules_tested > 0 else 0
        avg_speedup = sum(total_speedups) / len(total_speedups) if total_speedups else 0

        analysis['overall_summary'] = {
            'modules_tested': modules_tested,
            'modules_successful': modules_successful,
            'success_rate': success_rate,
            'average_speedup': avg_speedup,
            'total_speedups_measured': len(total_speedups),
            'best_speedup': max(total_speedups) if total_speedups else 0
        }

        # Generate insights
        analysis['key_insights'] = self._generate_insights(analysis)
        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def _analyze_module_performance(self, module_name, results):
        """Analyze performance results for a specific module."""
        if not results:
            return {'status': 'no_results', 'assessment': 'No test results available'}

        speedups = []
        test_successes = 0
        total_tests = 0
        key_metrics = {}

        for test_name, result in results.items():
            total_tests += 1

            if hasattr(result, 'speedup'):  # ComparisonResult
                speedup = result.speedup
                speedups.append(speedup)

                if speedup > 1.1 and result.is_significant:
                    test_successes += 1

                key_metrics[f'{test_name}_speedup'] = speedup

            elif isinstance(result, dict):
                # Module-specific success criteria
                success = self._determine_test_success(module_name, test_name, result)
                if success:
                    test_successes += 1

                # Extract key metrics
                if 'speedup' in result:
                    speedups.append(result['speedup'])
                if 'memory_reduction' in result:
                    key_metrics[f'{test_name}_memory'] = result['memory_reduction']
                if 'prediction_agreement' in result:
                    key_metrics[f'{test_name}_accuracy'] = result['prediction_agreement']

        success_rate = test_successes / total_tests if total_tests > 0 else 0
        overall_success = success_rate >= 0.6  # 60% threshold

        # Module-specific assessment
        assessment = self._generate_module_assessment(module_name, success_rate, speedups, key_metrics)

        return {
            'total_tests': total_tests,
            'successful_tests': test_successes,
            'success_rate': success_rate,
            'overall_success': overall_success,
            'speedups': speedups,
            'avg_speedup': sum(speedups) / len(speedups) if speedups else 0,
            'max_speedup': max(speedups) if speedups else 0,
            'key_metrics': key_metrics,
            'assessment': assessment
        }

    def _determine_test_success(self, module_name, test_name, result):
        """Determine if a specific test succeeded based on module context."""
        # Module-specific success criteria
        success_keys = {
            'Module 15: Profiling': [
                'timer_accuracy', 'memory_accuracy', 'linear_flop_accuracy',
                'overhead_acceptable', 'has_required_fields', 'results_match'
            ],
            'Module 16: Acceleration': [
                'speedup_achieved', 'dramatic_improvement', 'low_overhead',
                'cache_blocking_effective', 'naive_much_slower'
            ],
            'Module 17: Quantization': [
                'memory_test_passed', 'accuracy_preserved', 'all_good_precision',
                'analysis_logical', 'analyzer_working'
            ],
            'Module 19: KV Caching': [
                'memory_test_passed', 'cache_correctness_passed', 'sequential_speedup_achieved',
                'complexity_improvement_detected', 'cache_performance_good'
            ],
            'Module 20: Benchmarking': [
                'suite_loading_successful', 'reproducible', 'detection_working',
                'fairness_good', 'scaling_measurement_good', 'competition_scoring_working'
            ]
        }

        module_keys = success_keys.get(module_name, [])
        return any(result.get(key, False) for key in module_keys)

    def _generate_module_assessment(self, module_name, success_rate, speedups, metrics):
        """Generate human-readable assessment for each module."""
        if 'Profiling' in module_name:
            if success_rate >= 0.8:
                return f"✅ Profiling tools are accurate and reliable ({success_rate:.1%} success)"
            else:
                return f"⚠️  Profiling tools have accuracy issues ({success_rate:.1%} success)"

        elif 'Acceleration' in module_name:
            max_speedup = max(speedups) if speedups else 0
            if success_rate >= 0.7 and max_speedup > 5:
                return f"🚀 Acceleration delivers dramatic speedups ({max_speedup:.1f}× max speedup)"
            elif success_rate >= 0.5:
                return f"✅ Acceleration shows moderate improvements ({max_speedup:.1f}× max speedup)"
            else:
                return f"❌ Acceleration techniques ineffective ({success_rate:.1%} success)"

        elif 'Quantization' in module_name:
            memory_reduction = metrics.get('memory_reduction_memory', 0)
            accuracy = metrics.get('accuracy_preservation_accuracy', 0)
            if success_rate >= 0.7:
                return f"⚖️  Quantization balances performance and accuracy well ({memory_reduction:.1f}× memory, {accuracy:.1%} accuracy)"
            else:
                return f"⚠️  Quantization has trade-off issues ({success_rate:.1%} success)"

        elif 'Caching' in module_name:
            if success_rate >= 0.6:
                return f"💾 KV caching reduces complexity effectively ({success_rate:.1%} success)"
            else:
                return f"❌ KV caching implementation issues ({success_rate:.1%} success)"

        elif 'Benchmarking' in module_name:
            if success_rate >= 0.8:
                return f"🏆 Benchmarking system is fair and reliable ({success_rate:.1%} success)"
            else:
                return f"⚠️  Benchmarking system needs improvement ({success_rate:.1%} success)"

        else:
            return f"Module tested with {success_rate:.1%} success rate"

    def _generate_insights(self, analysis):
        """Generate key insights from the overall analysis."""
        insights = []

        summary = analysis['overall_summary']

        if summary['success_rate'] >= 0.7:
            insights.append("🎉 Most optimization modules deliver real performance benefits")
        elif summary['success_rate'] >= 0.5:
            insights.append("✅ Some optimization modules work well, others need improvement")
        else:
            insights.append("⚠️  Many optimization modules have significant issues")

        if summary['average_speedup'] > 2.0:
            insights.append(f"🚀 Significant speedups achieved (avg {summary['average_speedup']:.1f}×)")
        elif summary['average_speedup'] > 1.2:
            insights.append(f"📈 Moderate speedups achieved (avg {summary['average_speedup']:.1f}×)")
        else:
            insights.append(f"📉 Limited speedups achieved (avg {summary['average_speedup']:.1f}×)")

        if summary['best_speedup'] > 10:
            insights.append(f"⭐ Some optimizations show dramatic improvement ({summary['best_speedup']:.1f}× best)")

        # Module-specific insights
        for module, assessment in analysis['module_assessments'].items():
            if assessment.get('overall_success') and 'Acceleration' in module:
                insights.append("⚡ Hardware acceleration techniques are particularly effective")
            elif assessment.get('overall_success') and 'Quantization' in module:
                insights.append("⚖️  Quantization successfully balances speed and accuracy")

        return insights

    def _generate_recommendations(self, analysis):
        """Generate recommendations based on test results."""
        recommendations = []

        summary = analysis['overall_summary']

        if summary['success_rate'] < 0.8:
            recommendations.append("🔧 Focus on improving modules with low success rates")

        for module, assessment in analysis['module_assessments'].items():
            if not assessment.get('overall_success'):
                if 'Profiling' in module:
                    recommendations.append("📊 Fix profiling tool accuracy for reliable measurements")
                elif 'Quantization' in module:
                    recommendations.append("⚖️  Address quantization accuracy preservation issues")
                elif 'Caching' in module:
                    recommendations.append("💾 Improve KV caching implementation complexity benefits")

        if summary['average_speedup'] < 1.5:
            recommendations.append("🚀 Focus on optimizations that provide more significant speedups")

        recommendations.append("📈 Consider adding more realistic workloads for better validation")
        recommendations.append("🧪 Implement continuous performance testing to catch regressions")

        return recommendations

    def print_final_report(self, analysis):
        """Print comprehensive final validation report."""
        print(f"\n📋 FINAL VALIDATION REPORT")
        print("=" * 80)

        # Overall summary
        summary = analysis['overall_summary']
        print(f"🎯 OVERALL RESULTS:")
        print(f"   Modules tested: {summary['modules_tested']}")
        print(f"   Success rate: {summary['success_rate']:.1%} ({summary['modules_successful']}/{summary['modules_tested']})")
        print(f"   Average speedup: {summary['average_speedup']:.2f}×")
        print(f"   Best speedup: {summary['best_speedup']:.1f}×")
        print(f"   Total measurements: {summary['total_speedups_measured']}")

        # Module assessments
        print(f"\n🔍 MODULE ASSESSMENTS:")
        for module, assessment in analysis['module_assessments'].items():
            if assessment.get('status') == 'failed':
                print(f"   ❌ {module}: {assessment['assessment']}")
            else:
                print(f"   {'✅' if assessment.get('overall_success') else '❌'} {module}: {assessment['assessment']}")

        # Key insights
        print(f"\n💡 KEY INSIGHTS:")
        for insight in analysis['key_insights']:
            print(f"   {insight}")

        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        for recommendation in analysis['recommendations']:
            print(f"   {recommendation}")

        # Final verdict
        print(f"\n🏆 FINAL VERDICT:")
        if summary['success_rate'] >= 0.8:
            print("   🎉 TinyTorch optimization modules are working excellently!")
            print("   🚀 Students will see real, measurable performance improvements")
        elif summary['success_rate'] >= 0.6:
            print("   ✅ TinyTorch optimization modules are mostly working well")
            print("   📈 Some areas need improvement but core optimizations deliver")
        elif summary['success_rate'] >= 0.4:
            print("   ⚠️  TinyTorch optimization modules have mixed results")
            print("   🔧 Significant improvements needed for reliable performance gains")
        else:
            print("   ❌ TinyTorch optimization modules need major improvements")
            print("   🚨 Many claimed benefits are not being delivered in practice")

        total_duration = time.time() - self.start_time
        print(f"\n⏱️  Total validation time: {total_duration:.1f} seconds")
        print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def save_results(self, analysis, filename="tinytorch_performance_validation.json"):
        """Save complete results to JSON file."""
        complete_results = {
            'metadata': {
                'validation_time': datetime.now().isoformat(),
                'total_duration_seconds': time.time() - self.start_time,
                'validator_version': '1.0'
            },
            'raw_results': self.results,
            'analysis': analysis
        }

        filepath = Path(__file__).parent / "validation_results" / filename
        filepath.parent.mkdir(exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)

        print(f"💾 Results saved to {filepath}")
        return filepath

def main():
    """Main validation execution."""
    print("Starting TinyTorch Performance Validation...")

    validator = TinyTorchPerformanceValidator()

    try:
        # Run all tests
        results = validator.run_all_tests()

        # Analyze results
        analysis = validator.analyze_results()

        # Print final report
        validator.print_final_report(analysis)

        # Save results
        validator.save_results(analysis)

    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
