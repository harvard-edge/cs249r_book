"""
Checkpoint 0: Environment Setup (After Module 1 - Setup)
Question: "Can I configure my TinyTorch development environment?"
"""

import sys
import platform
import pytest

def test_checkpoint_00_environment():
    """
    Checkpoint 0: Environment Setup

    Validates that the development environment is properly configured
    and TinyTorch is available for use.
    """
    print("\n🔧 Checkpoint 0: Environment Setup")
    print("=" * 50)

    # Test 1: Python environment
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"✅ Python {python_version}")
    assert sys.version_info.major >= 3, "Python 3+ required"
    assert sys.version_info.minor >= 8, "Python 3.8+ recommended"

    # Test 2: Platform information
    system = platform.system()
    print(f"✅ Platform: {system}")

    # Test 3: TinyTorch availability
    try:
        import tinytorch
        version = getattr(tinytorch, '__version__', 'unknown')
        print(f"✅ TinyTorch {version} ready")
    except ImportError:
        pytest.fail("❌ TinyTorch not available - run installation first")

    # Test 4: Core dependencies
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError:
        pytest.fail("❌ NumPy not available")

    print("\n🎉 Environment Setup Complete!")
    print("📝 You can now configure TinyTorch development environments")
    print("🎯 Next: Build tensor foundations")

if __name__ == "__main__":
    test_checkpoint_00_environment()
