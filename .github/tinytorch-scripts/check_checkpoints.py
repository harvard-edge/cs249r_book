#!/usr/bin/env python3
"""
Validate checkpoint markers in long modules (8+ hours).
Ensures complex modules have progress markers to help students track completion.
"""

import re
import sys
from pathlib import Path


def extract_time_estimate(about_file):
    """Extract time estimate from ABOUT.md"""
    if not about_file.exists():
        return 0

    content = about_file.read_text()
    match = re.search(r'time_estimate:\s*"(\d+)-(\d+)\s+hours"', content)

    if match:
        return int(match.group(2))  # Return upper bound
    return 0


def count_checkpoints(about_file):
    """Count checkpoint markers in ABOUT.md"""
    if not about_file.exists():
        return 0

    content = about_file.read_text()
    # Look for checkpoint patterns
    return len(re.findall(r'\*\*✓ CHECKPOINT \d+:', content))


def main():
    """Validate checkpoint markers in long modules"""
    modules_dir = Path("modules")
    recommendations = []
    validated = []

    print("🏁 Validating Checkpoint Markers")
    print("=" * 60)

    # Find all module directories
    module_dirs = sorted([d for d in modules_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])

    for module_dir in module_dirs:
        module_name = module_dir.name
        about_file = module_dir / "ABOUT.md"

        time_estimate = extract_time_estimate(about_file)
        checkpoint_count = count_checkpoints(about_file)

        # Modules 8+ hours should have checkpoints
        if time_estimate >= 8:
            if checkpoint_count == 0:
                recommendations.append(
                    f"⚠️  {module_name} ({time_estimate}h): Consider adding checkpoint markers"
                )
            elif checkpoint_count >= 2:
                validated.append(
                    f"✅ {module_name} ({time_estimate}h): {checkpoint_count} checkpoints"
                )
            else:
                recommendations.append(
                    f"⚠️  {module_name} ({time_estimate}h): Only {checkpoint_count} checkpoint (recommend 2+)"
                )
        else:
            print(f"   {module_name} ({time_estimate}h): Checkpoints not required")

    print("\n" + "=" * 60)

    # Print validated modules
    if validated:
        print("\n✅ Modules with Good Checkpoint Coverage:")
        for item in validated:
            print(f"  {item}")

    # Print recommendations
    if recommendations:
        print("\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  {rec}")
        print("\nNote: This is informational only, not a blocker.")

    print("\n✅ Checkpoint validation complete!")
    sys.exit(0)


if __name__ == "__main__":
    main()
