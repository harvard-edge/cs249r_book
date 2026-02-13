#!/usr/bin/env python3
"""
Test script for external image detection.
This validates that we correctly extract ONLY actual image URLs,
not citation URLs inside captions.
"""

import re
import sys
from pathlib import Path

# Test cases
TEST_CASES = [
    {
        "name": "Local image with citation in caption",
        "markdown": '![**Caption**: Description. Source: [citation](https://www.numenta.com/blog/)](images/png/sprase-heat-map.png){#fig-sprase-heat-map}',
        "expected_url": "images/png/sprase-heat-map.png",
        "should_flag": False,  # Local image, citation URL should be ignored
    },
    {
        "name": "External image URL (should flag)",
        "markdown": '![Caption text](https://example.com/image.png){#fig-example}',
        "expected_url": "https://example.com/image.png",
        "should_flag": True,
    },
    {
        "name": "Local image with multiple citations in caption",
        "markdown": '![**Title**: Text [link1](https://site1.com) and [link2](https://site2.com)](./images/local.jpg){#fig-test}',
        "expected_url": "./images/local.jpg",
        "should_flag": False,
    },
    {
        "name": "External image without attributes",
        "markdown": '![Simple caption](https://hackster.imgix.net/image.png)',
        "expected_url": "https://hackster.imgix.net/image.png",
        "should_flag": True,
    },
    {
        "name": "Local image with width attribute",
        "markdown": '![Caption](images/png/test.png){width=80% fig-align="center"}',
        "expected_url": "images/png/test.png",
        "should_flag": False,
    },
]

def extract_image_url_improved(markdown_text):
    """
    Extract the actual image URL from markdown image syntax.
    This should extract ONLY the URL immediately after the caption,
    NOT any URLs inside the caption itself.

    Strategy: Parse line by line, find all ]( patterns and take the LAST one as the image URL.
    """
    matches = []

    for line in markdown_text.split('\n'):
        if '![' not in line:
            continue

        # Find image patterns on this line
        idx = 0
        while idx < len(line):
            start = line.find('![', idx)
            if start == -1:
                break

            # Find the end
            end_brace = line.find('}', start)
            next_img = line.find('![', start + 2)

            if end_brace != -1 and (next_img == -1 or end_brace < next_img):
                end = end_brace + 1
            elif next_img != -1:
                end = next_img
            else:
                end = len(line)

            full_match = line[start:end]

            # Find ALL ](url) patterns - take the LAST one
            url_patterns = list(re.finditer(r'\]\(([^)]+)\)', full_match))

            if url_patterns:
                url = url_patterns[-1].group(1).strip()
                print(f"    DEBUG: Pattern matched URL: {url}")
                if url.lower().startswith(('http://', 'https://')):
                    matches.append(url)

            idx = end

    return matches

def run_tests():
    """Run all test cases and report results."""
    print("🧪 Testing Image URL Extraction")
    print("=" * 70)

    passed = 0
    failed = 0

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"  Markdown: {test['markdown'][:80]}...")

        # Extract URLs
        external_urls = extract_image_url_improved(test['markdown'])

        # Check if it should be flagged
        is_flagged = len(external_urls) > 0

        # Validate result
        if is_flagged == test['should_flag']:
            if is_flagged and external_urls[0] == test['expected_url']:
                print(f"  ✅ PASS - Correctly flagged: {external_urls[0]}")
                passed += 1
            elif not is_flagged:
                print(f"  ✅ PASS - Correctly ignored (local image)")
                passed += 1
            else:
                print(f"  ❌ FAIL - Flagged wrong URL")
                print(f"     Expected: {test['expected_url']}")
                print(f"     Got: {external_urls[0] if external_urls else 'None'}")
                failed += 1
        else:
            print(f"  ❌ FAIL - Should {'flag' if test['should_flag'] else 'ignore'}")
            print(f"     Got: {external_urls if external_urls else 'No matches'}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"📊 Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")

    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
