import sys
import re

with open("books/vol3/MASTER_TEXTBOOK_OUTLINE_V2.md", "r", encoding="utf-8") as f:
    v2_text = f.read()

# Verify that V2 text has the expected markers
assert "## Part I: The Stochastic Processor" in v2_text
assert "### Chapter 02: The Foundation Model as a Processing Element" in v2_text
assert "### Chapter 03: Inference-Time Deliberation" in v2_text
assert "### Chapter 05: The KV-Cache Hierarchy" in v2_text

print("V2 markers verified successfully!")
