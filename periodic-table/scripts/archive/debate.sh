#!/bin/bash
echo "# 30-Round Expert Panel Debate" > debate-log.md

STATE="The Periodic Table of ML Systems currently has 80 primitives across 8 layers (Data, Math, Algorithm, Architecture, Optimization, Runtime, Hardware, Production) and 5 roles (Represent, Compute, Communicate, Control, Measure)."

for i in {1..5}; do
  echo "Running batch $i (Rounds $(( (i-1)*6 + 1 )) to $(( i*6 )))..."
  
  PROMPT="Act as a panel of 5 experts (Dave Patterson, Chris Lattner, Jeff Dean, Claude Shannon, Dmitri Mendeleev). You are rigorously auditing the Periodic Table of ML Systems. This is batch $i of 5 (Rounds $(( (i-1)*6 + 1 )) to $(( i*6 ))). The current state is: $STATE. \n\nReview the elements, debate violently about missing physical laws, entropy constraints, state/memory leaks, database indexing, caching anomalies, and architectural primitives. Output the dialogue for these 6 rounds. End with a summary of the concrete adjustments made to the elements. Keep it concise but deeply technical."
  
  RESPONSE=$(gemini -m gemini-3.1-pro-preview -y -p "$PROMPT")
  
  echo "## Batch $i (Rounds $(( (i-1)*6 + 1 )) to $(( i*6 )))" >> debate-log.md
  echo "$RESPONSE" >> debate-log.md
  
  # Extract the last few lines to feed into the next state
  STATE=$(echo "$RESPONSE" | tail -n 10)
done
