#!/bin/bash
echo "# 100-Round Expert Panel Debate (Rounds 31-100)" >> debate-log.md

STATE="The table currently has 80+ primitives across 8 layers (Data, Math, Algorithm, Architecture, Optimization, Runtime, Hardware, Production) and 5 roles (Represent, Compute, Communicate, Control, Measure). Recent additions include Thermodynamics (Td), Entropy (En), Virtualization (Vr), Indexing (Ix), Routing (Ro), and Resilience (Rs)."

for i in {4..10}; do
  START_ROUND=$(( (i-1)*10 + 1 ))
  END_ROUND=$(( i*10 ))
  echo "Running batch $i (Rounds $START_ROUND to $END_ROUND)..."
  
  PROMPT="Act as a panel of 5 experts (Dave Patterson, Chris Lattner, Jeff Dean, Claude Shannon, Dmitri Mendeleev). You are rigorously auditing the Periodic Table of ML Systems to prevent 'reward hacking' and ensure it is a pedagogically perfect, physically accurate framework for teaching Harvard CS249r students and conducting Staff ML interviews. NO superficial additions. If an element is a compound, ruthlessly remove it. Find the deep, hidden flaws. This is batch $i. The current state is: $STATE. \n\nDebate for 10 rounds about edge cases, formal verification, distributed consensus, data provenance, numerical stability, security, and hardware-software co-design limits. Output the dialogue. End with a summary of concrete adjustments (additions/removals/modifications). Keep it concise but deeply technical."
  
  RESPONSE=$(gemini -m gemini-3.1-pro-preview -y -p "$PROMPT")
  
  echo "## Batch $i (Rounds $START_ROUND to $END_ROUND)" >> debate-log.md
  echo "$RESPONSE" >> debate-log.md
  
  STATE=$(echo "$RESPONSE" | tail -n 15)
done
