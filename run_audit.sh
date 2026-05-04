#!/bin/bash
TRACK=$1
MODEL=$2
if [ -z "$MODEL" ]; then
  cd MLSysBook-yaml-audit
  gemini --yolo --skip-trust -p "$(cat interviews/vault/_pipeline/runs/gemini-self-audit/prompts/${TRACK}_audit_prompt.md)" < /dev/null
else
  cd MLSysBook-yaml-audit
  gemini -m "$MODEL" --yolo --skip-trust -p "$(cat interviews/vault/_pipeline/runs/gemini-self-audit/prompts/${TRACK}_audit_prompt.md)" < /dev/null
fi
