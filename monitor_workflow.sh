#!/bin/bash

echo "Monitoring Container Health Check Workflow"
echo "========================================="
echo "Workflow ID: 16862784790"
echo "Started at: $(date)"
echo ""

while true; do
  echo "Checking at $(date)..."
  
  STATUS=$(gh run view 16862784790 --json status --jq '.status')
  CONCLUSION=$(gh run view 16862784790 --json conclusion --jq '.conclusion')
  
  echo "Status: $STATUS"
  if [ "$CONCLUSION" != "null" ]; then
    echo "Conclusion: $CONCLUSION"
  fi
  
  echo "Jobs:"
  gh run view 16862784790 --json jobs --jq '.jobs[] | "  " + .name + ": " + .status + (if .conclusion != null then " (" + .conclusion + ")" else "" end)'
  
  echo ""
  
  if [ "$STATUS" = "completed" ]; then
    echo "WORKFLOW COMPLETED"
    echo "Final Result: $CONCLUSION"
    echo ""
    echo "Final Summary:"
    gh run view 16862784790
    break
  fi
  
  echo "Waiting 60 seconds..."
  echo "----------------------------------------"
  sleep 60
done
