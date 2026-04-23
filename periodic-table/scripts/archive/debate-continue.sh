#!/bin/bash
# Continuation: waits for the first debate.sh to finish, then runs rounds 21-40 + final pass.

set -e
cd "$(dirname "$0")"

echo "⏳ Waiting for current debate (rounds 2-20) to finish..."

# Wait for the original debate.sh to exit
while pgrep -f "debate.sh 20 2" > /dev/null 2>&1; do
  sleep 10
done

echo "✓ Rounds 2-20 complete. Starting rounds 21-40..."
echo ""

# Now run rounds 21-40 using the same script
./debate.sh 40 21
