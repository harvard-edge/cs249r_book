#!/bin/bash
set -e

# setup_micro_datasets.sh
# Bootstraps the MLPerf EDU environment by ensuring real-data shards are present.

DATA_DIR="$HOME/.mlperf_edu/data"
mkdir -p "$DATA_DIR"

echo "🛠️ Initializing MLPerf EDU Real-Data Shards..."

# Shard list
SHARDS=("cifar10_micro" "speech_commands_micro")
BASE_URL="https://raw.githubusercontent.com/MLSysBook/mlperf-edu-data/main/shards"

for SHARD in "${SHARDS[@]}"; do
    FILE="$DATA_DIR/$SHARD.npz"
    if [ ! -f "$FILE" ]; then
        echo "📥 Fetching $SHARD..."
        curl -L "$BASE_URL/$SHARD.npz" -o "$FILE" --silent
        echo "✅ Shard $SHARD ready."
    else
        echo "✅ Shard $SHARD already exists in $DATA_DIR."
    fi
done

echo "🚀 All educational shards are ready for training!"
