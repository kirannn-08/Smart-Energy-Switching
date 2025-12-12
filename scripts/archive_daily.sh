#!/usr/bin/env bash
# Usage: scripts/archive_daily.sh path/to/daily.csv
set -e
SRC="$1"
if [ -z "$SRC" ]; then
  echo "Usage: $0 path/to/daily_sim_YYYYMMDD.csv"
  exit 1
fi
mkdir -p data/archive
mkdir -p data/train_pool
cp "$SRC" "data/archive/"
cp "$SRC" "data/train_pool/"
echo "Copied $SRC -> data/archive/ and data/train_pool/"
