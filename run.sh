#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <MainClass> [args...]"
  echo "Example: $0 com.nexusai.assistant.ai.advanced.ActiveLearningSystem"
  exit 1
fi

java -cp bin "$@"
