#!/usr/bin/env bash
set -euo pipefail

# Compile all Java source files and place class files under bin/
mkdir -p bin
find . -name "*.java" | sort > sources.txt
javac -d bin @sources.txt
rm -f sources.txt

echo "Compiled Java sources to bin/"
