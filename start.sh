#!/bin/bash
# Starts both the API server and the inference server.
# API server (Node)   → http://localhost:3456
# Inference server (Python) → http://localhost:3457

cd "$(dirname "$0")"

npm install --silent 2>/dev/null

node server/index.js &
NODE_PID=$!

PYTHON_CMD=$(command -v python3 || command -v python)
if [ -z "$PYTHON_CMD" ]; then
  echo "Python not found. Inference server will not start."
else
  "$PYTHON_CMD" server/inference.py &
fi
PY_PID=$!

trap "kill $NODE_PID $PY_PID 2>/dev/null; exit" INT TERM

echo ""
echo "Both servers running. Press Ctrl+C to stop."
echo ""

wait
