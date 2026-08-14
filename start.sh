#!/bin/bash
# Starts both the API server and the inference server.
# API server (Node)   → http://localhost:3456
# Inference server (Python) → http://localhost:3457

cd "$(dirname "$0")"

npm install --silent 2>/dev/null

node server/index.js &
NODE_PID=$!

# The sidecar now fine-tunes as well as predicts. Training rebuilds the model
# from its yaml, which needs an ultralytics new enough for the architecture:
# 8.3.x loads and predicts with yolo26 weights but dies with
# "SPPF.__init__() takes from 3 to 4 positional arguments but 6 were given".
# Prefer an interpreter whose ultralytics can build yolo26.
pick_python() {
  for c in /usr/bin/python3 "$(command -v python3)" "$(command -v python)"; do
    [ -x "$c" ] || continue
    "$c" - <<'EOF' >/dev/null 2>&1 || continue
import ultralytics
major, minor, *_ = (int(x) for x in ultralytics.__version__.split(".")[:2])
raise SystemExit(0 if (major, minor) >= (8, 4) else 1)
EOF
    echo "$c"; return
  done
  command -v python3 || command -v python
}
PYTHON_CMD=$(pick_python)
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
