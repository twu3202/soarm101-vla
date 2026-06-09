#!/bin/bash
# Start the policy server (latest checkpoint) detached, then run the inference test client.
# The server STAYS UP after this returns (for the real-arm deploy).
cd ~/Projects/openpi
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# already serving?
if ss -ltn 2>/dev/null | grep -q ':8000'; then
    echo "server already listening on :8000"
else
    echo "launching policy server (detached) ..."
    nohup setsid bash ~/Projects/so101_sort/serve_so101.sh > ~/Projects/so101_sort/serve.log 2>&1 &
    sleep 2
fi

echo "=== running test client (waits for server to load, ~60-90s) ==="
uv run python ~/Projects/so101_sort/test_infer.py
echo "=== serve.log tail ==="
tail -6 ~/Projects/so101_sort/serve.log
