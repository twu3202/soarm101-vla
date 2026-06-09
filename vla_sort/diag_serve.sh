#!/bin/bash
cd ~/Projects/openpi
echo "===== serve_policy.py (head + args) ====="
sed -n 1,95p scripts/serve_policy.py
echo "===== openpi_client package files ====="
find packages -path '*openpi_client*' -name '*.py' 2>/dev/null | head -30
echo "===== websocket_client_policy.py (API) ====="
WS=$(find packages -name 'websocket_client_policy.py' 2>/dev/null | head -1)
echo "FILE: $WS"
sed -n 1,90p "$WS" 2>/dev/null
echo "===== an example client (simple_client / libero) ====="
ls examples/simple_client 2>/dev/null
sed -n 1,80p examples/simple_client/main.py 2>/dev/null
