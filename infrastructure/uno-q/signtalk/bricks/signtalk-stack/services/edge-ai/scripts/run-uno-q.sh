#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
test -f .env || { echo 'Missing .env. Copy .env.uno-q.example to .env and configure it.' >&2; exit 1; }

source .venv/bin/activate
exec python -m src.main --serve --port 8082
