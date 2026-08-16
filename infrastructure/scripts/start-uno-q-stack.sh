#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

test -f apps/api/.env || { echo 'Missing apps/api/.env; copy apps/api/.env.uno-q.example first.' >&2; exit 1; }
test -f services/edge-ai/.env || { echo 'Missing services/edge-ai/.env; copy services/edge-ai/.env.uno-q.example first.' >&2; exit 1; }

docker compose -f infrastructure/compose.uno-q.yml up -d postgres redis
set -a
source apps/api/.env
set +a
pnpm db:generate
pnpm db:deploy
pnpm db:seed
pnpm --filter @signtalk/api build

mkdir -p .run
nohup node apps/api/dist/main.js > .run/nest-api.log 2>&1 &
nohup bash services/edge-ai/scripts/run-uno-q.sh > .run/edge-ai.log 2>&1 &

echo 'Nest API:  http://127.0.0.1:3000/api/v1/health'
echo 'Edge AI:  http://127.0.0.1:8082/health'
