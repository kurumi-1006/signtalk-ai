# Phát triển cục bộ

Yêu cầu: Node.js 24, pnpm 10, Docker Desktop và Python 3.11 cho Edge AI.

```powershell
pnpm install
Copy-Item .env.example .env
Copy-Item apps/api/.env.example apps/api/.env
pnpm compose:up
pnpm db:generate
pnpm db:migrate
pnpm db:seed
```

Chạy API và web client ở hai terminal:

```bash
pnpm dev:api
pnpm dev:web
```

Để chạy Edge AI, tạo `services/edge-ai/.env` từ `.env.example`, đặt `MODEL_PATH` và `LABELS_PATH` tới model local, rồi:

```bash
cd services/edge-ai
python -m venv .venv
.venv/Scripts/activate  # Windows PowerShell
pip install -e '.[dev]'
python -m src.main --serve --port 8082
```

Kiểm tra chất lượng trước khi commit:

```bash
pnpm lint
pnpm typecheck
pnpm test
pnpm build
cd services/edge-ai && ruff check . && mypy src && pytest
```
