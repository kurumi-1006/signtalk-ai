# Chạy Edge AI trên Arduino UNO Q

UNO Q chạy FastAPI Edge AI trên Linux processor, còn PostgreSQL, Redis và NestJS được khởi động bằng Docker Compose. Chép repository sang thiết bị và bảo đảm model đã có dưới `services/edge-ai/models/`; model weights không được lưu trong Git.

```bash
cd /home/arduino/signtalk
corepack enable
pnpm install --frozen-lockfile
cd services/edge-ai
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
cd ../..
cp apps/api/.env.uno-q.example apps/api/.env
cp services/edge-ai/.env.uno-q.example services/edge-ai/.env
```

Đặt cùng một secret mạnh vào `DEVICE_SEED_SECRET` trong API và `DEVICE_SECRET` trong Edge AI. Sau đó khởi động stack:

```bash
chmod +x infrastructure/scripts/start-uno-q-stack.sh services/edge-ai/scripts/run-uno-q.sh
./infrastructure/scripts/start-uno-q-stack.sh
curl http://127.0.0.1:3000/api/v1/health
curl http://127.0.0.1:8082/health
```

Trên máy chạy Expo, cấu hình `apps/mobile/.env` với địa chỉ LAN của UNO Q:

```dotenv
EXPO_PUBLIC_API_URL=http://UNO_Q_IP:3000
EXPO_PUBLIC_SOCKET_URL=http://UNO_Q_IP:3000
EXPO_PUBLIC_EDGE_AI_URL=http://UNO_Q_IP:8082
EXPO_PUBLIC_DEVICE_ID=uno-q-demo
```

Mở web client bằng `pnpm dev:web`. Clip video được gửi trực tiếp tới `POST /predict` trên Edge AI; phản hồi trả về kết quả, top-K và chẩn đoán. Khi kết quả được chấp nhận, Edge gửi sự kiện đã ký về API để lưu và phát realtime.
