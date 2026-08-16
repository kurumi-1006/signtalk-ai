# Kiến trúc

SIGNTALK AI chuyển ngôn ngữ ký hiệu Việt Nam thành văn bản bằng mô hình chạy tại edge. Video hoặc chuỗi JPEG được xử lý tại thiết bị Edge AI; API chỉ nhận sự kiện nhận diện đã ký, không nhận hay lưu media.

```text
Expo client ── video/JPEG ──> Edge AI (FastAPI :8082)
                                  │
                                  │ signed recognition event
                                  v
                            NestJS API (:3000) ──> PostgreSQL
                                  │
                                  └──────────────> Socket.IO clients
```

Các thành phần chính:

- `apps/mobile`: ứng dụng Expo/React Native, chạy được trên web và thiết bị di động.
- `services/edge-ai`: FastAPI + MediaPipe + ONNX Runtime; nhận diện clip, áp ngưỡng chất lượng và gửi sự kiện nền về API.
- `apps/api`: NestJS, Better Auth, REST API phiên bản `/api/v1`, Swagger và Socket.IO.
- `database/prisma`: Prisma schema, migration và dữ liệu seed cho PostgreSQL.
- `packages/contracts`: hợp đồng TypeScript dùng chung giữa các ứng dụng.
- `infrastructure`: Docker Compose cho PostgreSQL/Redis và script vận hành UNO Q.

Redis được khởi tạo cùng môi trường phát triển để hỗ trợ mở rộng cache/Socket.IO. Hiện API lưu dữ liệu nghiệp vụ trong PostgreSQL.
