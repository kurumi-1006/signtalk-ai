# SIGNTALK AI

Nền tảng Edge AI chuyển ngôn ngữ ký hiệu Việt Nam thành văn bản. Ứng dụng Expo gửi video hoặc JPEG frames đến Edge AI; Edge xử lý local, sau đó gửi sự kiện HMAC tới NestJS API. API lưu sự kiện trong PostgreSQL và phát realtime qua Socket.IO. Hệ thống không lưu video/ảnh trên API.

## Khởi động nhanh

```bash
pnpm install
cp .env.example .env
cp apps/api/.env.example apps/api/.env
pnpm compose:up
pnpm db:generate && pnpm db:migrate && pnpm db:seed
pnpm dev:api
```

Ở terminal khác, chạy web client bằng `pnpm dev:web`. Để dùng nhận diện thật, làm theo [hướng dẫn phát triển](docs/development.md) để cấu hình và chạy `services/edge-ai` trên cổng 8082.

## Tài liệu

- [Kiến trúc](docs/architecture.md)
- [Phát triển cục bộ](docs/development.md)
- [Xác thực và bảo mật thiết bị](docs/authentication.md)
- [Triển khai Arduino UNO Q](docs/uno-q-web-integration.md)

## Notebook huấn luyện

Các notebook Kaggle được lưu tại `notebooks/` và không thuộc runtime của ứng dụng:

- `vsl-low-shot-train-from-saved-keypoints.ipynb`: huấn luyện low-shot từ keypoint đã lưu.
- `train-v4-0-analytist-to-v3-final.ipynb`: tinh chỉnh an toàn model VSL-30.
- `vsl-lowshot-metric-learning-kaggle.xpynb`: file nguồn hiện rỗng (0 byte), cần thay bằng notebook Kaggle hợp lệ trước khi chạy.

## Kiểm tra chất lượng

```bash
pnpm lint && pnpm typecheck && pnpm test && pnpm build
cd services/edge-ai && ruff check . && mypy src && pytest
```
