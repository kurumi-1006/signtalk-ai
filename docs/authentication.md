# Xác thực và bảo mật thiết bị

Người dùng đăng nhập qua Better Auth tại `/api/auth`. Web/mobile client gửi cookie session; API bật CORS có credential và chỉ chấp nhận các origin cấu hình trong `apps/api/.env`.

Edge device không dùng session người dùng. Khi gửi một sự kiện nhận diện về `POST /api/v1/recognition`, thiết bị phải kèm các header sau:

- `x-device-id`
- `x-key-id`
- `x-timestamp`
- `x-nonce`
- chữ ký HMAC-SHA256

Trong development, API dùng `DEVICE_SEED_SECRET`; Edge dùng `DEVICE_SECRET`. Giá trị hai bên phải khớp. Production cần cấp `DeviceCredential` riêng cho mỗi thiết bị, lưu secret trong secret manager và dùng nonce store dùng chung (ví dụ Redis) để chống replay.

Không commit `.env`, device secret, model weights hoặc dữ liệu người dùng. Chỉ dùng các file `.env.example` làm mẫu cấu hình.
