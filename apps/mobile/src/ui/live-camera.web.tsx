import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import type { StyleProp, ViewStyle } from 'react-native';
import type { CameraHandle } from './live-camera';

type Props = {
  onCaptureProgress?: (capturedFrames: number, targetFrames: number) => void;
  onError: (message: string) => void;
  onReady: () => void;
  style: StyleProp<ViewStyle>;
};

const wait = (milliseconds: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, milliseconds));

const captureJpeg = (canvas: HTMLCanvasElement, timeoutMs = 2_000) =>
  new Promise<Blob>((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      reject(new Error('Camera không phản hồi khi tạo ảnh. Hãy bật lại camera rồi thử lại.'));
    }, timeoutMs);
    canvas.toBlob((blob) => {
      window.clearTimeout(timeout);
      if (!blob || blob.size <= 512) {
        reject(new Error('Không thể lấy frame hợp lệ từ camera. Hãy giữ tab này ở màn hình chính rồi thử lại.'));
        return;
      }
      resolve(blob);
    }, 'image/jpeg', 0.82);
  });

export const LiveCamera = forwardRef<CameraHandle, Props>(function LiveCamera(
  { onCaptureProgress, onError, onReady, style },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | undefined>(undefined);
  const stopRequestedRef = useRef(false);

  useEffect(() => {
    let active = true;
    let openedStream: MediaStream | undefined;
    let onTrackEnded: (() => void) | undefined;
    const start = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('Trình duyệt này không hỗ trợ truy cập camera.');
        }
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        });
        if (!active) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        openedStream = stream;
        streamRef.current = stream;
        const handler = () => {
          if (active) onError('Camera đã bị ngắt bởi trình duyệt hoặc ứng dụng khác. Hãy thử lại.');
        };
        onTrackEnded = handler;
        stream.getVideoTracks().forEach((track) => track.addEventListener('ended', handler));
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        onReady();
      } catch (value) {
        const message = value instanceof DOMException && value.name === 'NotAllowedError'
          ? 'Quyền camera đang bị chặn. Hãy cho phép camera trong thanh địa chỉ rồi thử lại.'
          : value instanceof Error ? value.message : 'Không thể mở camera trên trình duyệt.';
        onError(message);
      }
    };
    void start();
    return () => {
      active = false;
      stopRequestedRef.current = true;
      const handler = onTrackEnded;
      if (handler) openedStream?.getVideoTracks().forEach((track) => track.removeEventListener('ended', handler));
      openedStream?.getTracks().forEach((track) => track.stop());
    };
  }, [onError, onReady]);

  useImperativeHandle(ref, () => ({
    recordAsync: async ({ maxDuration, maxFrames = 56 }) => {
      const stream = streamRef.current;
      const video = videoRef.current;
      if (!stream || !video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) {
        const message = 'Camera chưa sẵn sàng. Hãy chờ hình xem trước xuất hiện rồi thử lại.';
        onError(message);
        throw new Error(message);
      }
      if (!stream.getVideoTracks().some((track) => track.readyState === 'live')) {
        const message = 'Luồng camera đã dừng. Hãy bật lại camera rồi thử lại.';
        onError(message);
        throw new Error(message);
      }

      stopRequestedRef.current = false;
      const canvas = document.createElement('canvas');
      const sourceWidth = Math.max(video.videoWidth, 640);
      const sourceHeight = Math.max(video.videoHeight, 480);
      const targetRatio = 4 / 3;
      const sourceRatio = sourceWidth / sourceHeight;
      const cropWidth = sourceRatio > targetRatio ? sourceHeight * targetRatio : sourceWidth;
      const cropHeight = sourceRatio > targetRatio ? sourceHeight : sourceWidth / targetRatio;
      const sourceX = Math.max((sourceWidth - cropWidth) / 2, 0);
      const sourceY = Math.max((sourceHeight - cropHeight) / 2, 0);
      canvas.width = 640;
      canvas.height = 480;
      const context = canvas.getContext('2d', { alpha: false });
      if (!context) throw new Error('Không thể khởi tạo bộ chụp frame.');

      const captureFrame = () => {
        context.drawImage(
          video,
          sourceX,
          sourceY,
          cropWidth,
          cropHeight,
          0,
          0,
          canvas.width,
          canvas.height,
        );
        return captureJpeg(canvas);
      };

      const frames: Blob[] = [];
      // VSL-30 is trained on 48 temporal positions. Capture up to 56 source
      // frames so the server can uniformly resample without inventing motion.
      const frameInterval = 1000 / 20;
      const deadline = performance.now() + maxDuration * 1000;
      while (!stopRequestedRef.current && performance.now() < deadline && frames.length < maxFrames) {
        if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !stream.getVideoTracks().some((track) => track.readyState === 'live')) {
          const message = 'Luồng camera bị gián đoạn khi đang ghi. Hãy thử lại.';
          onError(message);
          throw new Error(message);
        }
        const frame = await captureFrame();
        frames.push(frame);
        onCaptureProgress?.(frames.length, maxFrames);
        await wait(frameInterval);
      }
      if (frames.length < 4) {
        const message = 'Camera chưa tạo đủ frame. Hãy giữ nhận diện ít nhất một giây.';
        onError(message);
        throw new Error(message);
      }
      return { frames };
    },
    stopRecording: () => {
      stopRequestedRef.current = true;
    },
  }), [onCaptureProgress, onError]);

  const flattened = Array.isArray(style) ? Object.assign({}, ...style) : style;
  return (
    <video
      ref={videoRef}
      autoPlay
      muted
      playsInline
      style={{
        ...flattened,
        backgroundColor: '#13201A',
        height: '100%',
        objectFit: 'cover',
        transform: 'scaleX(-1)',
        width: '100%',
      }}
    />
  );
});
