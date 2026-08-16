import { forwardRef, useImperativeHandle, useRef } from 'react';
import { CameraView } from 'expo-camera';
import type { StyleProp, ViewStyle } from 'react-native';

export type CapturedClip = { uri: string } | { frames: Blob[] };

export type CameraHandle = {
  recordAsync: (options: { maxDuration: number; maxFrames?: number }) => Promise<CapturedClip | undefined>;
  stopRecording: () => void;
};

type Props = {
  onCaptureProgress?: (capturedFrames: number, targetFrames: number) => void;
  onError: (message: string) => void;
  onReady: () => void;
  style: StyleProp<ViewStyle>;
};

export const LiveCamera = forwardRef<CameraHandle, Props>(function LiveCamera(
  { onError, onReady, style },
  ref,
) {
  const cameraRef = useRef<CameraView>(null);
  useImperativeHandle(ref, () => ({
    recordAsync: ({ maxDuration }) => cameraRef.current!.recordAsync({ maxDuration }),
    stopRecording: () => cameraRef.current?.stopRecording(),
  }), []);

  return (
    <CameraView
      ref={cameraRef}
      facing="front"
      mode="video"
      onCameraReady={onReady}
      onMountError={(event) => onError(event.message)}
      style={style}
    />
  );
});
