import { recognitionEventSchema, type RecognitionEvent } from '@signtalk/contracts';
import type { CapturedClip } from '../ui/live-camera';
import { env } from '../config/env';

type UploadedVideo = {
  file?: Blob;
  mimeType?: string;
  name: string;
  uri: string;
};

export type PipelineDiagnostics = Record<string, boolean | number | string>;
export type RecognitionResult = {
  diagnostics: PipelineDiagnostics;
  event: RecognitionEvent;
};
export type EdgeModel = { id: string; name: string; type: string };
export type EdgeModels = { active_model_id: string; models: EdgeModel[] };

export async function getEdgeModels(): Promise<EdgeModels> {
  if (!env.edgeAiUrl) throw new Error('Chưa cấu hình EXPO_PUBLIC_EDGE_AI_URL.');
  const response = await fetch(`${env.edgeAiUrl}/models`);
  if (!response.ok) throw new Error('Không thể tải danh sách mô hình.');
  return response.json() as Promise<EdgeModels>;
}

export async function activateEdgeModel(modelId: string): Promise<string> {
  if (!env.edgeAiUrl) throw new Error('Chưa cấu hình EXPO_PUBLIC_EDGE_AI_URL.');
  const response = await fetch(`${env.edgeAiUrl}/models/${encodeURIComponent(modelId)}/activate`, { method: 'POST' });
  const payload: unknown = await response.json();
  if (!response.ok) throw new Error(typeof payload === 'object' && payload && 'detail' in payload ? String(payload.detail) : 'Không thể đổi mô hình.');
  return String((payload as { active_model_id: string }).active_model_id);
}

async function submitRecognition(body: FormData): Promise<RecognitionResult> {
  const response = await fetch(`${env.edgeAiUrl}/predict`, { method: 'POST', body });
  const payload: unknown = await response.json();
  if (!response.ok) {
    throw new Error(
      typeof payload === 'object' && payload && 'detail' in payload
        ? String(payload.detail)
        : 'Edge AI không thể nhận diện clip.',
    );
  }
  const event = recognitionEventSchema.safeParse((payload as { event?: unknown }).event);
  if (!event.success) throw new Error('Edge AI trả về kết quả không đúng định dạng.');
  const rawDiagnostics = (payload as { diagnostics?: unknown }).diagnostics;
  const diagnostics = typeof rawDiagnostics === 'object' && rawDiagnostics
    ? Object.fromEntries(
      Object.entries(rawDiagnostics).filter((entry): entry is [string, boolean | number | string] =>
        ['boolean', 'number', 'string'].includes(typeof entry[1]),
      ),
    )
    : {};
  return { event: event.data, diagnostics };
}

function sampleFrames(frames: Blob[], count = 56): Blob[] {
  if (frames.length <= count) return frames;
  return Array.from({ length: count }, (_, index) => {
    const position = Math.round(index * (frames.length - 1) / (count - 1));
    return frames[position];
  });
}

export async function recognizeVideo(clip: CapturedClip): Promise<RecognitionResult> {
  if (!env.edgeAiUrl) throw new Error('Chưa cấu hình EXPO_PUBLIC_EDGE_AI_URL.');
  const body = new FormData();
  body.append('device_id', env.deviceId);

  if ('frames' in clip) {
    sampleFrames(clip.frames).forEach((frame, index) => {
      body.append('frames', frame, `frame-${String(index).padStart(3, '0')}.jpg`);
    });
  } else {
    body.append(
      'video',
      { uri: clip.uri, name: 'signtalk-clip.mp4', type: 'video/mp4' } as unknown as Blob,
    );
  }
  return submitRecognition(body);
}

export async function recognizeUploadedVideo(upload: UploadedVideo): Promise<RecognitionResult> {
  if (!env.edgeAiUrl) throw new Error('Chưa cấu hình EXPO_PUBLIC_EDGE_AI_URL.');
  const body = new FormData();
  body.append('device_id', env.deviceId);
  if (upload.file) {
    body.append('video', upload.file, upload.name);
  } else {
    body.append(
      'video',
      {
        uri: upload.uri,
        name: upload.name,
        type: upload.mimeType ?? 'video/mp4',
      } as unknown as Blob,
    );
  }
  return submitRecognition(body);
}
