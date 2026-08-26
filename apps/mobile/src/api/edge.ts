import type { CapturedClip } from '../ui/live-camera';
import { env } from '../config/env';
import type { RecognitionEvent } from '../types/recognition';

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

const EDGE_CONNECTION_TIMEOUT_MS = 3_000;
// Keep the last healthy UNO Q endpoint so repeated camera cycles do not probe
// every configured address. A failed request clears this cache and triggers
// endpoint discovery again, which covers DHCP/IP changes on the local network.
let activeEdgeAiUrl: string | undefined;

function configuredEdgeAiUrls(): string[] {
  const urls = (env.edgeAiUrls ?? '')
    .split(/[\s,;]+/)
    .map((url) => url.trim().replace(/\/$/, ''))
    .filter(Boolean);
  return [...new Set(urls)];
}

async function isHealthy(url: string): Promise<boolean> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), EDGE_CONNECTION_TIMEOUT_MS);
  try {
    const response = await fetch(`${url}/health`, { signal: controller.signal });
    if (!response.ok) return false;
    const health = await response.json() as { status?: string; model_id?: string };
    return health.status === 'ok' && health.model_id === 'vsl30_v4_3';
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

async function resolveEdgeAiUrl(force = false): Promise<string> {
  if (!force && activeEdgeAiUrl) return activeEdgeAiUrl;
  const urls = configuredEdgeAiUrls();
  if (!urls.length) throw new Error('EXPO_PUBLIC_EDGE_AI_URLS is not configured.');
  // Probe candidates concurrently: a powered-off fallback endpoint must not
  // delay a reachable UNO Q endpoint behind its full timeout.
  const reachable = await Promise.all(urls.map(async (url) => ({ url, healthy: await isHealthy(url) })));
  const endpoint = reachable.find((candidate) => candidate.healthy)?.url;
  if (!endpoint) throw new Error('Could not connect to Edge AI at any configured address.');
  activeEdgeAiUrl = endpoint;
  return endpoint;
}

async function edgeFetch(path: string, init?: RequestInit): Promise<Response> {
  const endpoint = await resolveEdgeAiUrl();
  try {
    return await fetch(`${endpoint}${path}`, init);
  } catch {
    // Wi-Fi/IP can change while the app remains open. Recheck the full list
    // and retry the request once using a reachable endpoint.
    activeEdgeAiUrl = undefined;
    const replacement = await resolveEdgeAiUrl(true);
    return fetch(`${replacement}${path}`, init);
  }
}

export async function getEdgeModels(): Promise<EdgeModels> {
  const response = await edgeFetch('/models');
  if (!response.ok) throw new Error('Could not load the model list.');
  return response.json() as Promise<EdgeModels>;
}

async function submitRecognition(body: FormData): Promise<RecognitionResult> {
  const response = await edgeFetch('/predict', { method: 'POST', body });
  const payload: unknown = await response.json();
  if (!response.ok) {
    throw new Error(
      typeof payload === 'object' && payload && 'detail' in payload
        ? String(payload.detail)
        : 'Edge AI could not recognize this clip.',
    );
  }
  const event = (payload as { event?: RecognitionEvent }).event;
  if (!event || event.eventType !== 'recognition.confirmed' || !event.payload?.label) {
    throw new Error('Edge AI returned an invalid response.');
  }
  const rawDiagnostics = (payload as { diagnostics?: unknown }).diagnostics;
  const diagnostics = typeof rawDiagnostics === 'object' && rawDiagnostics
    ? Object.fromEntries(
      Object.entries(rawDiagnostics).filter((entry): entry is [string, boolean | number | string] =>
        ['boolean', 'number', 'string'].includes(typeof entry[1]),
      ),
    )
    : {};
  return { event, diagnostics };
}

function sampleFrames(frames: Blob[], count = 32): Blob[] {
  if (frames.length <= count) return frames;
  // Uniform sampling keeps the first and last moments of the sign while
  // matching the Edge AI service's bounded MediaPipe workload.
  return Array.from({ length: count }, (_, index) => {
    const position = Math.round(index * (frames.length - 1) / (count - 1));
    return frames[position];
  });
}

export async function recognizeVideo(clip: CapturedClip): Promise<RecognitionResult> {
  const body = new FormData();
  if ('frames' in clip) {
    sampleFrames(clip.frames).forEach((frame, index) => {
      body.append('frames', frame, `frame-${String(index).padStart(3, '0')}.jpg`);
    });
  } else {
    body.append(
      'video',
      { uri: clip.uri, name: 'signtalk-ai-clip.mp4', type: 'video/mp4' } as unknown as Blob,
    );
  }
  return submitRecognition(body);
}

export async function recognizeUploadedVideo(upload: UploadedVideo): Promise<RecognitionResult> {
  const body = new FormData();
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
