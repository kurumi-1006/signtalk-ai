import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  type TextStyle,
  useWindowDimensions,
  View,
} from 'react-native';
import { useCameraPermissions } from 'expo-camera';
import * as DocumentPicker from 'expo-document-picker';
import { Ionicons } from '@expo/vector-icons';
import * as Speech from 'expo-speech';
import { router } from 'expo-router';
import type { SentenceToken } from '../src/types/recognition';
import {
  recognizeUploadedVideo,
  recognizeVideo,
  type PipelineDiagnostics,
} from '../src/api/edge';
import { useRecognitionStore } from '../src/store/recognition';
import { AppNavigation } from '../src/ui/app-navigation';
import { LiveCamera, type CameraHandle } from '../src/ui/live-camera';

type PipelineLog = {
  detail: string;
  status: 'active' | 'done' | 'error' | 'pending';
  title: string;
};

const VSL30_CAPTURE = { maxDuration: 3, maxFrames: 32 };

const confidenceValue = (value?: number) =>
  Math.round(Math.max(0, Math.min(value ?? 0, 1)) * 100);

const isMultiVslGloss = (label?: string) => /^vsl_gloss_\d{4}$/.test(label ?? '');
const displayGloss = (label?: string) => isMultiVslGloss(label)
  ? `Gloss #${label?.slice(-4)}`
  : label ?? '—';

const logRuntimeDiagnostics = (source: 'camera' | 'upload', diagnostics: PipelineDiagnostics) => {
  console.log('[SignTalk AI Edge AI] Processing diagnostics', {
    source,
    decodedFrames: diagnostics.input_frames ?? diagnostics.decoded_frames,
    sampledFrames: diagnostics.sampled_frames,
    mediaPipeFrames: diagnostics.mediapipe_input_frames,
    decodeMs: diagnostics.video_decode_ms ?? diagnostics.jpeg_decode_ms,
    keypointMs: diagnostics.mediapipe_preprocess_ms,
    inferenceMs: diagnostics.onnx_inference_ms ?? diagnostics.pytorch_inference_ms,
    totalMs: diagnostics.total_request_ms,
  });
};

const logRecognitionStep = (
  source: 'camera' | 'upload',
  step: string,
  message: string,
  details?: Record<string, unknown>,
) => {
  console.info(`[SignTalk AI Edge AI][${source}] ${step} ${message}`, details ?? '');
};

export default function Home() {
  const { width } = useWindowDimensions();
  const desktop = width >= 980;
  const compact = width < 640;
  const isWeb = Platform.OS === 'web';
  const cameraRef = useRef<CameraHandle>(null);
  const runningRef = useRef(false);
  // A prediction must be stable across captures before it becomes a sentence
  // token; these refs preserve that debounce state without rerendering the UI.
  const pendingCandidateRef = useRef<{ label: string; count: number } | undefined>(undefined);
  const lastCommittedRef = useRef<{ label: string; at: number } | undefined>(undefined);
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraReady, setCameraReady] = useState(false);
  const [webCameraEnabled, setWebCameraEnabled] = useState(false);
  const [cameraError, setCameraError] = useState<string>();
  const [capturedFrames, setCapturedFrames] = useState(0);
  const [qualityHint, setQualityHint] = useState<string>();
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState<string>();
  const [pipelineLogs, setPipelineLogs] = useState<PipelineLog[]>([]);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [tokens, setTokens] = useState<SentenceToken[]>([]);
  const [finalSentence, setFinalSentence] = useState<string>();
  const [error, setError] = useState<string>();
  const { current, setEvent } = useRecognitionStore();
  const cameraAllowed = isWeb ? webCameraEnabled : permission?.granted;

  const enableCamera = async () => {
    setCameraError(undefined);
    setCameraReady(false);
    if (isWeb) {
      setWebCameraEnabled(true);
      return;
    }
    await requestPermission();
  };

  const handleCameraReady = useCallback(() => {
    setCameraError(undefined);
    setCameraReady(true);
  }, []);

  const handleCameraError = useCallback((message: string) => {
    setCameraError(message);
    setCameraReady(false);
    if (isWeb) setWebCameraEnabled(false);
  }, [isWeb]);

  useEffect(() => {
    return () => {
      runningRef.current = false;
      Speech.stop();
    };
  }, []);

  const startRecognition = useCallback(async () => {
    if (!cameraReady || !cameraRef.current || runningRef.current) return;
    runningRef.current = true;
    setError(undefined);
    setQualityHint(undefined);
    pendingCandidateRef.current = undefined;
    lastCommittedRef.current = undefined;
    setTokens([]);
    setFinalSentence(undefined);
    setCapturedFrames(0);
    setIsRecognizing(true);
    setPipelineLogs([
      { title: '1. Capture camera', detail: 'Capture up to 32 frames in 3 seconds.', status: 'active' },
      { title: '2. Sample frames', detail: 'Preserve the full motion sequence before MediaPipe.', status: 'pending' },
      { title: '3. Extract landmarks', detail: '33 pose and 42 hand landmarks per frame.', status: 'pending' },
      { title: '4. Reconstruct landmarks', detail: 'Interpolate joints that are not visible.', status: 'pending' },
      { title: '5. Normalize input', detail: 'Center, scale by shoulders, and apply visibility masks.', status: 'pending' },
      { title: '6. Prepare tensor', detail: 'Resize the sequence to [1, 48, 75, 4].', status: 'pending' },
      { title: '7. Run ONNX V4.3', detail: 'CPU inference with softmax over 30 classes.', status: 'pending' },
      { title: '8. Filter confidence', detail: 'Check confidence, margin, and hand coverage.', status: 'pending' },
    ]);

    while (runningRef.current) {
      try {
        logRecognitionStep('camera', '1/4', 'Starting camera capture', {
          maxDurationSeconds: VSL30_CAPTURE.maxDuration,
          maxFrames: VSL30_CAPTURE.maxFrames,
        });
        setPipelineLogs((logs) => logs.map((log, index) => index === 0
          ? { ...log, detail: 'Capturing hand movement for 3 seconds…', status: 'active' }
          : { ...log, status: 'pending' }));
        setCapturedFrames(0);
        const clip = await cameraRef.current.recordAsync(VSL30_CAPTURE);
        if (!clip || !runningRef.current) continue;
        logRecognitionStep('camera', '2/4', 'Capture complete', {
          sourceFrames: 'frames' in clip ? clip.frames.length : 'video',
        });
        setIsProcessing(true);
        setPipelineLogs((logs) => logs.map((log, index) => {
          if (index === 0) return { ...log, detail: `Captured ${'frames' in clip ? clip.frames.length : 'video'} source frames.`, status: 'done' };
          if (index === 1) return { ...log, detail: 'Sending clip to Edge AI…', status: 'active' };
          return log;
        }));
        logRecognitionStep('camera', '3/4', 'Sending capture to Edge AI');
        const { event, diagnostics } = await recognizeVideo(clip);
        logRuntimeDiagnostics('camera', diagnostics);
        logRecognitionStep('camera', '4/4', 'Recognition complete', {
          label: event.payload.label,
          confidence: event.payload.confidence,
          accepted: event.payload.accepted,
        });
        const milliseconds = (key: string) => `${Math.round(Number(diagnostics[key] ?? 0))} ms`;
        setPipelineLogs([
          { title: '1. Capture camera', detail: `${diagnostics.received_frames ?? diagnostics.input_frames ?? '—'} source frames ready.`, status: 'done' },
          { title: '2. Sample frames', detail: `${diagnostics.input_frames ?? diagnostics.decoded_frames ?? '—'} → ${diagnostics.mediapipe_input_frames ?? '—'} frames for MediaPipe`, status: 'done' },
          { title: '3. Extract landmarks', detail: '33 pose + 42 hand landmarks • V4.3 keypoint format', status: 'done' },
          { title: '4. Reconstruct landmarks', detail: 'Interpolate each occluded joint over time.', status: 'done' },
          { title: '5. Normalize input', detail: 'Center, scale by shoulders, and apply visibility masks.', status: 'done' },
          { title: '6. Prepare tensor', detail: `${diagnostics.sampled_frames ?? 48} steps × 75 keypoints × 4 channels`, status: 'done' },
          { title: '7. Run ONNX V4.3', detail: `${diagnostics.model_name ?? 'VSL-30'} • ${milliseconds('onnx_inference_ms')}`, status: 'done' },
          { title: '8. Filter confidence', detail: `${Math.round(event.payload.confidence * 100)}% • margin ${Math.round((event.payload.margin ?? 0) * 100)}% • total ${milliseconds('total_request_ms')}`, status: 'done' },
        ]);
        setEvent(event);
        const confidence = event.payload.confidence;
        const margin = event.payload.margin ?? 0;
        const handCoverage = event.payload.landmarkCoverage ?? 0;
        // The service applies the same acceptance contract, but the frontend
        // rechecks it so camera and upload results share one visible policy.
        const accepted = event.payload.accepted !== false
          && confidence >= 0.62
          && margin >= 0.1
          && handCoverage >= 0.5;
        if (!accepted) {
          pendingCandidateRef.current = undefined;
          setQualityHint(handCoverage < 0.5
            ? `Hands are not clearly visible (${Math.round(handCoverage * 100)}% of frames). Keep them inside the guide.`
            : `Not confident enough (${Math.round(confidence * 100)}%). Hold the sign steady and try again.`);
          continue;
        }

        const previousCandidate: { label: string; count: number } | undefined =
          pendingCandidateRef.current;
        const candidateCount: number = previousCandidate?.label === event.payload.label
          ? previousCandidate.count + 1
          : 1;
        pendingCandidateRef.current = { label: event.payload.label, count: candidateCount };
        // High-confidence results can commit immediately; weaker results need
        // two consecutive matching clips to avoid flickering gloss tokens.
        const strongPrediction = confidence >= 0.82 && margin >= 0.2;
        const recentlyCommitted = lastCommittedRef.current?.label === event.payload.label
          && Date.now() - lastCommittedRef.current.at < 8_000;

        if ((strongPrediction || candidateCount >= 2) && !recentlyCommitted) {
          const token: SentenceToken = {
            label: event.payload.label,
            confidence,
            margin,
            candidates: event.payload.topK,
          };
          setTokens((previous) =>
            previous.at(-1)?.label === token.label ? previous : [...previous, token].slice(-40),
          );
          lastCommittedRef.current = { label: token.label, at: Date.now() };
          pendingCandidateRef.current = undefined;
          setQualityHint(undefined);
        } else if (recentlyCommitted) {
          setQualityHint(`Skipped repeated word “${event.payload.label}”.`);
        } else {
          setQualityHint(`Confirming “${event.payload.label}”… hold the movement a little longer.`);
        }
      } catch (value) {
        if (runningRef.current) {
          const message = value instanceof Error ? value.message : 'Could not recognize this video clip.';
          setError(message);
          setPipelineLogs((logs) => [
            ...logs.map((log) => ['active', 'pending'].includes(log.status) ? { ...log, status: 'error' as const } : log),
            { title: 'Recognition stopped', detail: message, status: 'error' },
          ]);
        }
        runningRef.current = false;
        setIsRecognizing(false);
      } finally {
        setIsProcessing(false);
      }
    }
  }, [cameraReady, setEvent]);

  const stopRecognition = async () => {
    runningRef.current = false;
    cameraRef.current?.stopRecording();
    setIsRecognizing(false);
    setIsProcessing(false);
    if (!tokens.length) return;
    const rawText = tokens.map((token) => token.label).join(' ');
    setFinalSentence(rawText);
    setQualityHint('Recognized directly by the UNO Q Edge AI.');
    if (speechEnabled && !tokens.every((token) => isMultiVslGloss(token.label))) {
      Speech.stop();
      Speech.speak(rawText, { language: 'vi-VN', rate: 0.88 });
    }
  };

  const uploadAndRecognize = async () => {
    if (runningRef.current || isProcessing || isUploading) return;
    const selection = await DocumentPicker.getDocumentAsync({
      type: 'video/*',
      copyToCacheDirectory: true,
      multiple: false,
    });
    if (selection.canceled || !selection.assets[0]) return;
    const asset = selection.assets[0];
    logRecognitionStep('upload', '1/4', 'Video selected', {
      name: asset.name,
      mimeType: asset.mimeType ?? 'video/*',
      sizeBytes: asset.size,
    });
    setIsUploading(true);
    setIsProcessing(true);
    setError(undefined);
    setQualityHint(undefined);
    setUploadedFileName(asset.name);
    setPipelineLogs([
      { title: '1. Select video', detail: `${asset.name} • ${asset.mimeType ?? 'video'}`, status: 'done' },
      { title: '2. Send to Edge AI', detail: 'Uploading and validating the file…', status: 'pending' },
    ]);
    setFinalSentence(undefined);
    setTokens([]);
    try {
      logRecognitionStep('upload', '2/4', 'Uploading video to Edge AI');
      const result = await recognizeUploadedVideo({
        file: asset.file,
        mimeType: asset.mimeType,
        name: asset.name,
        uri: asset.uri,
      });
      const { diagnostics, event } = result;
      logRuntimeDiagnostics('upload', diagnostics);
      logRecognitionStep('upload', '3/4', 'Edge AI processing complete', {
        inputFrames: diagnostics.input_frames,
        sampledFrames: diagnostics.mediapipe_input_frames ?? diagnostics.sampled_frames,
        model: diagnostics.model_name,
      });
      setEvent(event);
      const confidence = event.payload.confidence;
      const margin = event.payload.margin ?? 0;
      const handCoverage = event.payload.landmarkCoverage ?? 0;
      const acceptedUpload =
        event.payload.accepted !== false
        && confidence >= 0.62
        && margin >= 0.1
        && handCoverage >= 0.5;
      const milliseconds = (key: string) =>
        `${Math.round(Number(diagnostics[key] ?? 0))} ms`;
      setPipelineLogs([
        { title: '1. Read video', detail: `${asset.name} • ${asset.mimeType ?? 'video'}`, status: 'done' },
        {
          title: '2. Decode frames',
          detail: `${diagnostics.input_frames ?? diagnostics.decoded_frames ?? '—'} frames • ${milliseconds('video_decode_ms')}`,
          status: 'done',
        },
        {
          title: '3. Sample frames',
          detail: `${diagnostics.input_frames ?? diagnostics.decoded_frames ?? '—'} → ${diagnostics.mediapipe_input_frames ?? diagnostics.sampled_frames ?? 16} frames`,
          status: 'done',
        },
        {
          title: '4. Extract landmarks',
          detail: '33 pose + 42 hand landmarks • V4.3 keypoint format',
          status: 'done',
        },
        {
          title: '5. Reconstruct and normalize',
          detail: `Interpolate missing joints • center and scale by shoulders • ${milliseconds('mediapipe_preprocess_ms')}`,
          status: 'done',
        },
        {
          title: '6. Prepare tensor',
          detail: `${diagnostics.sampled_frames ?? 48} steps × 75 keypoints × 4 channels`,
          status: 'done',
        },
        {
          title: '7. Run ONNX V4.3',
          detail: `${diagnostics.model_name ?? 'VSL-30'} • softmax over 30 classes • ${milliseconds('onnx_inference_ms')}`,
          status: 'done',
        },
        {
          title: '8. Filter confidence',
          detail: acceptedUpload
            ? `${event.payload.label} • ${Math.round(confidence * 100)}% • margin ${Math.round(margin * 100)}% • total ${milliseconds('total_request_ms')}`
            : `Uncertain • raw top-1: ${event.payload.label} ${Math.round(confidence * 100)}% • margin ${Math.round(margin * 100)}% • total ${milliseconds('total_request_ms')}`,
          status: acceptedUpload ? 'done' : 'error',
        },
      ]);
      if (!acceptedUpload) {
        setQualityHint(
          `Uncertain — “${event.payload.label}” is only a raw top-1 result at ${Math.round(confidence * 100)}%, not an accepted result. Video “${asset.name}” may differ from the VSL2 training data or signing style.`,
        );
        return;
      }
      const token: SentenceToken = {
        label: event.payload.label,
        confidence,
        margin,
        candidates: event.payload.topK,
      };
      setTokens([token]);
      setFinalSentence(displayGloss(token.label));
      setQualityHint(`Recognized video “${asset.name}”.`);
      logRecognitionStep('upload', '4/4', 'Recognition complete', {
        label: event.payload.label,
        confidence: event.payload.confidence,
        accepted: event.payload.accepted,
      });
    } catch (value) {
      const message = value instanceof Error ? value.message : 'Could not recognize the selected video.';
      setError(message);
      setPipelineLogs((logs) => [
        ...logs.map((log) => ['active', 'pending'].includes(log.status) ? { ...log, status: 'error' as const } : log),
        { title: 'Recognition stopped', detail: message, status: 'error' },
      ]);
    } finally {
      setIsProcessing(false);
      setIsUploading(false);
    }
  };

  const rawSentence = useMemo(() => tokens.map((token) => token.label).join(' '), [tokens]);
  const transcript = finalSentence ?? rawSentence;
  const onlyGlossCodes = tokens.length > 0 && tokens.every((token) => isMultiVslGloss(token.label));
  const cameraStatus = isProcessing
    ? 'Analyzing'
    : isRecognizing
      ? 'Listening'
      : 'Ready';
  const speakCurrent = () => {
    if (!transcript) return;
    Speech.stop();
    Speech.speak(transcript, { language: 'vi-VN', rate: 0.88 });
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" />
      <View style={styles.shell}>
        {desktop ? <AppNavigation variant="sidebar" /> : null}
        <ScrollView contentContainerStyle={[styles.page, compact && styles.pageCompact]} showsVerticalScrollIndicator={false}>
          <View style={[styles.header, compact && styles.headerCompact]}>
            <View style={styles.headerIntro}>
              <Text style={styles.kicker}>SIGNTALK AI</Text>
              <Text style={styles.title}>Live translation</Text>
            </View>
            <View style={styles.headerActions}>
              <View style={styles.systemStatus}>
                <View style={styles.onlineDot} />
                <Text style={styles.systemStatusText}>Edge AI ready</Text>
              </View>
            </View>
          </View>

          <View style={[styles.workspace, desktop && styles.workspaceDesktop]}>
            <View style={[styles.cameraColumn, desktop && styles.cameraColumnDesktop]}>
              <View style={[styles.cameraCard, compact && styles.cameraCardCompact]}>
                {cameraAllowed ? (
                  <LiveCamera
                    ref={cameraRef}
                    onCaptureProgress={setCapturedFrames}
                    onReady={handleCameraReady}
                    onError={handleCameraError}
                    style={styles.cameraPreview}
                  />
                ) : (
                  <View style={styles.permissionView}>
                    <View style={styles.permissionIcon}>
                      <Ionicons name="videocam-outline" size={26} color="#E7F7B7" />
                    </View>
                    <Text style={styles.permissionTitle}>Allow camera access</Text>
                    <Text style={styles.permissionCopy}>
                      The camera reads hand movements. Video is sent only to the recognition model.
                    </Text>
                    <Pressable onPress={() => void enableCamera()} style={styles.permissionButton}>
                      <Text style={styles.permissionButtonText}>Enable camera</Text>
                    </Pressable>
                  </View>
                )}

                <View style={styles.cameraOverlay}>
                  <View style={styles.cameraTop}>
                    <View style={styles.liveBadge}>
                      <View style={[styles.liveDot, !isRecognizing && styles.liveDotIdle]} />
                      <Text style={styles.liveText}>{isRecognizing ? 'RECORDING' : 'PREVIEW'}</Text>
                    </View>
                    <Text style={styles.cameraSource}>FRONT CAMERA</Text>
                  </View>
                  <View style={styles.guide}>
                    <View style={[styles.corner, styles.cornerTL]} />
                    <View style={[styles.corner, styles.cornerTR]} />
                    <View style={[styles.corner, styles.cornerBL]} />
                    <View style={[styles.corner, styles.cornerBR]} />
                  </View>
                  <View style={styles.cameraBottom}>
                    <View>
                      <Text style={styles.cameraStatus}>{cameraStatus}</Text>
                    </View>
                    {isProcessing ? <ActivityIndicator color="#E7F7B7" /> : null}
                  </View>
                </View>
              </View>

              {cameraError ? (
                <View style={styles.cameraError}>
                  <Ionicons name="alert-circle-outline" size={18} color="#9A4036" />
                  <Text style={styles.cameraErrorText}>{cameraError}</Text>
                  <Pressable
                    onPress={() => void enableCamera()}
                  >
                    <Text style={styles.retryText}>Retry</Text>
                  </Pressable>
                </View>
              ) : null}

              <View style={styles.controls}>
                <Pressable
                  disabled={!cameraAllowed || !cameraReady}
                  onPress={() => (isRecognizing ? void stopRecognition() : void startRecognition())}
                  style={({ pressed }) => [
                    styles.primaryButton,
                    pressed && styles.buttonPressed,
                    (!cameraAllowed || !cameraReady) && styles.primaryButtonDisabled,
                  ]}
                >
                  <Ionicons name={isRecognizing ? 'stop' : 'play'} size={18} color="#102019" />
                  <Text style={styles.primaryButtonText}>
                    {isRecognizing ? 'Stop translation' : 'Start translation'}
                  </Text>
                </Pressable>
                <Pressable
                  accessibilityLabel={speechEnabled ? 'Disable speech' : 'Enable speech'}
                  onPress={() => setSpeechEnabled((value) => !value)}
                  style={[styles.secondaryButton, speechEnabled && styles.secondaryButtonActive]}
                >
                  <Ionicons
                    name={speechEnabled ? 'volume-high-outline' : 'volume-mute-outline'}
                    size={20}
                    color="#26362F"
                  />
                </Pressable>
              </View>
              <Pressable
                disabled={isRecognizing || isProcessing || isUploading}
                onPress={() => void uploadAndRecognize()}
                style={({ pressed }) => [
                  styles.uploadButton,
                  pressed && styles.buttonPressed,
                  (isRecognizing || isProcessing || isUploading) && styles.uploadButtonDisabled,
                ]}
              >
                {isUploading
                  ? <ActivityIndicator color="#334139" />
                  : <Ionicons name="cloud-upload-outline" size={19} color="#334139" />}
                <View style={styles.uploadCopy}>
                  <Text style={styles.uploadButtonText}>
                    {isUploading ? 'Processing video' : 'Upload video for recognition'}
                  </Text>
                  <Text numberOfLines={1} style={styles.uploadHint}>
                    {uploadedFileName ?? 'MP4, MOV, and WebM supported'}
                  </Text>
                </View>
                <Ionicons name="chevron-forward" size={17} color="#8A948E" />
              </Pressable>
            </View>

            <View style={[styles.resultPanel, desktop && styles.resultPanelDesktop, compact && styles.resultPanelCompact]}>
              <View style={styles.resultTop}>
                <View>
                <Text style={styles.panelLabel}>{onlyGlossCodes ? 'PREDICTED GLOSS CODE' : 'TRANSLATION'}</Text>
                </View>
                <View style={styles.tokenCount}>
                  <Text style={styles.tokenCountText}>{tokens.length} {tokens.length === 1 ? 'word' : 'words'}</Text>
                </View>
              </View>

              <View style={styles.transcriptArea}>
                <Text style={[styles.transcript, !transcript && styles.transcriptEmpty]}>
                  {transcript || 'Recognition results will appear here.'}
                </Text>
              </View>

              {current ? (
                <View style={styles.confidenceSection}>
                  <View style={styles.confidenceHeading}>
                    <Text style={styles.latestLabel}>
                      {current.payload.accepted === false ? 'Uncertain' : displayGloss(current.payload.label)}
                    </Text>
                    <Text style={styles.confidenceNumber}>
                      {confidenceValue(current.payload.confidence)}%
                    </Text>
                  </View>
                  <View style={styles.confidenceTrack}>
                    <View
                      style={[
                        styles.confidenceFill,
                        { width: `${confidenceValue(current.payload.confidence)}%` },
                      ]}
                    />
                  </View>
                  <Text style={styles.confidenceCaption}>
                    {onlyGlossCodes ? 'Gloss code has no display name' : 'Confidence for the latest word'}
                  </Text>
                </View>
              ) : (
                <View style={styles.tip}>
                  <Ionicons name="information-circle-outline" size={18} color="#637169" />
                  <Text style={styles.tipText}>
                    Sign clearly, with a short pause between words.
                  </Text>
                </View>
              )}

              {qualityHint ? (
                <View style={styles.tip}>
                  <Ionicons name="pulse-outline" size={18} color="#637169" />
                  <Text style={styles.tipText}>{qualityHint}</Text>
                </View>
              ) : null}

              {error ? (
                <View style={styles.errorBox}>
                  <Ionicons name="alert-circle-outline" size={18} color="#A13D33" />
                  <Text style={styles.errorText}>{error}</Text>
                </View>
              ) : null}

              <View style={styles.resultActions}>
                <Pressable
                  disabled={!transcript || onlyGlossCodes}
                  onPress={speakCurrent}
                  style={[styles.textButton, (!transcript || onlyGlossCodes) && styles.textButtonDisabled]}
                >
                  <Ionicons name="volume-high-outline" size={18} color="#314039" />
                  <Text style={styles.textButtonLabel}>Read aloud</Text>
                </Pressable>
                <Pressable
                  onPress={() => router.push('/history' as never)}
                  style={styles.textButton}
                >
                  <Ionicons name="time-outline" size={18} color="#314039" />
                  <Text style={styles.textButtonLabel}>View history</Text>
                </Pressable>
              </View>
            </View>
          </View>

          {!desktop ? <AppNavigation /> : null}
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#F6F8F5' },
  shell: { flex: 1, flexDirection: 'row' },
  page: { alignSelf: 'center', gap: 22, maxWidth: 1240, padding: 28, paddingBottom: 42, width: '100%' },
  pageCompact: { gap: 16, padding: 16, paddingBottom: 24 },
  header: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between' },
  headerCompact: { alignItems: 'flex-start' },
  headerIntro: { gap: 4 },
  kicker: { color: '#728078', fontSize: 11, fontWeight: '800', letterSpacing: 1.8 },
  title: { color: '#14221B', fontSize: 32, fontWeight: '800', letterSpacing: -1, marginTop: 2 },
  headerActions: { alignItems: 'center', flexDirection: 'row', gap: 10 },
  systemStatus: { alignItems: 'center', backgroundColor: '#FFFFFF', borderColor: '#DDE6DC', borderRadius: 999, borderWidth: 1, flexDirection: 'row', gap: 8, paddingHorizontal: 13, paddingVertical: 9 },
  onlineDot: { backgroundColor: '#4BA96E', borderRadius: 5, height: 9, width: 9 },
  systemStatusText: { color: '#53645A', fontSize: 12, fontWeight: '700' },
  iconControl: { alignItems: 'center', backgroundColor: '#FFFFFF', borderColor: '#E0E4E0', borderRadius: 10, borderWidth: 1, height: 40, justifyContent: 'center', width: 40 },
  workspace: { gap: 20 },
  workspaceDesktop: { alignItems: 'stretch', flexDirection: 'row' },
  cameraColumn: { gap: 14 },
  cameraColumnDesktop: { flex: 1.45 },
  cameraCard: { backgroundColor: '#0E2018', borderRadius: 22, height: 500, overflow: 'hidden', position: 'relative' },
  cameraCardCompact: { borderRadius: 18, height: 350 },
  cameraPreview: { height: '100%', width: '100%' },
  permissionView: { alignItems: 'center', flex: 1, justifyContent: 'center', padding: 28 },
  permissionIcon: { alignItems: 'center', backgroundColor: '#172D23', borderColor: '#3D5949', borderRadius: 16, borderWidth: 1, height: 58, justifyContent: 'center', marginBottom: 17, width: 58 },
  permissionTitle: { color: '#FFFFFF', fontSize: 18, fontWeight: '700' },
  permissionCopy: { color: '#AEB8B2', fontSize: 13, lineHeight: 19, marginTop: 8, maxWidth: 360, textAlign: 'center' },
  permissionButton: { backgroundColor: '#DDF5A6', borderRadius: 12, marginTop: 20, paddingHorizontal: 20, paddingVertical: 13 },
  permissionButtonText: { color: '#16211B', fontSize: 13, fontWeight: '700' },
  cameraOverlay: { bottom: 0, left: 0, padding: 20, pointerEvents: 'none', position: 'absolute', right: 0, top: 0 },
  cameraTop: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between' },
  liveBadge: { alignItems: 'center', backgroundColor: 'rgba(6,18,12,.76)', borderColor: 'rgba(221,245,166,.18)', borderRadius: 999, borderWidth: 1, flexDirection: 'row', gap: 7, paddingHorizontal: 11, paddingVertical: 7 },
  liveDot: { backgroundColor: '#F16A5B', borderRadius: 4, height: 7, width: 7 },
  liveDotIdle: { backgroundColor: '#8E9993' },
  liveText: { color: '#FFFFFF', fontSize: 10, fontWeight: '700', letterSpacing: 0.8 },
  cameraSource: { color: '#E1E7E3', fontSize: 12, fontWeight: '600', textShadow: '0px 0px 4px #000000' } as TextStyle,
  guide: { alignSelf: 'center', aspectRatio: 4 / 3, height: '70%', marginVertical: 26, position: 'relative' },
  corner: { borderColor: 'rgba(231,247,183,.82)', height: 44, position: 'absolute', width: 44 },
  cornerTL: { borderLeftWidth: 2, borderTopWidth: 2, left: 0, top: 0 },
  cornerTR: { borderRightWidth: 2, borderTopWidth: 2, right: 0, top: 0 },
  cornerBL: { borderBottomWidth: 2, borderLeftWidth: 2, bottom: 0, left: 0 },
  cornerBR: { borderBottomWidth: 2, borderRightWidth: 2, bottom: 0, right: 0 },
  cameraBottom: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between' },
  cameraStatus: { color: '#FFFFFF', fontSize: 13, fontWeight: '700' },
  cameraError: { alignItems: 'center', backgroundColor: '#FFF3F1', borderColor: '#F0D3CF', borderRadius: 9, borderWidth: 1, flexDirection: 'row', gap: 8, padding: 11 },
  cameraErrorText: { color: '#843A32', flex: 1, fontSize: 11, lineHeight: 16 },
  retryText: { color: '#843A32', fontSize: 11, fontWeight: '700' },
  controls: { flexDirection: 'row', gap: 10 },
  primaryButton: { alignItems: 'center', backgroundColor: '#DDF5A6', borderRadius: 14, flex: 1, flexDirection: 'row', gap: 9, justifyContent: 'center', minHeight: 58 },
  primaryButtonDisabled: { backgroundColor: '#DCE1D9' },
  primaryButtonText: { color: '#102019', fontSize: 14, fontWeight: '700' },
  secondaryButton: { alignItems: 'center', backgroundColor: '#FFFFFF', borderColor: '#DCE5DB', borderRadius: 14, borderWidth: 1, height: 58, justifyContent: 'center', width: 58 },
  secondaryButtonActive: { backgroundColor: '#F1F8DE', borderColor: '#BBD477' },
  uploadButton: { alignItems: 'center', backgroundColor: '#FFFFFF', borderColor: '#DCE5DB', borderRadius: 14, borderWidth: 1, flexDirection: 'row', gap: 12, minHeight: 68, paddingHorizontal: 16 },
  uploadButtonDisabled: { opacity: 0.5 },
  uploadCopy: { flex: 1, gap: 3 },
  uploadButtonText: { color: '#24352B', fontSize: 13, fontWeight: '800' },
  uploadHint: { color: '#849289', fontSize: 11 },
  processedVideoCard: { backgroundColor: '#FFFFFF', borderColor: '#DDE2DD', borderRadius: 11, borderWidth: 1, overflow: 'hidden' },
  processedVideoHeader: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', padding: 14 },
  processedVideoTitle: { color: '#2A3830', fontSize: 12, fontWeight: '700' },
  processedVideoCaption: { color: '#89938D', fontSize: 10, marginTop: 3, maxWidth: 280 },
  processedVideo: { aspectRatio: 16 / 9, backgroundColor: '#13201A', width: '100%' },
  pipelinePanel: { backgroundColor: '#FFFFFF', borderColor: '#DDE2DD', borderRadius: 11, borderWidth: 1, overflow: 'hidden', paddingHorizontal: 14 },
  pipelineHeader: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 13 },
  pipelineTitle: { color: '#2A3830', fontSize: 12, fontWeight: '700' },
  pipelineSubtitle: { color: '#8A948E', fontSize: 10, marginTop: 2 },
  pipelineToggle: { alignItems: 'center', flexDirection: 'row', gap: 5, paddingHorizontal: 4, paddingVertical: 6 },
  pipelineToggleText: { color: '#657168', fontSize: 11, fontWeight: '700' },
  pipelineCollapsedHint: { borderTopColor: '#EDF0ED', borderTopWidth: 1, color: '#818B85', fontSize: 10, paddingVertical: 12 },
  pipelineRow: { alignItems: 'center', flexDirection: 'row', gap: 10, minHeight: 52 },
  pipelineRowBorder: { borderTopColor: '#EDF0ED', borderTopWidth: 1 },
  pipelineDot: { backgroundColor: '#D8DDD9', borderRadius: 4, height: 8, width: 8 },
  pipelineDotActive: { backgroundColor: '#D4A843' }, pipelineDotDone: { backgroundColor: '#7E9845' },
  pipelineDotError: { backgroundColor: '#C35A4D' },
  pipelineCopy: { flex: 1, gap: 2 },
  pipelineStep: { color: '#344139', fontSize: 11, fontWeight: '700' },
  pipelineDetail: { color: '#818B85', fontSize: 10 },
  buttonPressed: { opacity: 0.82 },
  resultPanel: { backgroundColor: '#FFFFFF', borderColor: '#DFE8DE', borderRadius: 22, borderWidth: 1, minHeight: 500, padding: 24 },
  resultPanelDesktop: { flex: 1, minWidth: 350 },
  resultPanelCompact: { borderRadius: 18, minHeight: 340, padding: 18 },
  resultTop: { alignItems: 'flex-start', flexDirection: 'row', justifyContent: 'space-between' },
  panelLabel: { color: '#738279', fontSize: 11, fontWeight: '800', letterSpacing: 1.5 },
  tokenCount: { backgroundColor: '#F1F5EC', borderRadius: 999, paddingHorizontal: 10, paddingVertical: 7 },
  tokenCountText: { color: '#64745F', fontSize: 11, fontWeight: '800' },
  transcriptArea: { flex: 1, justifyContent: 'center', minHeight: 175, paddingVertical: 26 },
  transcript: { color: '#15241B', fontSize: 31, fontWeight: '700', letterSpacing: -0.8, lineHeight: 40 },
  transcriptEmpty: { color: '#A7B0A9', fontSize: 23, fontWeight: '400', lineHeight: 32 },
  rawGloss: { color: '#808A83', fontSize: 12, lineHeight: 18, marginTop: 12 },
  resultDetailsToggle: { alignItems: 'center', alignSelf: 'flex-start', flexDirection: 'row', gap: 6, paddingBottom: 9, paddingTop: 3 },
  resultDetailsToggleText: { color: '#657168', fontSize: 11, fontWeight: '700' },
  confidenceSection: { borderTopColor: '#E8EEE6', borderTopWidth: 1, paddingTop: 18 },
  confidenceHeading: { flexDirection: 'row', justifyContent: 'space-between' },
  latestLabel: { color: '#314039', fontSize: 13, fontWeight: '700' },
  confidenceNumber: { color: '#536158', fontSize: 12, fontWeight: '700' },
  confidenceTrack: { backgroundColor: '#EAF0E7', borderRadius: 999, height: 7, marginTop: 10, overflow: 'hidden' },
  confidenceFill: { backgroundColor: '#86A94A', borderRadius: 999, height: '100%' },
  confidenceCaption: { color: '#8B978E', fontSize: 11, marginTop: 8 },
  candidatesCard: { backgroundColor: '#F7F9F4', borderColor: '#E4EAE0', borderRadius: 12, borderWidth: 1, marginTop: 16, padding: 14 },
  candidatesHeading: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', marginBottom: 9 },
  candidatesTitle: { color: '#405047', fontSize: 11, fontWeight: '700' },
  candidatesMeta: { color: '#89938D', fontSize: 10 },
  candidateRow: { alignItems: 'center', flexDirection: 'row', gap: 8, minHeight: 29 },
  candidateRank: { color: '#849075', fontSize: 10, fontWeight: '700', width: 18 },
  candidateLabel: { color: '#46544B', flex: 1, fontSize: 11, fontWeight: '600' },
  candidateConfidence: { color: '#65744D', fontSize: 11, fontWeight: '700' },
  tip: { alignItems: 'flex-start', backgroundColor: '#F2F6EF', borderColor: '#E4ECE1', borderRadius: 12, borderWidth: 1, flexDirection: 'row', gap: 9, marginTop: 14, padding: 14 },
  tipText: { color: '#607067', flex: 1, fontSize: 12, lineHeight: 18 },
  errorBox: { alignItems: 'flex-start', backgroundColor: '#FFF3F1', borderColor: '#F0D7D1', borderRadius: 12, borderWidth: 1, flexDirection: 'row', gap: 8, marginTop: 12, padding: 13 },
  errorText: { color: '#8F3A32', flex: 1, fontSize: 12, lineHeight: 18 },
  resultActions: { borderTopColor: '#E8EEE6', borderTopWidth: 1, flexDirection: 'row', gap: 10, marginTop: 18, paddingTop: 18 },
  textButton: { alignItems: 'center', borderColor: '#DCE5DB', borderRadius: 12, borderWidth: 1, flex: 1, flexDirection: 'row', gap: 7, justifyContent: 'center', minHeight: 46 },
  textButtonDisabled: { opacity: 0.4 },
  textButtonLabel: { color: '#314039', fontSize: 12, fontWeight: '700' },
  recentSection: { gap: 12 },
  sectionHeader: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between' },
  sectionTitle: { color: '#25332C', fontSize: 16, fontWeight: '700' },
  sectionLink: { color: '#5F7040', fontSize: 12, fontWeight: '700' },
  recentList: { backgroundColor: '#FFFFFF', borderColor: '#E0E4E0', borderRadius: 13, borderWidth: 1, paddingHorizontal: 16 },
  recentRow: { alignItems: 'center', flexDirection: 'row', gap: 12, minHeight: 68 },
  recentRowBorder: { borderTopColor: '#EBEEEB', borderTopWidth: 1 },
  recentIcon: { alignItems: 'center', backgroundColor: '#F0F2EE', borderRadius: 9, height: 38, justifyContent: 'center', width: 38 },
  recentCopy: { flex: 1, gap: 3 },
  recentText: { color: '#29372F', fontSize: 13, fontWeight: '700' },
  recentMeta: { color: '#858F89', fontSize: 11 },
  emptyRecent: { alignItems: 'center', minHeight: 76, justifyContent: 'center' },
  emptyRecentText: { color: '#89928D', fontSize: 12 },
});
