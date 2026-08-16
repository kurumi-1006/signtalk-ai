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
  useWindowDimensions,
  View,
} from 'react-native';
import { useCameraPermissions } from 'expo-camera';
import * as DocumentPicker from 'expo-document-picker';
import { Ionicons } from '@expo/vector-icons';
import * as Speech from 'expo-speech';
import { useVideoPlayer, VideoView } from 'expo-video';
import { router } from 'expo-router';
import type { SentenceToken } from '@signtalk/contracts';
import {
  recognizeUploadedVideo,
  recognizeVideo,
  getEdgeModels,
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

const VSL30_CAPTURE = { maxDuration: 3, maxFrames: 56 };

const confidenceValue = (value?: number) =>
  Math.round(Math.max(0, Math.min(value ?? 0, 1)) * 100);

const isMultiVslGloss = (label?: string) => /^vsl_gloss_\d{4}$/.test(label ?? '');
const displayGloss = (label?: string) => isMultiVslGloss(label)
  ? `Gloss #${label?.slice(-4)}`
  : label ?? '—';

const logRuntimeDiagnostics = (source: 'camera' | 'upload', diagnostics: PipelineDiagnostics) => {
  console.log('[SignTalk Edge AI] Thống kê xử lý', {
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

function ProcessedVideo({ uri }: { uri: string }) {
  const player = useVideoPlayer(uri, (instance) => {
    instance.loop = false;
  });

  return <VideoView allowsFullscreen nativeControls player={player} style={styles.processedVideo} />;
}

export default function Home() {
  const { width } = useWindowDimensions();
  const desktop = width >= 900;
  const isWeb = Platform.OS === 'web';
  const cameraRef = useRef<CameraHandle>(null);
  const runningRef = useRef(false);
  const pendingCandidateRef = useRef<{ label: string; count: number } | undefined>(undefined);
  const lastCommittedRef = useRef<{ label: string; at: number } | undefined>(undefined);
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraReady, setCameraReady] = useState(false);
  const [webCameraEnabled, setWebCameraEnabled] = useState(false);
  const [cameraError, setCameraError] = useState<string>();
  const [capturedFrames, setCapturedFrames] = useState(0);
  const [activeModelId, setActiveModelId] = useState<string>();
  const [qualityHint, setQualityHint] = useState<string>();
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState<string>();
  const [processedVideoUri, setProcessedVideoUri] = useState<string>();
  const [pipelineLogs, setPipelineLogs] = useState<PipelineLog[]>([]);
  const [pipelineCollapsed, setPipelineCollapsed] = useState(false);
  const [resultDetailsCollapsed, setResultDetailsCollapsed] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [tokens, setTokens] = useState<SentenceToken[]>([]);
  const [finalSentence, setFinalSentence] = useState<string>();
  const [error, setError] = useState<string>();
  const { current, events, setEvent } = useRecognitionStore();
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

  useEffect(() => {
    getEdgeModels().then((result) => setActiveModelId(result.active_model_id)).catch(() => undefined);
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
      { title: '1. Ghi chuyển động', detail: 'Đang chuẩn bị chuỗi 56 frame cho VSL-30…', status: 'active' },
      { title: '2. Gửi Edge AI', detail: 'Gửi tối đa 56 JPEG trong 3 giây.', status: 'pending' },
      { title: '3. Trích xuất keypoint', detail: 'MediaPipe Holistic đọc 33 pose + 42 điểm tay.', status: 'pending' },
      { title: '4. Chuẩn hoá chuỗi', detail: '48 time-step • 75 keypoint • chuẩn hoá theo vai.', status: 'pending' },
      { title: '5. VSL-30 suy luận', detail: 'Softmax classifier • 30 gloss tiếng Việt.', status: 'pending' },
      { title: '6. Kiểm tra độ tin cậy', detail: 'Đối chiếu confidence và khoảng cách top-2.', status: 'pending' },
    ]);

    while (runningRef.current) {
      try {
        setPipelineLogs((logs) => logs.map((log, index) => index === 0
          ? { ...log, detail: 'Đang ghi chuyển động tay trong 3 giây…', status: 'active' }
          : { ...log, status: 'pending' }));
        setCapturedFrames(0);
        const clip = await cameraRef.current.recordAsync(VSL30_CAPTURE);
        if (!clip || !runningRef.current) continue;
        setIsProcessing(true);
        setPipelineLogs((logs) => logs.map((log, index) => {
          if (index === 0) return { ...log, detail: `Đã ghi ${'frames' in clip ? clip.frames.length : 'video'} nguồn.`, status: 'done' };
          if (index === 1) return { ...log, detail: 'Đang tải clip tới Edge AI…', status: 'active' };
          return log;
        }));
        const { event, diagnostics } = await recognizeVideo(clip);
        logRuntimeDiagnostics('camera', diagnostics);
        const milliseconds = (key: string) => `${Math.round(Number(diagnostics[key] ?? 0))} ms`;
        setPipelineLogs([
          { title: '1. Ghi chuyển động', detail: `${diagnostics.received_frames ?? diagnostics.input_frames ?? '—'} frame nguồn đã sẵn sàng.`, status: 'done' },
          { title: '2. Gửi Edge AI', detail: `${diagnostics.input_mode === 'jpeg_frames' ? 'JPEG frames' : 'Video clip'} • ${milliseconds('jpeg_decode_ms')}`, status: 'done' },
          { title: '3. Trích xuất keypoint', detail: `${diagnostics.input_frames ?? diagnostics.decoded_frames ?? '—'} → ${diagnostics.mediapipe_input_frames ?? '—'} frame MediaPipe`, status: 'done' },
          { title: '4. Chuẩn hoá chuỗi', detail: `${diagnostics.sampled_frames ?? 48} time-step • 75 keypoint • theo vai`, status: 'done' },
          { title: '5. VSL-30 suy luận', detail: `${diagnostics.model_name ?? 'VSL-30'} • softmax 30 gloss`, status: 'done' },
          { title: '6. Kiểm tra độ tin cậy', detail: `${Math.round(event.payload.confidence * 100)}% • margin ${Math.round((event.payload.margin ?? 0) * 100)}% • ${milliseconds('total_request_ms')}`, status: 'done' },
        ]);
        setEvent(event);
        const confidence = event.payload.confidence;
        const margin = event.payload.margin ?? 0;
        const handCoverage = event.payload.landmarkCoverage ?? 0;
        const accepted = event.payload.accepted !== false
          && confidence >= 0.62
          && margin >= 0.1
          && handCoverage >= 0.5;
        if (!accepted) {
          pendingCandidateRef.current = undefined;
          setQualityHint(handCoverage < 0.5
            ? `Không thấy rõ bàn tay (${Math.round(handCoverage * 100)}% khung hình). Hãy đưa tay vào khung vuông.`
            : `Chưa đủ chắc chắn (${Math.round(confidence * 100)}%). Hãy giữ ký hiệu ổn định và thử lại.`);
          continue;
        }

        const previousCandidate: { label: string; count: number } | undefined =
          pendingCandidateRef.current;
        const candidateCount: number = previousCandidate?.label === event.payload.label
          ? previousCandidate.count + 1
          : 1;
        pendingCandidateRef.current = { label: event.payload.label, count: candidateCount };
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
          setQualityHint(`Đã bỏ qua từ lặp “${event.payload.label}”.`);
        } else {
          setQualityHint(`Đang xác nhận “${event.payload.label}”… giữ động tác thêm một nhịp.`);
        }
      } catch (value) {
        if (runningRef.current) {
          const message = value instanceof Error ? value.message : 'Không thể nhận diện đoạn video này.';
          setError(message);
          setPipelineLogs((logs) => [
            ...logs.map((log) => ['active', 'pending'].includes(log.status) ? { ...log, status: 'error' as const } : log),
            { title: 'Pipeline dừng', detail: message, status: 'error' },
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
    setQualityHint('Kết quả được nhận diện trực tiếp trên UNO Q.');
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
    setIsUploading(true);
    setIsProcessing(true);
    setError(undefined);
    setQualityHint(undefined);
    setUploadedFileName(asset.name);
    setProcessedVideoUri(asset.uri);
    setPipelineLogs([
      { title: '1. Chọn video', detail: `${asset.name} • ${asset.mimeType ?? 'video'}`, status: 'done' },
      { title: '2. Gửi tới Edge AI', detail: 'Đang tải file và kiểm tra định dạng…', status: 'pending' },
    ]);
    setFinalSentence(undefined);
    setTokens([]);
    try {
      const result = await recognizeUploadedVideo({
        file: asset.file,
        mimeType: asset.mimeType,
        name: asset.name,
        uri: asset.uri,
      });
      const { diagnostics, event } = result;
      logRuntimeDiagnostics('upload', diagnostics);
      setEvent(event);
      const confidence = event.payload.confidence;
      const margin = event.payload.margin ?? 0;
      const handCoverage = event.payload.landmarkCoverage ?? 0;
      const acceptedUpload =
        event.payload.accepted !== false
        && confidence >= 0.62
        && margin >= 0.1
        && handCoverage >= 0.5;
      const topCandidates = (event.payload.topK ?? [])
        .slice(0, 3)
        .map((candidate) => `${candidate.label} ${Math.round(candidate.confidence * 100)}%`)
        .join(' · ');
      const milliseconds = (key: string) =>
        `${Math.round(Number(diagnostics[key] ?? 0))} ms`;
      setPipelineLogs([
        { title: '1. Chọn video', detail: `${asset.name} • ${asset.mimeType ?? 'video'}`, status: 'done' },
        {
          title: '2. Giải mã',
          detail: `${diagnostics.input_frames ?? diagnostics.decoded_frames ?? '—'} frame • ${milliseconds('video_decode_ms')}`,
          status: 'done',
        },
        {
          title: '3. Lấy mẫu video',
          detail: `${diagnostics.input_frames ?? diagnostics.decoded_frames ?? '—'} → ${diagnostics.mediapipe_input_frames ?? diagnostics.sampled_frames ?? 16} frame`,
          status: 'done',
        },
        {
          title: '4. Chuẩn hoá video',
          detail: 'Pose 33 + hai tay 42 • chuẩn hoá theo vai',
          status: 'done',
        },
        {
          title: '5. Chuỗi keypoint',
          detail: `${diagnostics.sampled_frames ?? 48}×75×4 • ${milliseconds('mediapipe_preprocess_ms')}`,
          status: 'done',
        },
        {
          title: '6. VSL-30 classifier',
          detail: `${diagnostics.model_name ?? 'VSL-30'} • softmax 30 gloss • ${milliseconds('onnx_inference_ms')}`,
          status: 'done',
        },
        {
          title: '7. Quyết định',
          detail: acceptedUpload
            ? `${event.payload.label} • ${Math.round(confidence * 100)}% • margin ${Math.round(margin * 100)}% • tổng ${milliseconds('total_request_ms')}`
            : `Không xác định • top-1 thô: ${event.payload.label} ${Math.round(confidence * 100)}% • margin ${Math.round(margin * 100)}% • tổng ${milliseconds('total_request_ms')}`,
          status: acceptedUpload ? 'done' : 'error',
        },
        {
          title: '8. Top-3 thô (chỉ tham khảo)',
          detail: topCandidates || 'Model không trả về danh sách ứng viên.',
          status: 'done',
        },
      ]);
      if (!acceptedUpload) {
        setQualityHint(
          `Không xác định — “${event.payload.label}” chỉ là top-1 thô ${Math.round(confidence * 100)}%, không phải kết quả được chấp nhận. Video “${asset.name}” có thể khác miền dữ liệu VSL2 hoặc cách thực hiện ký hiệu chưa khớp mẫu train.`,
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
      setQualityHint(`Đã nhận diện video “${asset.name}”.`);
    } catch (value) {
      const message = value instanceof Error ? value.message : 'Không thể nhận diện video đã chọn.';
      setError(message);
      setPipelineLogs((logs) => [
        ...logs.map((log) => ['active', 'pending'].includes(log.status) ? { ...log, status: 'error' as const } : log),
        { title: 'Pipeline dừng', detail: message, status: 'error' },
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
    ? 'Đang phân tích'
    : isRecognizing
      ? 'Đang lắng nghe'
      : 'Sẵn sàng';
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
        <ScrollView contentContainerStyle={styles.page} showsVerticalScrollIndicator={false}>
          <View style={styles.header}>
            <View>
              <Text style={styles.kicker}>SIGNTALK • UNO Q EDGE AI</Text>
              <Text style={styles.title}>Phiên dịch trực tiếp</Text>
              <Text style={styles.subtitle}>Ngôn ngữ ký hiệu thành văn bản và giọng nói.</Text>
            </View>
            <View style={styles.headerActions}>
              <View style={styles.systemStatus}>
                <View style={styles.onlineDot} />
                <Text style={styles.systemStatusText}>Hệ thống sẵn sàng</Text>
              </View>
              <Pressable
                accessibilityLabel="Mở cài đặt"
                onPress={() => router.push('/settings')}
                style={styles.iconControl}
              >
                <Ionicons name="settings-outline" size={20} color="#34433C" />
              </Pressable>
            </View>
          </View>

          <View style={[styles.workspace, desktop && styles.workspaceDesktop]}>
            <View style={[styles.cameraColumn, desktop && styles.cameraColumnDesktop]}>
              <View style={styles.cameraCard}>
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
                    <Text style={styles.permissionTitle}>Cho phép truy cập camera</Text>
                    <Text style={styles.permissionCopy}>
                      Camera được dùng để đọc chuyển động tay. Video chỉ được gửi tới mô hình nhận diện.
                    </Text>
                    <Pressable onPress={() => void enableCamera()} style={styles.permissionButton}>
                      <Text style={styles.permissionButtonText}>Bật camera</Text>
                    </Pressable>
                  </View>
                )}

                <View pointerEvents="none" style={styles.cameraOverlay}>
                  <View style={styles.cameraTop}>
                    <View style={styles.liveBadge}>
                      <View style={[styles.liveDot, !isRecognizing && styles.liveDotIdle]} />
                      <Text style={styles.liveText}>{isRecognizing ? 'ĐANG GHI' : 'XEM TRƯỚC'}</Text>
                    </View>
                    <Text style={styles.cameraSource}>VSL-30 • camera trước</Text>
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
                    <Text style={styles.retryText}>Thử lại</Text>
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
                    {isRecognizing ? 'Kết thúc câu' : 'Bắt đầu phiên dịch'}
                  </Text>
                </Pressable>
                <Pressable
                  accessibilityLabel={speechEnabled ? 'Tắt đọc thành tiếng' : 'Bật đọc thành tiếng'}
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
                    {isUploading ? 'Đang xử lý video' : 'Tải video lên để nhận diện'}
                  </Text>
                  <Text numberOfLines={1} style={styles.uploadHint}>
                    {uploadedFileName ?? 'Hỗ trợ MP4, MOV và WebM'}
                  </Text>
                </View>
                <Ionicons name="chevron-forward" size={17} color="#8A948E" />
              </Pressable>
              {processedVideoUri ? (
                <View style={styles.processedVideoCard}>
                  <View style={styles.processedVideoHeader}>
                    <View>
                      <Text style={styles.processedVideoTitle}>Video đã xử lý</Text>
                      <Text numberOfLines={1} style={styles.processedVideoCaption}>
                        {uploadedFileName ?? 'Video đã tải lên'}
                      </Text>
                    </View>
                    <Ionicons name="checkmark-circle" size={19} color="#6F8D31" />
                  </View>
                  <ProcessedVideo uri={processedVideoUri} />
                </View>
              ) : null}
              {pipelineLogs.length ? (
                <View style={styles.pipelinePanel}>
                  <View style={styles.pipelineHeader}>
                    <View>
                      <Text style={styles.pipelineTitle}>Tiến trình UNO Q Edge AI</Text>
                      <Text style={styles.pipelineSubtitle}>VSL-30 • 75 keypoint • 30 gloss tiếng Việt</Text>
                    </View>
                    <Pressable
                      accessibilityLabel={pipelineCollapsed ? 'Hiện tiến trình' : 'Ẩn tiến trình'}
                      onPress={() => setPipelineCollapsed((collapsed) => !collapsed)}
                      style={styles.pipelineToggle}
                    >
                      <Ionicons name={pipelineCollapsed ? 'eye-outline' : 'eye-off-outline'} size={16} color="#657168" />
                      <Text style={styles.pipelineToggleText}>{pipelineCollapsed ? 'Hiện' : 'Ẩn'}</Text>
                    </Pressable>
                  </View>
                  {pipelineCollapsed ? (
                    <Text style={styles.pipelineCollapsedHint}>Tiến trình đang được ẩn. Nhấn “Hiện” để xem chi tiết.</Text>
                  ) : pipelineLogs.map((log, index) => (
                    <View key={`${log.title}-${index}`} style={[styles.pipelineRow, index > 0 && styles.pipelineRowBorder]}>
                      <View style={[
                        styles.pipelineDot,
                        log.status === 'active' && styles.pipelineDotActive,
                        log.status === 'done' && styles.pipelineDotDone,
                        log.status === 'error' && styles.pipelineDotError,
                      ]} />
                      <View style={styles.pipelineCopy}>
                        <Text style={styles.pipelineStep}>{log.title}</Text>
                        <Text style={styles.pipelineDetail}>{log.detail}</Text>
                      </View>
                      {log.status === 'active' ? <ActivityIndicator size="small" color="#6C8731" /> : null}
                      {log.status === 'done' ? <Ionicons name="checkmark-circle" size={18} color="#6F8D31" /> : null}
                      {log.status === 'error' ? <Ionicons name="alert-circle" size={18} color="#C35A4D" /> : null}
                      {log.status === 'pending' ? <Ionicons name="ellipse-outline" size={16} color="#B0B8B2" /> : null}
                    </View>
                  ))}
                </View>
              ) : null}
            </View>

            <View style={[styles.resultPanel, desktop && styles.resultPanelDesktop]}>
              <View style={styles.resultTop}>
                <View>
                  <Text style={styles.panelLabel}>{onlyGlossCodes ? 'MÃ GLOSS DỰ ĐOÁN' : 'BẢN DỊCH'}</Text>
                </View>
                <View style={styles.tokenCount}>
                  <Text style={styles.tokenCountText}>{tokens.length} từ</Text>
                </View>
              </View>

              <View style={styles.transcriptArea}>
                <Text style={[styles.transcript, !transcript && styles.transcriptEmpty]}>
                  {transcript || 'Kết quả nhận diện sẽ xuất hiện tại đây.'}
                </Text>
                {finalSentence && rawSentence ? (
                  <Text style={styles.rawGloss}>Chuỗi ký hiệu: {rawSentence}</Text>
                ) : null}
              </View>

              {current ? (
                <>
                  <Pressable
                    accessibilityLabel={resultDetailsCollapsed ? 'Hiện chi tiết kết quả' : 'Ẩn chi tiết kết quả'}
                    onPress={() => setResultDetailsCollapsed((collapsed) => !collapsed)}
                    style={styles.resultDetailsToggle}
                  >
                    <Ionicons name={resultDetailsCollapsed ? 'eye-outline' : 'eye-off-outline'} size={16} color="#657168" />
                    <Text style={styles.resultDetailsToggleText}>
                      {resultDetailsCollapsed ? 'Hiện chi tiết độ tin cậy' : 'Ẩn chi tiết độ tin cậy'}
                    </Text>
                  </Pressable>
                  {!resultDetailsCollapsed ? <>
                <View style={styles.confidenceSection}>
                  <View style={styles.confidenceHeading}>
                    <Text style={styles.latestLabel}>
                      {current.payload.accepted === false ? 'Không xác định' : displayGloss(current.payload.label)}
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
                    {onlyGlossCodes ? 'Mã gloss chưa có bảng tên' : 'Độ tin cậy của từ gần nhất'}
                  </Text>
                </View>
                  {current.payload.topK?.length ? (
                    <View style={styles.candidatesCard}>
                      <View style={styles.candidatesHeading}>
                        <Text style={styles.candidatesTitle}>Top-3 thô (chỉ tham khảo)</Text>
                        <Text style={styles.candidatesMeta}>{activeModelId === 'vsl30_keypoint_classifier' ? 'VSL-30 classifier' : 'UNO Q Edge AI'}</Text>
                      </View>
                      {current.payload.topK.slice(0, 3).map((candidate, index) => (
                        <View key={candidate.label} style={styles.candidateRow}>
                          <Text style={styles.candidateRank}>#{index + 1}</Text>
                          <Text numberOfLines={1} style={styles.candidateLabel}>{candidate.label}</Text>
                          <Text style={styles.candidateConfidence}>{confidenceValue(candidate.confidence)}%</Text>
                        </View>
                      ))}
                    </View>
                  ) : null}
                  </> : null}
                </>
              ) : (
                <View style={styles.tip}>
                  <Ionicons name="information-circle-outline" size={18} color="#637169" />
                  <Text style={styles.tipText}>
                    Thực hiện từng ký hiệu rõ ràng, với khoảng nghỉ ngắn giữa các từ.
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
                  <Text style={styles.textButtonLabel}>Đọc câu</Text>
                </Pressable>
                <Pressable
                  onPress={() => router.push('/history' as never)}
                  style={styles.textButton}
                >
                  <Ionicons name="time-outline" size={18} color="#314039" />
                  <Text style={styles.textButtonLabel}>Xem lịch sử</Text>
                </Pressable>
              </View>
            </View>
          </View>

          <View style={styles.recentSection}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Hoạt động gần đây</Text>
              <Pressable onPress={() => router.push('/history' as never)}>
                <Text style={styles.sectionLink}>Xem tất cả</Text>
              </Pressable>
            </View>
            <View style={styles.recentList}>
              {events.length ? (
                events.slice(0, 3).map((event, index) => (
                  <View key={event.eventId} style={[styles.recentRow, index > 0 && styles.recentRowBorder]}>
                    <View style={styles.recentIcon}>
                      <Ionicons name="hand-left-outline" size={18} color="#405047" />
                    </View>
                    <View style={styles.recentCopy}>
                      <Text numberOfLines={1} style={styles.recentText}>{event.payload.text}</Text>
                      <Text style={styles.recentMeta}>
                        {event.payload.label} • {confidenceValue(event.payload.confidence)}% tin cậy
                      </Text>
                    </View>
                    <Ionicons name="chevron-forward" size={17} color="#A0A9A4" />
                  </View>
                ))
              ) : (
                <View style={styles.emptyRecent}>
                  <Text style={styles.emptyRecentText}>Chưa có hoạt động nhận diện.</Text>
                </View>
              )}
            </View>
          </View>
          {!desktop ? <AppNavigation /> : null}
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#F4F5F2' },
  shell: { flex: 1, flexDirection: 'row' },
  page: { alignSelf: 'center', gap: 24, maxWidth: 1180, padding: 24, paddingBottom: 38, width: '100%' },
  header: { alignItems: 'flex-start', flexDirection: 'row', justifyContent: 'space-between' },
  kicker: { color: '#657168', fontSize: 11, fontWeight: '700', letterSpacing: 1.2 },
  title: { color: '#17231D', fontSize: 28, fontWeight: '700', letterSpacing: -0.6, marginTop: 6 },
  subtitle: { color: '#707A74', fontSize: 14, marginTop: 6 },
  headerActions: { alignItems: 'center', flexDirection: 'row', gap: 10 },
  systemStatus: { alignItems: 'center', backgroundColor: '#FFFFFF', borderColor: '#E0E4E0', borderRadius: 10, borderWidth: 1, flexDirection: 'row', gap: 8, paddingHorizontal: 12, paddingVertical: 10 },
  onlineDot: { backgroundColor: '#42A56B', borderRadius: 4, height: 8, width: 8 },
  systemStatusText: { color: '#526058', fontSize: 12, fontWeight: '600' },
  iconControl: { alignItems: 'center', backgroundColor: '#FFFFFF', borderColor: '#E0E4E0', borderRadius: 10, borderWidth: 1, height: 40, justifyContent: 'center', width: 40 },
  workspace: { gap: 16 },
  workspaceDesktop: { alignItems: 'stretch', flexDirection: 'row' },
  cameraColumn: { gap: 12 },
  cameraColumnDesktop: { flex: 1.55 },
  cameraCard: { backgroundColor: '#13201A', borderRadius: 16, height: 430, overflow: 'hidden', position: 'relative' },
  cameraPreview: { height: '100%', width: '100%' },
  permissionView: { alignItems: 'center', flex: 1, justifyContent: 'center', padding: 32 },
  permissionIcon: { alignItems: 'center', borderColor: '#425149', borderRadius: 14, borderWidth: 1, height: 54, justifyContent: 'center', marginBottom: 16, width: 54 },
  permissionTitle: { color: '#FFFFFF', fontSize: 18, fontWeight: '700' },
  permissionCopy: { color: '#AEB8B2', fontSize: 13, lineHeight: 19, marginTop: 8, maxWidth: 360, textAlign: 'center' },
  permissionButton: { backgroundColor: '#E7F7B7', borderRadius: 10, marginTop: 18, paddingHorizontal: 18, paddingVertical: 12 },
  permissionButtonText: { color: '#16211B', fontSize: 13, fontWeight: '700' },
  cameraOverlay: { bottom: 0, left: 0, padding: 18, position: 'absolute', right: 0, top: 0 },
  cameraTop: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between' },
  liveBadge: { alignItems: 'center', backgroundColor: 'rgba(9,17,13,.72)', borderRadius: 7, flexDirection: 'row', gap: 7, paddingHorizontal: 9, paddingVertical: 6 },
  liveDot: { backgroundColor: '#F16A5B', borderRadius: 4, height: 7, width: 7 },
  liveDotIdle: { backgroundColor: '#8E9993' },
  liveText: { color: '#FFFFFF', fontSize: 10, fontWeight: '700', letterSpacing: 0.8 },
  cameraSource: { color: '#E1E7E3', fontSize: 12, fontWeight: '600', textShadowColor: '#000000', textShadowRadius: 4 },
  guide: { alignSelf: 'center', aspectRatio: 4 / 3, height: '72%', marginVertical: 26, position: 'relative' },
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
  primaryButton: { alignItems: 'center', backgroundColor: '#DFF4A7', borderRadius: 11, flex: 1, flexDirection: 'row', gap: 9, justifyContent: 'center', minHeight: 52 },
  primaryButtonDisabled: { backgroundColor: '#D8DCD5' },
  primaryButtonText: { color: '#102019', fontSize: 14, fontWeight: '700' },
  secondaryButton: { alignItems: 'center', backgroundColor: '#FFFFFF', borderColor: '#DDE2DD', borderRadius: 11, borderWidth: 1, height: 52, justifyContent: 'center', width: 52 },
  secondaryButtonActive: { borderColor: '#B9CF7E', backgroundColor: '#F6FBE8' },
  uploadButton: { alignItems: 'center', backgroundColor: '#FFFFFF', borderColor: '#DDE2DD', borderRadius: 11, borderWidth: 1, flexDirection: 'row', gap: 11, minHeight: 58, paddingHorizontal: 14 },
  uploadButtonDisabled: { opacity: 0.5 },
  uploadCopy: { flex: 1, gap: 3 },
  uploadButtonText: { color: '#2D3A33', fontSize: 12, fontWeight: '700' },
  uploadHint: { color: '#89938D', fontSize: 10 },
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
  resultPanel: { backgroundColor: '#FFFFFF', borderColor: '#E0E4E0', borderRadius: 16, borderWidth: 1, minHeight: 430, padding: 22 },
  resultPanelDesktop: { flex: 1, minWidth: 330 },
  resultTop: { alignItems: 'flex-start', flexDirection: 'row', justifyContent: 'space-between' },
  panelLabel: { color: '#7B857F', fontSize: 10, fontWeight: '700', letterSpacing: 1.1 },
  tokenCount: { backgroundColor: '#F0F2EE', borderRadius: 7, paddingHorizontal: 9, paddingVertical: 6 },
  tokenCountText: { color: '#667169', fontSize: 11, fontWeight: '600' },
  transcriptArea: { flex: 1, justifyContent: 'center', minHeight: 145, paddingVertical: 24 },
  transcript: { color: '#17231D', fontSize: 27, fontWeight: '600', letterSpacing: -0.5, lineHeight: 36 },
  transcriptEmpty: { color: '#A2AAA5', fontSize: 22, fontWeight: '400', lineHeight: 31 },
  rawGloss: { color: '#808A83', fontSize: 12, lineHeight: 18, marginTop: 12 },
  resultDetailsToggle: { alignItems: 'center', alignSelf: 'flex-start', flexDirection: 'row', gap: 6, paddingBottom: 9, paddingTop: 3 },
  resultDetailsToggleText: { color: '#657168', fontSize: 11, fontWeight: '700' },
  confidenceSection: { borderTopColor: '#EBEEEB', borderTopWidth: 1, paddingTop: 16 },
  confidenceHeading: { flexDirection: 'row', justifyContent: 'space-between' },
  latestLabel: { color: '#314039', fontSize: 13, fontWeight: '700' },
  confidenceNumber: { color: '#536158', fontSize: 12, fontWeight: '700' },
  confidenceTrack: { backgroundColor: '#EBEEEA', height: 5, marginTop: 9, overflow: 'hidden' },
  confidenceFill: { backgroundColor: '#87A348', height: '100%' },
  confidenceCaption: { color: '#909994', fontSize: 10, marginTop: 7 },
  candidatesCard: { backgroundColor: '#F7F9F4', borderColor: '#E4EAE0', borderRadius: 10, borderWidth: 1, marginTop: 14, padding: 12 },
  candidatesHeading: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 },
  candidatesTitle: { color: '#405047', fontSize: 11, fontWeight: '700' },
  candidatesMeta: { color: '#89938D', fontSize: 10 },
  candidateRow: { alignItems: 'center', flexDirection: 'row', gap: 8, minHeight: 27 },
  candidateRank: { color: '#849075', fontSize: 10, fontWeight: '700', width: 18 },
  candidateLabel: { color: '#46544B', flex: 1, fontSize: 11, fontWeight: '600' },
  candidateConfidence: { color: '#65744D', fontSize: 11, fontWeight: '700' },
  tip: { alignItems: 'flex-start', backgroundColor: '#F4F5F2', borderRadius: 9, flexDirection: 'row', gap: 8, padding: 12 },
  tipText: { color: '#637169', flex: 1, fontSize: 12, lineHeight: 18 },
  errorBox: { alignItems: 'flex-start', backgroundColor: '#FFF3F1', borderRadius: 9, flexDirection: 'row', gap: 8, marginTop: 12, padding: 12 },
  errorText: { color: '#8F3A32', flex: 1, fontSize: 12, lineHeight: 18 },
  resultActions: { borderTopColor: '#EBEEEB', borderTopWidth: 1, flexDirection: 'row', gap: 10, marginTop: 16, paddingTop: 16 },
  textButton: { alignItems: 'center', borderColor: '#DEE3DE', borderRadius: 9, borderWidth: 1, flex: 1, flexDirection: 'row', gap: 7, justifyContent: 'center', minHeight: 42 },
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
