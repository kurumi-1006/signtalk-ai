export type RecognitionCandidate = { label: string; confidence: number };

export type RecognitionEvent = {
  eventId: string;
  eventType: 'recognition.confirmed';
  occurredAt: string;
  payload: {
    label: string;
    text: string;
    confidence: number;
    margin?: number;
    landmarkCoverage?: number;
    accepted?: boolean;
    topK: RecognitionCandidate[];
  };
};

export type SentenceToken = {
  label: string;
  confidence: number;
  margin?: number;
  candidates: RecognitionCandidate[];
};
