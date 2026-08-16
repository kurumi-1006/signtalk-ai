import { z } from 'zod';

export const uuidSchema = z.string().uuid();
export const isoDateSchema = z.string().datetime({ offset: true });
export const recognitionCandidateSchema = z.object({
  label: z.string().trim().min(1).max(100),
  confidence: z.number().min(0).max(1),
}).strict();
export const recognitionPayloadSchema = z
  .object({
    label: z.string().trim().min(1).max(100),
    text: z.string().trim().min(1).max(500),
    confidence: z.number().min(0).max(1),
    margin: z.number().min(0).max(1).optional(),
    landmarkCoverage: z.number().min(0).max(1).optional(),
    accepted: z.boolean().optional(),
    topK: z.array(recognitionCandidateSchema).min(1).max(5).optional(),
  })
  .strict();
export const recognitionEventSchema = z
  .object({ schemaVersion: z.literal(1), eventId: uuidSchema, eventType: z.literal('recognition.confirmed'), deviceId: z.string().trim().min(1).max(100), occurredAt: isoDateSchema, payload: recognitionPayloadSchema })
  .strict();
export type RecognitionEvent = z.infer<typeof recognitionEventSchema>;
export const deviceHeartbeatSchema = z.object({ status: z.enum(['ONLINE', 'OFFLINE', 'DEGRADED']) }).strict();
export const createSessionSchema = z.object({ deviceId: z.string().min(1) }).strict();
export const endSessionSchema = z.object({ endedAt: isoDateSchema.optional() }).strict();
export const sentenceTokenSchema = z.object({
  label: z.string().trim().min(1).max(100),
  confidence: z.number().min(0).max(1),
  margin: z.number().min(0).max(1).optional(),
  candidates: z.array(recognitionCandidateSchema).min(1).max(5).optional(),
}).strict();
export const refineSentenceRequestSchema = z.object({
  tokens: z.array(sentenceTokenSchema).min(1).max(40),
  previousSentence: z.string().trim().max(500).optional(),
}).strict();
export const refineSentenceResponseSchema = z.object({
  rawText: z.string().trim().min(1).max(1000),
  correctedText: z.string().trim().min(1).max(1000),
  uncertain: z.boolean(),
  provider: z.enum(['deepseek', 'fallback']),
}).strict();
export type SentenceToken = z.infer<typeof sentenceTokenSchema>;
export type RefineSentenceRequest = z.infer<typeof refineSentenceRequestSchema>;
export type RefineSentenceResponse = z.infer<typeof refineSentenceResponseSchema>;
export const socketRecognitionConfirmedSchema = recognitionEventSchema;
export const apiEnvironmentSchema = z.object({ NODE_ENV: z.enum(['development', 'test', 'production']).default('development'), PORT: z.coerce.number().int().positive().default(3000), DATABASE_URL: z.string().url(), REDIS_URL: z.string().url(), BETTER_AUTH_SECRET: z.string().min(32), BETTER_AUTH_URL: z.string().url(), CORS_ORIGINS: z.string(), SWAGGER_ENABLED: z.enum(['true', 'false']).default('false'), LOG_LEVEL: z.string().default('info'), DEEPSEEK_API_KEY: z.string().optional(), DEEPSEEK_MODEL: z.string().default('deepseek-v4-flash') });
export type ApiEnvironment = z.infer<typeof apiEnvironmentSchema>;
