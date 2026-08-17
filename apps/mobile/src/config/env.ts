import { z } from 'zod';
const schema = z.object({
  edgeAiUrl: z.string().url().default('http://localhost:8082'),
  deviceId: z.string().min(1).default('uno-q-demo'),
  environment: z.string().default('development'),
});

export const env = schema.parse({
  edgeAiUrl: process.env.EXPO_PUBLIC_EDGE_AI_URL,
  deviceId: process.env.EXPO_PUBLIC_DEVICE_ID,
  environment: process.env.EXPO_PUBLIC_ENVIRONMENT,
});
