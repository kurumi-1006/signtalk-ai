import { z } from 'zod';
const schema = z.object({
  edgeAiUrls: z.string().optional(),
});

export const env = schema.parse({
  edgeAiUrls: process.env.EXPO_PUBLIC_EDGE_AI_URLS ?? process.env.EXPO_PUBLIC_EDGE_AI_URL,
});
