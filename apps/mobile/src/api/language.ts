import {
  type RefineSentenceRequest,
  type RefineSentenceResponse,
  refineSentenceResponseSchema,
} from '@signtalk/contracts';
import { api } from './client';

export async function refineSentence(input: RefineSentenceRequest): Promise<RefineSentenceResponse> {
  const response = await api.post('/language/refine', input);
  return refineSentenceResponseSchema.parse(response.data);
}
