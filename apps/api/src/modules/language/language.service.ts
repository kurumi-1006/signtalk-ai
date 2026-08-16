import { Injectable, Logger } from '@nestjs/common';
import {
  type RefineSentenceRequest,
  type RefineSentenceResponse,
  refineSentenceResponseSchema,
} from '@signtalk/contracts';

@Injectable()
export class LanguageService {
  private readonly logger = new Logger(LanguageService.name);

  async refine(input: RefineSentenceRequest): Promise<RefineSentenceResponse> {
    const rawText = input.tokens.map((token) => token.label).join(' ');
    const apiKey = process.env.DEEPSEEK_API_KEY;
    if (!apiKey) return this.fallback(rawText);

    const candidates = input.tokens.map((token) => ({
      selected: token.label,
      confidence: token.confidence,
      margin: token.margin,
      candidates: token.candidates,
    }));
    const systemPrompt = [
      'Bạn là bộ hậu xử lý cho hệ thống nhận diện Ngôn ngữ ký hiệu Việt Nam.',
      'Chỉ được sắp xếp và sửa ngữ pháp dựa trên các gloss/candidate được cung cấp.',
      'Không thêm sự kiện, con người, địa điểm, đồ vật hoặc ý nghĩa mới.',
      'Giữ nguyên token có confidence cao; nếu dữ liệu không đủ chắc chắn, đặt uncertain=true.',
      'Trả về JSON đúng dạng: {"rawText":"...","correctedText":"...","uncertain":true|false,"provider":"deepseek"}.',
    ].join(' ');

    try {
      const response = await fetch('https://api.deepseek.com/chat/completions', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: process.env.DEEPSEEK_MODEL ?? 'deepseek-v4-flash',
          thinking: { type: 'disabled' },
          temperature: 0.1,
          max_tokens: 300,
          response_format: { type: 'json_object' },
          messages: [
            { role: 'system', content: systemPrompt },
            {
              role: 'user',
              content: JSON.stringify({
                instruction: 'Sửa chuỗi gloss thành một câu tiếng Việt tự nhiên. Chỉ trả JSON.',
                rawText,
                previousSentence: input.previousSentence ?? '',
                tokens: candidates,
              }),
            },
          ],
        }),
        signal: AbortSignal.timeout(8000),
      });
      if (!response.ok) throw new Error(`DeepSeek returned HTTP ${response.status}`);
      const body = await response.json() as {
        choices?: Array<{ message?: { content?: string | null } }>;
      };
      const content = body.choices?.[0]?.message?.content;
      if (!content) throw new Error('DeepSeek returned an empty response');
      const parsed = refineSentenceResponseSchema.safeParse(JSON.parse(content));
      if (!parsed.success) throw new Error('DeepSeek returned an invalid sentence payload');
      return { ...parsed.data, rawText, provider: 'deepseek' };
    } catch (error) {
      this.logger.warn(`Sentence refinement fallback: ${error instanceof Error ? error.message : String(error)}`);
      return this.fallback(rawText);
    }
  }

  private fallback(rawText: string): RefineSentenceResponse {
    const correctedText = `${rawText.charAt(0).toLocaleUpperCase('vi-VN')}${rawText.slice(1)}${/[.!?]$/.test(rawText) ? '' : '.'}`;
    return { rawText, correctedText, uncertain: true, provider: 'fallback' };
  }
}
