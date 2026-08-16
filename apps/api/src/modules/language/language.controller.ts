import { Body, Controller, Post } from '@nestjs/common';
import {
  type RefineSentenceRequest,
  refineSentenceRequestSchema,
} from '@signtalk/contracts';
import { ZodValidationPipe } from '../../core/validation/zod-validation.pipe';
import { LanguageService } from './language.service';

@Controller('language')
export class LanguageController {
  constructor(private readonly language: LanguageService) {}

  @Post('refine')
  refine(
    @Body(new ZodValidationPipe(refineSentenceRequestSchema)) body: RefineSentenceRequest,
  ) {
    return this.language.refine(body);
  }
}
