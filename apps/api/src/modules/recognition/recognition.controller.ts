import { Body, Controller, Post, UseGuards } from '@nestjs/common';
import { recognitionEventSchema, RecognitionEvent } from '@signtalk/contracts';
import { PrismaService } from '../../core/database/database.module'; import { ZodValidationPipe } from '../../core/validation/zod-validation.pipe'; import { DeviceAuthGuard } from '../devices/device-auth.guard'; import { RecognitionGateway } from '../../realtime/recognition.gateway';
@Controller('recognition')
export class RecognitionController {
  constructor(private readonly prisma: PrismaService, private readonly gateway: RecognitionGateway) {}
  @Post('events') @UseGuards(DeviceAuthGuard)
  async create(@Body(new ZodValidationPipe(recognitionEventSchema)) event: RecognitionEvent) {
    const existing = await this.prisma.recognitionEvent.findUnique({ where: { eventId: event.eventId } });
    if (existing) return { event: existing, duplicate: true };
    const saved = await this.prisma.recognitionEvent.create({ data: { eventId: event.eventId, deviceId: event.deviceId, label: event.payload.label, text: event.payload.text, confidence: event.payload.confidence, occurredAt: new Date(event.occurredAt) } });
    this.gateway.emitConfirmed(event); return { event: saved, duplicate: false };
  }
}
