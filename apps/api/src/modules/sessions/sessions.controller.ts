import { Body, Controller, Get, Param, Patch, Post, UseGuards } from '@nestjs/common';
import { createSessionSchema, endSessionSchema } from '@signtalk/contracts';
import { SessionGuard } from '../../auth/session.guard'; import { PrismaService } from '../../core/database/database.module'; import { ZodValidationPipe } from '../../core/validation/zod-validation.pipe';
@Controller('sessions') @UseGuards(SessionGuard)
export class SessionsController {
  constructor(private readonly prisma: PrismaService) {}
  @Post() create(@Body(new ZodValidationPipe(createSessionSchema)) body: { deviceId: string }) { return this.prisma.recognitionSession.create({ data: body }); }
  @Get(':id') one(@Param('id') id: string) { return this.prisma.recognitionSession.findUniqueOrThrow({ where: { id } }); }
  @Get(':id/events') events(@Param('id') id: string) { return this.prisma.recognitionEvent.findMany({ where: { sessionId: id }, orderBy: { occurredAt: 'desc' } }); }
  @Patch(':id/end') end(@Param('id') id: string, @Body(new ZodValidationPipe(endSessionSchema)) body: { endedAt?: string }) { return this.prisma.recognitionSession.update({ where: { id }, data: { endedAt: body.endedAt ? new Date(body.endedAt) : new Date() } }); }
}
