import { Body, Controller, Get, Param, Post, Req, UseGuards } from '@nestjs/common';
import { Request } from 'express';
import { ApiBody } from '@nestjs/swagger';
import { deviceHeartbeatSchema } from '@signtalk/contracts';
import { SessionGuard } from '../../auth/session.guard'; import { PrismaService } from '../../core/database/database.module'; import { ZodValidationPipe } from '../../core/validation/zod-validation.pipe'; import { DeviceAuthGuard } from './device-auth.guard';
@Controller('devices')
export class DevicesController {
  constructor(private readonly prisma: PrismaService) {}
  @Get() @UseGuards(SessionGuard) list() { return this.prisma.device.findMany({ orderBy: { updatedAt: 'desc' } }); }
  @Get(':id') @UseGuards(SessionGuard) one(@Param('id') id: string) { return this.prisma.device.findUniqueOrThrow({ where: { id } }); }
  @Post('heartbeat') @UseGuards(DeviceAuthGuard) @ApiBody({ schema: { example: { status: 'ONLINE' } } })
  heartbeat(@Body(new ZodValidationPipe(deviceHeartbeatSchema)) body: { status: 'ONLINE' | 'OFFLINE' | 'DEGRADED' }, @Req() req: Request & { deviceId: string }) { return this.prisma.device.update({ where: { id: req.deviceId }, data: { status: body.status, lastSeenAt: new Date() } }); }
}
