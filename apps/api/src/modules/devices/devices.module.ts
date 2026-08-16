import { Module } from '@nestjs/common'; import { DevicesController } from './devices.controller'; import { DeviceAuthGuard } from './device-auth.guard'; import { AuthModule } from '../../auth/auth.module';
@Module({ imports: [AuthModule], controllers: [DevicesController], providers: [DeviceAuthGuard], exports: [DeviceAuthGuard] }) export class DevicesModule {}
