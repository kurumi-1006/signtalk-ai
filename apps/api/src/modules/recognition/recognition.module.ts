import { Module } from '@nestjs/common'; import { RecognitionController } from './recognition.controller'; import { DevicesModule } from '../devices/devices.module'; import { RealtimeModule } from '../../realtime/realtime.module';
@Module({ imports: [DevicesModule, RealtimeModule], controllers: [RecognitionController] }) export class RecognitionModule {}
