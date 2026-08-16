import { Module } from '@nestjs/common'; import { RecognitionGateway } from './recognition.gateway';
@Module({ providers: [RecognitionGateway], exports: [RecognitionGateway] }) export class RealtimeModule {}
