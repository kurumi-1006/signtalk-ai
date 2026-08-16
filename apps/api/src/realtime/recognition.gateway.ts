import { ConnectedSocket, SubscribeMessage, WebSocketGateway, WebSocketServer } from '@nestjs/websockets';
import { Socket } from 'socket.io';
import { Server } from 'socket.io';
import { RecognitionEvent } from '@signtalk/contracts';
@WebSocketGateway({ namespace: '/recognition', cors: { origin: true, credentials: true } })
export class RecognitionGateway {
  @WebSocketServer() server!: Server;
  @SubscribeMessage('device:join')
  joinDevice(@ConnectedSocket() socket: Socket, deviceId: string) { if (/^[a-zA-Z0-9_-]{1,100}$/.test(deviceId)) socket.join(`device:${deviceId}`); }
  emitConfirmed(event: RecognitionEvent) { this.server.to(`device:${event.deviceId}`).emit('recognition:confirmed', event); }
  emitDeviceStatus(deviceId: string, status: string) { this.server.to(`device:${deviceId}`).emit('device:status', { deviceId, status }); }
}
