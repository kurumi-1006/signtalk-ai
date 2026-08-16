import { CanActivate, ExecutionContext, Injectable, UnauthorizedException } from '@nestjs/common';
import { createHash, createHmac, timingSafeEqual } from 'node:crypto';
import { PrismaService } from '../../core/database/database.module';
const digest = (v: string) => createHash('sha256').update(v).digest('hex');
export const canonical = (method: string, path: string, timestamp: string, nonce: string, body: string) => `${method}\n${path}\n${timestamp}\n${nonce}\n${body}`;
@Injectable()
export class DeviceAuthGuard implements CanActivate {
  constructor(private readonly prisma: PrismaService) {}
  async canActivate(context: ExecutionContext) {
    const req = context.switchToHttp().getRequest(); const h = req.headers as Record<string, string | undefined>;
    const [deviceId, keyId, timestamp, nonce, signature] = [h['x-device-id'], h['x-key-id'], h['x-timestamp'], h['x-nonce'], h['x-signature']];
    if (!deviceId || !keyId || !timestamp || !nonce || !signature || Math.abs(Date.now() - Date.parse(timestamp)) > 300_000) throw new UnauthorizedException('Invalid device authentication headers');
    const credential = await this.prisma.deviceCredential.findUnique({ where: { keyId } });
    const secret = process.env.DEVICE_SEED_SECRET;
    if (!credential || credential.deviceId !== deviceId || credential.revokedAt || !secret || credential.secretHash !== digest(secret)) throw new UnauthorizedException('Unknown device credential');
    const path = String(req.originalUrl ?? req.url).replace(/^\/api\/v1/, '').split('?')[0];
    // Sign the exact request bytes. Serializing parsed JSON independently in
    // Python and Node changes representations such as 0.0 versus 0.
    const rawBody = Buffer.isBuffer(req.rawBody)
      ? req.rawBody.toString('utf8')
      : JSON.stringify(req.body);
    const expected = createHmac('sha256', secret).update(canonical(req.method, path, timestamp, nonce, rawBody)).digest('hex');
    if (expected.length !== signature.length || !timingSafeEqual(Buffer.from(expected), Buffer.from(signature))) throw new UnauthorizedException('Invalid device signature');
    req.deviceId = deviceId; return true;
  }
}
