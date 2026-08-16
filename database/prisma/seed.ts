import { createHash } from 'node:crypto';
import { PrismaClient, Role } from '@prisma/client';
import { betterAuth } from 'better-auth';
import { prismaAdapter } from 'better-auth/adapters/prisma';

const prisma = new PrismaClient();
const hash = (value: string) => createHash('sha256').update(value).digest('hex');

async function main() {
  const auth = betterAuth({
    database: prismaAdapter(prisma, { provider: 'postgresql' }),
    secret: process.env.BETTER_AUTH_SECRET ?? 'replace-with-a-minimum-32-character-development-secret',
    emailAndPassword: { enabled: true },
  });
  if (!(await prisma.user.findUnique({ where: { email: 'admin@signbridge.local' } }))) {
    await auth.api.signUpEmail({ body: { name: 'Signbridge Admin', email: 'admin@signbridge.local', password: 'Signbridge123!' } });
  }
  const admin = await prisma.user.update({ where: { email: 'admin@signbridge.local' }, data: { role: Role.ADMIN } });
  const device = await prisma.device.upsert({ where: { id: 'uno-q-demo' }, update: {}, create: { id: 'uno-q-demo', name: 'Arduino UNO Q Demo' } });
  const secret = process.env.DEVICE_SEED_SECRET ?? 'replace-with-device-secret';
  await prisma.deviceCredential.upsert({ where: { keyId: 'demo-key' }, update: { deviceId: device.id, secretHash: hash(secret) }, create: { deviceId: device.id, keyId: 'demo-key', secretHash: hash(secret) } });
  console.log(`Seeded ${admin.email} with the development password.`);
}
main().finally(() => prisma.$disconnect());
