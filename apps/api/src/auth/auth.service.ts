import { Injectable, UnauthorizedException } from '@nestjs/common';
import { betterAuth } from 'better-auth';
import { prismaAdapter } from 'better-auth/adapters/prisma';
import { PrismaService } from '../core/database/database.module';
@Injectable()
export class AuthService {
  readonly auth;
  constructor(prisma: PrismaService) {
    this.auth = betterAuth({ database: prismaAdapter(prisma, { provider: 'postgresql' }), secret: process.env.BETTER_AUTH_SECRET, baseURL: process.env.BETTER_AUTH_URL, emailAndPassword: { enabled: true } });
  }
  async session(headers: Headers) { return this.auth.api.getSession({ headers }); }
  async requireUser(headers: Headers) { const session = await this.session(headers); if (!session) throw new UnauthorizedException(); return session.user; }
}
