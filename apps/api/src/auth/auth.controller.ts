import { Controller, Get, Headers as RequestHeaders, UseGuards } from '@nestjs/common';
import { AuthService } from './auth.service';
import { SessionGuard } from './session.guard';
@Controller()
export class AuthController {
  constructor(private readonly auth: AuthService) {}
  @Get('profile') @UseGuards(SessionGuard)
  async profile(@RequestHeaders() headers: Record<string, string>) { return { user: await this.auth.requireUser(new globalThis.Headers(headers)) }; }
}
