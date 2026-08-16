import { Logger, VersioningType } from '@nestjs/common';
import { config as loadEnvironment } from 'dotenv';
import { join } from 'node:path';
import { NestFactory } from '@nestjs/core';
import helmet from 'helmet';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import { AppModule } from './app.module';
import { ApiExceptionFilter } from './core/exceptions/http-exception.filter';
import { AuthService } from './auth/auth.service';
import { toNodeHandler } from 'better-auth/node';
loadEnvironment({ path: join(process.cwd(), 'apps/api/.env') });
async function bootstrap() {
  const app = await NestFactory.create(AppModule, { bufferLogs: true, rawBody: true });
  app.use('/api/auth', toNodeHandler(app.get(AuthService).auth));
  const origins = new Set([
    ...(process.env.CORS_ORIGINS ?? '').split(',').map((value) => value.trim()).filter(Boolean),
    'http://localhost:8081', 'http://127.0.0.1:8081',
  ]);
  app.use(helmet()); app.enableCors({
    origin: (origin: string | undefined, callback: (error: Error | null, allow?: boolean) => void) => callback(null, !origin || origins.has(origin)),
    credentials: true,
  }); app.setGlobalPrefix('api'); app.enableVersioning({ type: VersioningType.URI, defaultVersion: '1' }); app.useGlobalFilters(new ApiExceptionFilter()); app.enableShutdownHooks();
  if (process.env.SWAGGER_ENABLED === 'true') { const config = new DocumentBuilder().setTitle('SIGNTALK AI API').setVersion('1').build(); SwaggerModule.setup('api/docs', app, SwaggerModule.createDocument(app, config)); }
  await app.listen(process.env.PORT ?? 3000, '0.0.0.0'); Logger.log('SIGNTALK API started');
}
bootstrap().catch((error: unknown) => {
  console.error('Bootstrap failed:', error);
  process.exit(1);
});
