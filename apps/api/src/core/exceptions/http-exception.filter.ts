import { ArgumentsHost, Catch, ExceptionFilter, HttpException, HttpStatus } from '@nestjs/common';
import { Request, Response } from 'express';
import { randomUUID } from 'node:crypto';
@Catch()
export class ApiExceptionFilter implements ExceptionFilter {
  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp(); const response = ctx.getResponse<Response>(); const request = ctx.getRequest<Request>();
    const status = exception instanceof HttpException ? exception.getStatus() : HttpStatus.INTERNAL_SERVER_ERROR;
    const body = exception instanceof HttpException ? exception.getResponse() : {};
    const detail = typeof body === 'object' && body ? body as Record<string, unknown> : {};
    response.status(status).json({ statusCode: status, error: detail.error ?? (status >= 500 ? 'INTERNAL_ERROR' : 'REQUEST_ERROR'), message: detail.message ?? 'Request failed', details: detail.details ?? [], requestId: request.headers['x-request-id'] ?? randomUUID(), timestamp: new Date().toISOString(), path: request.originalUrl });
  }
}
