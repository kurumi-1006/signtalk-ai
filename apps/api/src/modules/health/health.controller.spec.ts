import { HealthController } from './health.controller';
describe('HealthController', () => { it('returns a healthy service response', () => { expect(new HealthController().check()).toMatchObject({ status: 'ok', service: 'signtalk-api' }); }); });
