import { writeFileSync } from 'node:fs';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { recognitionEventSchema } from '../src/index.js';

writeFileSync('recognition-event.schema.json', JSON.stringify(zodToJsonSchema(recognitionEventSchema, 'RecognitionEvent'), null, 2));
