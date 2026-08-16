import tseslint from 'typescript-eslint';
export default [
  { ignores: ['**/dist/**', '**/node_modules/**', 'pnpm-lock.yaml'] },
  { files: ['**/*.{ts,tsx}'], languageOptions: { parser: tseslint.parser, parserOptions: { ecmaVersion: 2022, sourceType: 'module' } } },
];
