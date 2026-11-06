import {defineConfig} from '@playwright/test';

export default defineConfig({
  testDir: './tests', testMatch: 'reader-*.spec.mjs',
  fullyParallel: false, workers: 1,
  outputDir: process.env.SOCRATIQ_TEST_OUTPUT || '/tmp/socratiq-reader-browser',
  // Existing reader flows model a returning learner; tour specs explicitly reset this flag.
  use: {storageState: {cookies: [], origins: [{origin: 'http://127.0.0.1:4176', localStorage: [{name: 'socratiq_reader_tour_v1', value: 'true'}]}]}, baseURL: 'http://127.0.0.1:4176', screenshot: 'only-on-failure', trace: 'retain-on-failure'},
  webServer: {command: 'node tests/reader-fixture-server.mjs', url: 'http://127.0.0.1:4176/vol1/chapter/', reuseExistingServer: false},
});
