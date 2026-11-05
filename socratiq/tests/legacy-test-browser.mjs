import {chromium, webkit} from '@playwright/test';

export const testBrowserName = process.env.SOCRATIQ_TEST_BROWSER || 'chromium';
export const testBrowser = {chromium, webkit}[testBrowserName];
if (!testBrowser) throw new Error('SOCRATIQ_TEST_BROWSER must be chromium or webkit');
console.info(`Legacy browser fixture: ${testBrowserName}`);
