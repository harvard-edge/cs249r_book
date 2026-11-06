import {test, expect} from '@playwright/test';
const endpoint = 'https://mlsysbook.ai/api/socratiq/ai/stream**';

test('first open introduces controls without requests, supports Escape and replay', async ({page}, testInfo) => {
  let requests = 0;
  await page.route(endpoint, route => {requests++; return route.abort();});
  await page.addInitScript(() => localStorage.removeItem('socratiq_reader_tour_v1'));
  await page.goto('/vol1/chapter/');
  await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
  const tour = page.getByRole('dialog', {name: 'SocratiQ guided tour'});
  await expect(tour).toBeVisible();
  await page.screenshot({path: testInfo.outputPath('tour-desktop.png')});
  await expect(tour.getByText('Step 1 of 3')).toBeVisible();
  await tour.getByRole('button', {name: 'Next', exact: true}).click();
  await expect(tour.getByText('Step 2 of 3')).toBeVisible();
  await tour.getByRole('button', {name: 'Back', exact: true}).click();
  await expect(tour.getByText('Step 1 of 3')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(tour).toBeHidden();
  await expect(page.getByRole('region', {name: 'SocratiQ focused reader', exact: true})).toBeVisible();
  await page.getByRole('button', {name: 'Replay tour'}).click();
  await expect(tour).toBeVisible();
  await tour.getByRole('button', {name: 'Skip tour'}).click();
  await page.getByRole('button', {name: 'Close SocratiQ reader'}).click();
  await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
  await expect(tour).toBeHidden();
  expect(requests).toBe(0);
});

test('preferences persist locally, synchronize legacy difficulty, and clear personal text', async ({page}, testInfo) => {
  let requests = 0;
  await page.route(endpoint, route => {requests++; return route.abort();});
  await page.addInitScript(() => {
    if (!localStorage.getItem('userSettings')) localStorage.setItem('userSettings', JSON.stringify({sliderValue: 0, unrelated: 'keep'}));
  });
  await page.goto('/vol1/chapter/');
  await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
  await page.getByText('Learning preferences', {exact: true}).click();
  await expect(page.getByLabel('Explanation level')).toHaveValue('beginner');
  await page.getByLabel('Explanation level').selectOption('advanced');
  await page.getByLabel('What do you already know?', {exact: false}).fill('I know Python.');
  await page.getByLabel('How should explanations help you?', {exact: false}).fill('Define symbols.');
  await page.getByRole('button', {name: 'Save preferences', exact: true}).click();
  await expect(page.getByText('Preferences saved in this browser.', {exact: true})).toBeVisible();
  await page.getByLabel('Explanation level').scrollIntoViewIfNeeded();
  await page.screenshot({path: testInfo.outputPath('preferences-desktop.png')});
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem('userSettings')))).toEqual({sliderValue: 2, unrelated: 'keep'});
  await page.reload();
  await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
  await page.getByText('Learning preferences', {exact: true}).click();
  await expect(page.getByLabel('Explanation level')).toHaveValue('advanced');
  await expect(page.getByLabel('What do you already know?', {exact: false})).toHaveValue('I know Python.');
  await page.getByRole('button', {name: 'Clear preferences', exact: true}).click();
  await expect(page.getByLabel('Explanation level')).toHaveValue('intermediate');
  await expect(page.getByLabel('What do you already know?', {exact: false})).toHaveValue('');
  await expect(page.getByLabel('How should explanations help you?', {exact: false})).toHaveValue('');
  expect(requests).toBe(0);
});

test('phone tour stays in viewport with dark mode and reduced motion', async ({page}, testInfo) => {
  await page.setViewportSize({width: 375, height: 667});
  await page.emulateMedia({colorScheme: 'dark', reducedMotion: 'reduce'});
  await page.goto('/vol1/chapter/');
  await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
  await page.getByRole('button', {name: 'Replay tour'}).click();
  const tour = page.getByRole('dialog', {name: 'SocratiQ guided tour'});
  for (const step of [1, 2, 3]) {
    await expect(tour.getByText(`Step ${step} of 3`)).toBeVisible();
    if (step === 1) await page.screenshot({path: testInfo.outputPath('tour-phone-dark.png')});
    const box = await tour.boundingBox();
    expect(box.x).toBeGreaterThanOrEqual(0); expect(box.y).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(375); expect(box.y + box.height).toBeLessThanOrEqual(667);
    await tour.getByRole('button', {name: step === 3 ? 'Done' : 'Next', exact: true}).click();
  }
  await expect(tour).toBeHidden();
});
