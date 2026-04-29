const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  const questionId = 'edge-0050';
  const url = `http://localhost:3000/practice?q=${questionId}`;
  console.log(`Navigating to ${url}`);
  
  await page.addInitScript(() => {
    window.localStorage.setItem('staffml_star_gate', JSON.stringify({ verified: true }));
    window.localStorage.setItem('staffml_attempts', JSON.stringify([{ id: 'dummy', date: '2020-01-01' }]));
    window.localStorage.setItem('staffml_firstrun_welcome', '1');
  });

  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);
  
  // Reveal Solution
  const revealButton = await page.getByRole('button', { name: /Reveal Answer/i });
  if (await revealButton.isVisible()) {
      await revealButton.click();
  }

  await page.waitForTimeout(2000);

  await page.setViewportSize({ width: 1280, height: 2000 });
  await page.screenshot({ path: 'final_polish_check.png', fullPage: true });
  console.log('Screenshot saved to final_polish_check.png');
  
  await browser.close();
})();
