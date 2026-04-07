const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 1000, deviceScaleFactor: 2 });
  
  const filePath = `file://${path.resolve(__dirname, '../index.html')}`;
  await page.goto(filePath, { waitUntil: 'networkidle0' });
  
  await page.evaluate(() => {
    const header = document.querySelector('.header');
    if(header) header.style.display = 'none';
    const detailPanel = document.querySelector('.detail-panel');
    if(detailPanel) detailPanel.style.display = 'none';
    const container = document.querySelector('.container');
    if(container) {
      container.style.gridTemplateColumns = '1fr';
      container.style.gap = '0';
    }
    const compounds = document.querySelector('.compounds');
    if (compounds) compounds.style.display = 'none';
    const h3s = document.querySelectorAll('h3');
    h3s.forEach(h => h.style.display = 'none');
  });

  const element = await page.$('.container');
  if (!fs.existsSync('figures')) fs.mkdirSync('figures');
  await element.screenshot({ path: 'figures/periodic_table_hero.png', omitBackground: true });
  
  await browser.close();
})();