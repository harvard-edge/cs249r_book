import {test, expect} from '@playwright/test';

// Author-constructed renderer regressions. Network responses below are local mocks.
test('literal math-marker text and code survive alongside rendered math', async ({page}, info) => {
  const answer = 'Literal marker: `SQRMTOKEN0END`\n\nLiteral sentence: SQRMTOKEN0END and SQRMTOKEN1END.\n\n```text\nSQRMTOKEN0END\n```\n\nMath: $t=D/B$\n\n$$B=D/t$$';
  await page.route('https://proxy-worker-streaming.mlsysbook.workers.dev/**', route => route.fulfill({
    contentType: 'text/event-stream',
    body: `data: ${JSON.stringify({choices:[{delta:{content:answer},finish_reason:'stop'}]})}\n\ndata: [DONE]\n\n`,
  }));
  await page.goto('/vol1/chapter/');
  await page.locator('#passage').evaluate(element => {
    const range = document.createRange(); range.selectNodeContents(element);
    const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
    document.dispatchEvent(new Event('selectionchange'));
  });
  await page.getByRole('button', {name:'Explain this', exact:true}).click();
  await expect(page.getByText('Explanation ready.', {exact:true})).toBeVisible();
  const result = page.locator('#socratiq-reader .reader-answer');
  await expect(result.locator('p > code')).toHaveText('SQRMTOKEN0END');
  await expect(result.locator('pre > code')).toHaveText('SQRMTOKEN0END\n');
  await expect(result.locator('code .katex')).toHaveCount(0);
  await expect(result).toContainText('Literal sentence: SQRMTOKEN0END and SQRMTOKEN1END.');
  await expect(result.locator('.katex')).toHaveCount(2);
  await page.screenshot({path:info.outputPath('literal-markers-and-math.png')});
});


test('Markdown entity decoding cannot manufacture an internal math marker', async ({page}) => {
  const answer = 'Literal sentence: SQR&#77;TOKEN0END.\n\nMath: $t=D/B$';
  await page.route('https://proxy-worker-streaming.mlsysbook.workers.dev/**', route => route.fulfill({contentType:'text/event-stream',body:`data: ${JSON.stringify({choices:[{delta:{content:answer},finish_reason:'stop'}]})}\n\ndata: [DONE]\n\n`}));
  await page.goto('/vol1/chapter/');
  await page.locator('#passage').evaluate(element => {const range=document.createRange();range.selectNodeContents(element);const selection=getSelection();selection.removeAllRanges();selection.addRange(range);document.dispatchEvent(new Event('selectionchange'));});
  await page.getByRole('button',{name:'Explain this',exact:true}).click();
  await expect(page.getByText('Explanation ready.',{exact:true})).toBeVisible();
  const result=page.locator('#socratiq-reader .reader-answer');
  await expect(result).toContainText('Literal sentence: SQRMTOKEN0END.');
  await expect(result.locator('.katex')).toHaveCount(1);
});

test('literal placeholder HTML and marker link attributes do not become math', async ({page}) => {
  const answer = 'Literal HTML: <span data-reader-math="0">literal</span>\n\n[Literal reference](https://constructed.invalid/SQRMTOKEN0END "SQRMTOKEN0END")\n\nMath: $t=D/B$';
  await page.route('https://proxy-worker-streaming.mlsysbook.workers.dev/**', route => route.fulfill({contentType:'text/event-stream',body:`data: ${JSON.stringify({choices:[{delta:{content:answer},finish_reason:'stop'}]})}\n\ndata: [DONE]\n\n`}));
  await page.goto('/vol1/chapter/');
  await page.locator('#passage').evaluate(element => {const range=document.createRange();range.selectNodeContents(element);const selection=getSelection();selection.removeAllRanges();selection.addRange(range);document.dispatchEvent(new Event('selectionchange'));});
  await page.getByRole('button',{name:'Explain this',exact:true}).click();
  await expect(page.getByText('Explanation ready.',{exact:true})).toBeVisible();
  const result=page.locator('#socratiq-reader .reader-answer');
  await expect(result).toContainText('Literal HTML: <span data-reader-math="0">literal</span>');
  await expect(result.getByRole('link',{name:'Literal reference'})).toHaveAttribute('href','https://constructed.invalid/SQRMTOKEN0END');
  await expect(result.getByRole('link',{name:'Literal reference'})).toHaveAttribute('title','SQRMTOKEN0END');
  await expect(result.locator('.katex')).toHaveCount(1);
});
