"""Exercise a running Volume III notebook: evidence, save, export, and track reset."""
import argparse
import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright
async def main(url, screenshots):
 screenshots.mkdir(parents=True, exist_ok=True)
 async with async_playwright() as p:
  b=await p.chromium.launch();page=await b.new_page(viewport={'width':1440,'height':1000});await page.goto(url);await page.wait_for_timeout(5000)
  for i in range(4):
   await page.get_by_role('tab').nth(i).click();await page.get_by_role('radio').nth(2).check();await page.wait_for_timeout(400)
   await page.get_by_role('combobox').last.select_option(index=2);await page.wait_for_timeout(600)
   await page.get_by_role('textbox').first.fill(f'Part {i}: We changed the policy at fixed task conditions; the trace supports a bounded decision and leaves production evidence unresolved.');await page.wait_for_timeout(600)
   await page.get_by_role('button',name='Save comparison and interpretation',exact=True).click();await page.wait_for_timeout(500)
   print('SAVED',i,flush=True)
  await page.get_by_role('textbox').last.fill('Hold target release pending a matched production evaluation. The supplied policy trace supports mediation; an available endpoint reconciliation would change the next action.');await page.wait_for_timeout(700)
  save=page.get_by_role('button',name='Save discussion in this session',exact=True);await save.click();await page.wait_for_timeout(600)
  async with page.expect_download() as info:
   await page.get_by_text('Download discussion and evidence',exact=True).click()
  dl=await info.value;data=json.load(open(await dl.path()));assert data['missing_parts']==[];assert len(data['evidence'])==4
  print('COMPLETE_DOWNLOAD',data['completed_parts'],flush=True)
  await page.get_by_role('textbox').last.fill('Edited decision after saving.');await page.wait_for_timeout(600)
  assert await page.get_by_text('Discussion changed after saving; save again.',exact=True).count()==1
  await page.get_by_role('tab').nth(1).click();await page.get_by_role('table').first.scroll_into_view_if_needed();await page.screenshot(path=str(screenshots / 'final1440.png'))
  await page.set_viewport_size({'width':390,'height':844});await page.get_by_role('combobox').last.scroll_into_view_if_needed();await page.screenshot(path=str(screenshots / 'final390.png'))
  await page.get_by_role('combobox').first.select_option(index=1);await page.wait_for_timeout(1200)
  assert await page.get_by_role('textbox').last.input_value()==''
  print('TRACK_RESET_PASS',flush=True)
  assert not await page.evaluate('document.documentElement.scrollWidth > innerWidth')
  await b.close()
if __name__ == "__main__":
 parser = argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--url", required=True)
 parser.add_argument("--screenshots", type=Path, default=Path("/tmp/mlsysbook-agentic-discussion-qa"))
 args = parser.parse_args()
 asyncio.run(main(args.url, args.screenshots))
