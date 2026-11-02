"""Check every original-art endpoint, local SVG references, keys, and print in real chapter builds."""
from pathlib import Path
import argparse
import json


def resolve_chapter_html(run, jobs, source, volume):
    """Find actual HTML from a successful chapter or full-volume build.

    Prefer the owning chapter build, but fall back to a full-volume build
    when the chapter job is absent, failed, or lacks the requested HTML.
    Job output paths follow the build runner's canonical directory layout.
    """
    source = Path(source)
    relative_html = source.relative_to("books").with_suffix(".html")
    expected_volume = f"vol{volume}"
    candidates = [
        job for job in jobs
        if job.get("ok") is True
        and job.get("format") == "html"
        and job.get("volume") == expected_volume
        and job.get("chapter") in (source.stem, None)
    ]
    candidates.sort(key=lambda job: job.get("chapter") != source.stem)
    for job in candidates:
        path = Path(run) / job["job"] / "output" / relative_html
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No successful {expected_volume} chapter or full-volume HTML job "
        f"contains {relative_html} under {run}"
    )


def main():
    import yaml
    from playwright.sync_api import sync_playwright

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root',type=Path,required=True)
    parser.add_argument('--base-url',default='http://127.0.0.1:8882')
    parser.add_argument('--engine',choices=['chromium','webkit'],default='chromium')
    parser.add_argument('--screenshots',type=Path,default=Path('/tmp/originals-final-review'))
    parser.add_argument('--keys',help='Optional comma-separated implementation keys to check')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    run=args.build_root if args.build_root.is_absolute() else root/args.build_root
    ext=root/'books/_extensions/mlsysbook-ext/figure-explore'
    manifest=json.loads((ext/'mechanisms/implementations.json').read_text());registry=json.loads((ext/'figure-registry.json').read_text());summary=json.loads((run/'summary.json').read_text())
    timing=yaml.safe_load((root/'books/config/shared/html/figure-explore.yml').read_text())['figure-explore']
    if args.keys:
     selected=args.keys.split(',')
     unknown=set(selected)-set(manifest)
     if unknown:parser.error('Unknown implementation keys: '+', '.join(sorted(unknown)))
     manifest={key:manifest[key] for key in selected}
    out=args.screenshots;out.mkdir(parents=True,exist_ok=True)
    report=[]
    with sync_playwright() as p:
     b=getattr(p,args.engine).launch()
     for key,impl in manifest.items():
      entry=registry[key];src=Path(entry['source'])
      path=resolve_chapter_html(run, summary['jobs'], src, entry['volume'])
      url=args.base_url.rstrip('/')+'/'+path.relative_to(root/'books').as_posix()+'?view=explore#'+entry['figure_id']
      for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
       page=b.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='reduce');errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
       page.goto(url);outer=page.locator('#'+entry['figure_id']);page.wait_for_function('(id)=>document.getElementById(id).dataset.mlsysExploreState==="ready"',arg=entry['figure_id'])
       comp=outer.locator('.mlsys-original-player,.life-panel');assert comp.is_visible();assert outer.get_attribute('data-mlsys-movement-ms')==str(timing['movement-ms']);assert outer.get_attribute('data-mlsys-reading-pause-ms')==str(timing['reading-pause-ms'])
       menu=comp.locator('.mlsys-original-sequence select,.life-sequence select');values=menu.locator('option').evaluate_all('(xs)=>xs.map(x=>x.value)')
       assert comp.get_attribute('data-playing') in [None,'false'];assert comp.get_attribute('data-running') in [None,'false']
       for value in values:
        menu.select_option(value)
        missing=comp.locator('svg').evaluate_all(r'''svgs=>svgs.flatMap(svg=>{const ids=new Set([...svg.querySelectorAll('[id]')].map(x=>x.id));return [...svg.querySelectorAll('*')].flatMap(e=>[...e.attributes].flatMap(a=>[...a.value.matchAll(/url\(["']?#([^\s)'"%]+)["']?\)/g)].map(m=>m[1]).filter(id=>!ids.has(id))));})''')
        assert not missing,(key,value,missing)
       assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),key
       for value in [values[len(values)//2],values[-1]]:
        menu.select_option(value);comp.screenshot(path=str(out/f"{key.replace(':','--')}-{width}-{theme}-{value}.png"))
       # Full-component screenshots can leave the controls outside the viewport.
       # Test keyboard navigation with the figure visible and its observer settled.
       comp.locator('button').first.evaluate('(button)=>button.scrollIntoView({block:"center",behavior:"instant"})')
       page.evaluate('()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
       before=int(comp.get_attribute('data-step'));comp.focus();page.keyboard.press('p')
       assert int(comp.get_attribute('data-step'))==max(0,before-1),(key,'back')
       page.keyboard.press('n')
       page.wait_for_function('([id,step])=>document.getElementById(id).querySelector(".mlsys-original-player,.life-panel").dataset.step===step',arg=[entry['figure_id'],str(before)])
       page.emulate_media(media='print');assert not comp.is_visible();assert outer.locator('.mlsys-diagram-panel').is_visible()
       assert not errors,(key,errors)
       report.append({'key':key,'width':width,'theme':theme,'states':len(values),'paint_references':'local','keys':'passed','print':'static'})
       page.close()
     b.close()
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'profiles':len(report),'figures':len(manifest),'passed':True}))


if __name__ == "__main__":
    main()
