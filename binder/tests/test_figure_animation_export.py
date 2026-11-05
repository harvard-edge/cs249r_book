"""The shareable GIF remains finite and identifies the source figure."""
import importlib.util
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/scripts/export_figure_animation.py'
spec = importlib.util.spec_from_file_location('animation_export', SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_gif_has_all_frames_durations_and_no_loop_extension(tmp_path):
    frames = []
    for name, color in [('first', 'red'), ('second', 'blue'), ('last', 'green')]:
        path = tmp_path / (name + '.png')
        Image.new('RGB', (32, 24), color).save(path)
        frames.append(path)
    output = tmp_path / 'preview.gif'
    durations = [3500, 4200, 5100]
    exporter.encode_gif(frames, durations, output)
    with Image.open(output) as gif:
        assert gif.n_frames == len(frames)
        assert 'loop' not in gif.info
        for i, duration in enumerate(durations):
            gif.seek(i)
            assert gif.info['duration'] == duration
            assert gif.size == (32, 24)


def test_source_link_preserves_quarto_path_and_opens_exact_figure():
    assert exporter.canonical_url('books/vol2/05_collective_communication/05_collective_communication.qmd',
                                  'fig-ring-allreduce', 'https://mlsysbook.ai/vol2/') == (
        'https://mlsysbook.ai/vol2/vol2/05_collective_communication/05_collective_communication.html?view=explore#fig-ring-allreduce')


def test_reading_duration_uses_gif_ticks_and_longer_text_gets_more_time():
    short = exporter.frame_duration('One short state.', 3500, 180)
    long = exporter.frame_duration('word ' * 37, 3500, 180)
    assert short == 3500
    assert long > short and long % 10 == 0


def test_capture_expands_scroll_pane_and_rejects_clipped_svg_text():
    import pytest
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1100, 'height': 900})
        page.set_content('''<div id="root" style="width:521px">
          <div class="mlsys-original-stage" style="overflow:auto">
            <div style="overflow:auto"><svg width="660" height="200" style="min-width:660px;overflow:hidden" viewBox="0 0 660 200">
              <text x="530" y="100">Test-log turn</text>
            </svg></div>
          </div><div class="mlsys-original-jump" style="position:fixed;top:40px">Overlay</div>
        </div>''')
        root = page.locator('#root')
        page.add_style_tag(content=exporter.CAPTURE_CSS)
        exporter.prepare_capture(root, 1044)
        geometry = exporter.validate_capture(root)
        assert geometry['width'] >= 660
        assert page.locator('.mlsys-original-jump').evaluate('(e)=>getComputedStyle(e).visibility') == 'hidden'
        page.locator('text').evaluate('(e)=>e.setAttribute("x","650")')
        with pytest.raises(ValueError, match='outside SVG'):
            exporter.validate_capture(root)
        browser.close()
