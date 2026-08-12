#!/usr/bin/env python3
"""One-shot bulk .m_as() -> .to(unit).magnitude migration for LEGO cells."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPLACEMENTS = [
    (".m_as(TFLOPs / second)", ".to(TFLOPs / second).magnitude"),
    (".m_as(TFLOPs/second)", ".to(TFLOPs/second).magnitude"),
    (".m_as(PFLOPs/second)", ".to(PFLOPs/second).magnitude"),
    (".m_as(PFLOPs / second)", ".to(PFLOPs / second).magnitude"),
    (".m_as(TB/second)", ".to(TB/second).magnitude"),
    (".m_as(TB / second)", ".to(TB / second).magnitude"),
    (".m_as(GB / second)", ".to(GB / second).magnitude"),
    (".m_as(GB/second)", ".to(GB/second).magnitude"),
    (".m_as(MB/second)", ".to(MB/second).magnitude"),
    (".m_as(MB / second)", ".to(MB / second).magnitude"),
    (".m_as(byte/second)", ".to(byte/second).magnitude"),
    (".m_as('byte/second')", ".to(byte/second).magnitude"),
    (".m_as(flop/second)", ".to(flop/second).magnitude"),
    (".m_as(flop / second)", ".to(flop / second).magnitude"),
    (".m_as(flop/byte)", ".to(flop/byte).magnitude"),
    (".m_as('flop/byte')", ".to(flop/byte).magnitude"),
    ('.m_as("flop/byte")', '.to(flop/byte).magnitude'),
    (".m_as(GiB)", ".to(GiB).magnitude"),
    (".m_as(Bparam)", ".to(Bparam).magnitude"),
    (".m_as(Mparam)", ".to(Mparam).magnitude"),
    (".m_as(Tparam)", ".to(Tparam).magnitude"),
    (".m_as(Kparam)", ".to(Kparam).magnitude"),
    (".m_as(GFLOPs)", ".to(GFLOPs).magnitude"),
    (".m_as(MFLOPs)", ".to(MFLOPs).magnitude"),
    (".m_as(TFLOPs)", ".to(TFLOPs).magnitude"),
    (".m_as(TOPS)", ".to(TOPS).magnitude"),
    (".m_as(PB)", ".to(PB).magnitude"),
    (".m_as(TB)", ".to(TB).magnitude"),
    (".m_as(GB)", ".to(GB).magnitude"),
    (".m_as(MB)", ".to(MB).magnitude"),
    (".m_as(KB)", ".to(KB).magnitude"),
    (".m_as('MB')", ".to(MB).magnitude"),
    (".m_as(byte)", ".to(byte).magnitude"),
    (".m_as(watt)", ".to(watt).magnitude"),
    (".m_as(kW)", ".to(kW).magnitude"),
    (".m_as(kilowatt)", ".to(kW).magnitude"),
    (".m_as(MW)", ".to(MW).magnitude"),
    (".m_as(milliwatt)", ".to(mW).magnitude"),
    (".m_as(mW)", ".to(mW).magnitude"),
    (".m_as(Gbps)", ".to(Gbps).magnitude"),
    (".m_as(param)", ".to(param).magnitude"),
    (".m_as('param')", ".to(param).magnitude"),
    (".m_as(count)", ".to(count).magnitude"),
    (".m_as('count')", ".to(count).magnitude"),
    (".m_as('B')", ".to(byte).magnitude"),
    (".m_as(second)", ".to(second).magnitude"),
    (".m_as('second')", ".to(second).magnitude"),
    ('.m_as("second")', ".to(second).magnitude"),
    (".m_as(hour)", ".to(hour).magnitude"),
    ('.m_as("hour")', ".to(hour).magnitude"),
    (".m_as(ms)", ".to(ms).magnitude"),
    (".m_as('ms')", ".to(ms).magnitude"),
    ('.m_as("ms")', ".to(ms).magnitude"),
    (".m_as(NS)", ".to(NS).magnitude"),
    (".m_as(kWh)", ".to(kWh).magnitude"),
    ('.m_as("kWh")', ".to(kWh).magnitude"),
    (".m_as(MWh)", ".to(MWh).magnitude"),
    (".m_as(Wh)", ".to(Wh).magnitude"),
    ('.m_as("Wh")', ".to(Wh).magnitude"),
    (".m_as(ureg.kilowatt_hour)", ".to(kWh).magnitude"),
    (".m_as(ureg.picojoule)", ".to(pJ).magnitude"),
    (".m_as(ureg.picojoule/ureg.count)", ".to(pJ).magnitude"),
    (".m_as(ureg.picojoule / ureg.count)", ".to(pJ).magnitude"),
    (".m_as(ureg.millijoule)", ".to(mJ).magnitude"),
    (".m_as(ureg.byte)", ".to(byte).magnitude"),
    (".m_as(ureg.second)", ".to(second).magnitude"),
    (".m_as(ureg.hour)", ".to(hour).magnitude"),
    (".m_as(ureg.minute)", ".to(minute).magnitude"),
    (".m_as('pJ')", ".to(pJ).magnitude"),
    ('.m_as("")', '.to("").magnitude'),
    ('.m_as(\'\')', '.to("").magnitude'),
    ('.m_as("count")', '.to("").magnitude'),
    ('.m_as("dollar / hour")', ".to(USD / hour).magnitude"),
    ('.m_as("dollar / kWh")', ".to(USD / kWh).magnitude"),
    ('.m_as("dollar / kilowatt_hour")', ".to(USD / kWh).magnitude"),
    ('.m_as("dollar / GB")', ".to(USD / GB).magnitude"),
    ('.m_as("dollar / GB / month")', ".to(USD / GB / ureg.month).magnitude"),
    ('.m_as("dollar / TB / month")', ".to(USD / TB / ureg.month).magnitude"),
    ('.m_as("dollar")', ".to(USD).magnitude"),
    (".m_as(USD)", ".to(USD).magnitude"),
    (".m_as(US)", ".to(US).magnitude"),
    (".m_as(c.param)", ".to(c.param).magnitude"),
    (".m_as(c.flop / c.second)", ".to(c.flop / c.second).magnitude"),
    (".m_as(c.byte)", ".to(c.byte).magnitude"),
    (".m_as(c.GB)", ".to(c.GB).magnitude"),
    (".m_as(c.second)", ".to(c.second).magnitude"),
    (".m_as(c.hour)", ".to(c.hour).magnitude"),
    (".m_as(c.kWh)", ".to(c.kWh).magnitude"),
    (".m_as(c.kW)", ".to(c.kW).magnitude"),
    (".m_as(c.MW)", ".to(c.MW).magnitude"),
    (".m_as(c.watt)", ".to(c.watt).magnitude"),
    (".m_as(c.TFLOPs / c.second)", ".to(c.TFLOPs / c.second).magnitude"),
    (".m_as(c.GB / c.second)", ".to(c.GB / c.second).magnitude"),
    (".m_as(c.TB / c.second)", ".to(c.TB / c.second).magnitude"),
    (".m_as(c.flop / c.byte)", ".to(c.flop / c.byte).magnitude"),
    (".m_as(ureg.km)", ".to(km).magnitude"),
    (".m_as(ureg.kg)", ".to(kg).magnitude"),
    (".m_as(ureg.gram)", ".to(gram).magnitude"),
    (".m_as(ureg.degC)", ".to(ureg.degC).magnitude"),
    (".m_as('millisecond')", ".to(ms).magnitude"),
    (".m_as('hour')", ".to(hour).magnitude"),
    (".m_as('GB')", ".to(GB).magnitude"),
    (".m_as('MWh')", ".to(MWh).magnitude"),
    (".m_as('kW')", ".to(kW).magnitude"),
    (".m_as('MW')", ".to(MW).magnitude"),
    (".m_as(TFLOP / second)", ".to(TFLOP / second).magnitude"),
    (".m_as(TFLOP/second)", ".to(TFLOP/second).magnitude"),
    (".m_as(byte / second)", ".to(byte / second).magnitude"),
    (".m_as(count / second)", ".to(count / second).magnitude"),
    (".m_as(watt * hour)", ".to(watt * hour).magnitude"),
    (".m_as(joule)", ".to(joule).magnitude"),
    (".m_as(day)", ".to(day).magnitude"),
    (".m_as(flop)", ".to(flop).magnitude"),
    (".m_as('GB/s')", ".to(GB / second).magnitude"),
    (".m_as('GFLOPs')", ".to(GFLOPs).magnitude"),
    (".m_as('TFLOP/s')", ".to(TFLOP / second).magnitude"),
    (".m_as('Gbps')", ".to(Gbps).magnitude"),
    (".m_as('byte')", ".to(byte).magnitude"),
    ('.m_as("byte")', ".to(byte).magnitude"),
    ('.m_as("MB")', ".to(MB).magnitude"),
    ('.m_as("GB")', ".to(GB).magnitude"),
    ('.m_as("J")', ".to(joule).magnitude"),
    ('.m_as("minute")', ".to(minute).magnitude"),
    (".m_as('bit/second')", ".to(bit / second).magnitude"),
    (".m_as('ns')", ".to(NS).magnitude"),
    (".m_as('pJ/B')", ".to(pJ / byte).magnitude"),
    (".m_as('kibibyte')", ".to(KiB).magnitude"),
    (".m_as('mebibyte')", ".to(MiB).magnitude"),
    (".m_as('megabyte')", ".to(MB).magnitude"),
    (".m_as('nanosecond')", ".to(NS).magnitude"),
    (".m_as('picojoule')", ".to(pJ).magnitude"),
    (".m_as(ureg.ms)", ".to(ms).magnitude"),
    (".m_as(ureg.microjoule)", ".to(ureg.microjoule).magnitude"),
    (".m_as(ureg.kilogram)", ".to(kg).magnitude"),
    (".m_as(ureg.megawatt)", ".to(MW).magnitude"),
    (".m_as(ureg.dimensionless)", '.to("").magnitude'),
    (".m_as(ureg.count)", ".to(count).magnitude"),
    (".m_as(ureg.millisecond)", ".to(ms).magnitude"),
    (".m_as(ureg.flop)", ".to(flop).magnitude"),
    (".m_as(ureg.picojoule/ureg.byte)", ".to(pJ / byte).magnitude"),
    (".m_as(ureg.picojoule/byte)", ".to(pJ / byte).magnitude"),
    (".m_as(ureg.picojoule/ureg.flop)", ".to(pJ / flop).magnitude"),
    (".m_as(ureg.kilometer / ureg.millisecond)", ".to(km / ms).magnitude"),
    (".m_as(Models.Language.GPT3.training_ops.units)", ".to(Models.Language.GPT3.training_ops.units).magnitude"),
]


def _regex_pass(text: str) -> str:
    """Replace remaining .m_as(...) on non-comment lines (incl. multiline)."""
    def repl(match: re.Match[str]) -> str:
        start = match.start()
        line_start = text.rfind("\n", 0, start) + 1
        prefix = text[line_start:start]
        if prefix.lstrip().startswith("#"):
            return match.group(0)
        inner = match.group(1).strip()
        return f".to({inner}).magnitude"

    return re.sub(r"\.m_as\(\s*([^)]+)\s*\)", repl, text, flags=re.DOTALL)


def migrate_file(path: Path) -> int:
    text = path.read_text()
    before = text.count(".m_as(")
    if before == 0:
        return 0
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    text = _regex_pass(text)
    path.write_text(text)
    return before - text.count(".m_as(")


def main() -> int:
    root = Path(__file__).resolve().parents[2] / "quarto" / "contents"
    paths = sorted(root.rglob("*.qmd"))
    total_fixed = 0
    remaining: list[tuple[str, int]] = []
    for p in paths:
        n = p.read_text().count(".m_as(")
        if n:
            migrate_file(p)
            left = p.read_text().count(".m_as(")
            total_fixed += n - left
            if left:
                remaining.append((str(p.relative_to(root)), left))
    print(f"Fixed {total_fixed} occurrences")
    if remaining:
        print(f"Remaining in {len(remaining)} files:")
        for path, count in remaining:
            print(f"  {count:4d}  {path}")
        return 1
    print("All .m_as() migrated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
