#!/usr/bin/env python3
"""Export canonical chapter LEGO arithmetic; no browser hardware registry."""
import re,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'mlsysim'))
source=ROOT/'books/vol4/08_perception/08_perception.qmd'
namespace={}
for cell in re.findall(r'```\{python\}[^\n]*\n(.*?)\n```',source.read_text(),re.S):
 if any('class '+name+':' in cell for name in ['RollingShutterSkew']):exec(compile(cell,str(source),'exec'),namespace)
R=namespace['RollingShutterSkew']
data=dict(rows=R.rows,row_ms=R.t_row.to('ms').magnitude,v_mm_ms=R.v.to('mm/ms').magnitude,blur_mm=R.blur.to('mm').magnitude,exposure_ms=R.t_exp.to('ms').magnitude)
if __name__=='__main__':
 out=Path(__file__).with_name('data.json');out.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
