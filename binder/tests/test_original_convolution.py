"""The live drawing uses reproducible chapter-example values and real products."""
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
EXT=ROOT/'books/_extensions/mlsysbook-ext/figure-explore'


def test_live_values_are_generated_from_the_companion_calculation(tmp_path):
    generator=tmp_path/'convolution-values.calc.py'
    shutil.copyfile(EXT/generator.name,generator)
    subprocess.run(['python3',str(generator)],check=True)
    assert (tmp_path/'convolution-values.js').read_bytes()==(EXT/'convolution-values.js').read_bytes()


def test_every_output_is_the_selected_patch_dot_shared_filter():
    data=json.loads((EXT/'convolution-values.js').read_text().split(' = ',1)[1].rstrip(';\n'))
    weights=data['weights'];inputs=data['inputs']
    rows=len(inputs)-len(weights)+1;columns=len(inputs[0])-len(weights[0])+1
    assert len(data['positions'])==rows*columns
    for index,value in enumerate(data['positions']):
        row,column=divmod(index,columns)
        assert (value['row'],value['column'])==(row,column)
        patch=[[inputs[row+r][column+c] for c in range(len(weights[0]))] for r in range(len(weights))]
        products=[[patch[r][c]*weights[r][c] for c in range(len(weights[0]))] for r in range(len(weights))]
        assert value['patch']==patch and value['products']==products
        assert value['row_sums']==[sum(line) for line in products]
        assert value['output']==sum(map(sum,products))
