"""Freeze a local versioned draft snapshot; never deploy or certify a release.

Run concepts build and build --check first. Usage: archive_taxonomy.py OUTPUT_DIR
The output must not exist. Its manifest pins every included byte, including the
source and provenance; no timestamps or study outcomes are invented.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[3]


def archive(root, output):
    output = output.resolve()
    if output.exists():
        raise ValueError('Archive output already exists; snapshots are immutable')
    subprocess.run([str(root/'binder/binder'), 'concepts', 'build', '--check'], cwd=root, check=True)
    source = root/'taxonomy/taxonomy.yml'
    version = yaml.safe_load(source.read_text())['version']
    if not version.endswith('-draft'):
        raise ValueError('This command archives editorial drafts, not reviewed releases')
    paths = [root/'taxonomy'/name for name in ('taxonomy.yml', 'provenance.yml', 'METHOD.md', 'README.md', 'CHANGELOG.md', 'schema/mlsys-taxonomy.linkml.yml')]
    paths += sorted(p for p in (root/'taxonomy/downloads').iterdir() if p.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='taxonomy-archive-', dir=output.parent) as temporary:
        stage = Path(temporary)/'snapshot'
        stage.mkdir()
        hashes = {}
        for path in paths:
            relative = path.relative_to(root)
            target = stage/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest = dict(status='editorial-draft', version=version,
                        revision=hashes['taxonomy/taxonomy.yml'], files=hashes)
        (stage/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        if output.exists():
            raise ValueError('Output was created concurrently; refusing to replace it')
        stage.rename(output)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = archive(ROOT, args.output)
    print(f"Archived {result['version']} revision {result['revision']} at {args.output}")
