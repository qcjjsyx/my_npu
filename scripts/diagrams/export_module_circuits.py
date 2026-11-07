"""Export existing draw.io pages using configurable local tools."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'assets/diagrams/generated/module_circuits'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pages', type=int, nargs='*', help='page numbers 1..6 (default: all)')
    args = parser.parse_args()
    home = Path.home()
    runtime_node = home / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
    node = os.environ.get('DRAWIO_NODE') or shutil.which('node') or str(runtime_node)
    skill_scripts = home / '.codex/skills/draw-io/scripts'
    helper = os.environ.get('DRAWIO_EXPORT_HELPER', str(skill_scripts / 'export-drawio.mjs'))
    lint = os.environ.get('DRAWIO_LINT_HELPER', str(skill_scripts / 'check-drawio-svg-overlaps.mjs'))
    app = os.environ.get('DRAWIO_APP', '/Applications/draw.io.app/Contents/MacOS/draw.io')
    for tool in (node, helper, lint, app):
        if not Path(tool).is_file():
            parser.error(f'local tool not found: {tool}; configure DRAWIO_* environment variables')
    indices = args.pages or list(range(1, 7))
    if any(i < 1 or i > 6 for i in indices):
        parser.error('page numbers must be in 1..6')
    OUT.mkdir(parents=True, exist_ok=True)
    for index in indices:
        source = OUT / f'{index:02d}.drawio'
        for fmt in ('svg', 'png'):
            subprocess.run([node, helper, str(source), '--format', fmt,
                            '--drawio', app, '--no-transparent'], check=True)
        result = subprocess.run([node, lint, str(source)], text=True, capture_output=True)
        (OUT / f'{index:02d}.lint.txt').write_text(result.stdout + result.stderr)
        print(f'Page {index} lint exit {result.returncode}', flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    subprocess.run([node, helper, str(ROOT / 'assets/diagrams/source/各模块电路图.drawio'),
                    '--format', 'pdf', '--all-pages', '--output', str(OUT / '各模块电路图_预览.pdf'),
                    '--drawio', app], check=True)


if __name__ == '__main__':
    main()
