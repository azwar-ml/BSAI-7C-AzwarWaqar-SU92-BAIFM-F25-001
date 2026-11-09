"""Run deliverables sequentially with the invoking (project venv) interpreter."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile
from PIL import Image
from pypdf import PdfReader
import fitz

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for week in (6, 7):
        folder = ROOT / f'week{week}' / 'assignments'
        raw = {p: digest(p) for p in (folder / 'data').iterdir() if p.is_file()}
        scripts = ['selfcheck.py', 'partA_process.py', 'partA_charts.py',
                   'partB_charts.py', 'partB_analytics.py' if week == 6 else 'partB_analysis.py',
                   'partA_dashboard.py', 'partC_experiment.py',
                   'partC_figure.py' if week == 6 else 'partC_interactive.py',
                   'partD_figure.py', 'generate_partE_pdf.py']
        if week == 7:
            scripts += ['partD_screen.py', 'dashlib.py']
        log = []
        for script in scripts:
            print(f'Week {week}: {script}', flush=True)
            result = subprocess.run([sys.executable, script], cwd=folder, capture_output=True, text=True, encoding='utf-8', env={**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8'})
            log.append(f'>>> {script}\n{result.stdout}\n{result.stderr}')
            (folder / 'verification_stdout.txt').write_text('\n'.join(line.rstrip() for line in '\n'.join(log).splitlines()).rstrip() + '\n', encoding='utf-8')
            if result.returncode:
                print(result.stdout, result.stderr)
                raise RuntimeError(f'{script} exited {result.returncode}')
        assert raw == {p: digest(p) for p in raw}, 'Raw input changed'
        for image in folder.glob('part*.png'):
            with Image.open(image) as im:
                assert all(abs(x - 300) < .1 for x in im.info['dpi']), image
                assert min(im.size) > 500
        pdf = folder / 'partE_justification.pdf'
        assert len(PdfReader(pdf).pages) == 1
        preview = ROOT / 'tmp' / 'review'
        preview.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(pdf)
        doc[0].get_pixmap(matrix=fitz.Matrix(1.3, 1.3)).save(preview / f'week{week}_pdf.png')
        doc.close()
        for image in folder.glob('part*.png'):
            with Image.open(image) as im:
                im.thumbnail((1300, 1100))
                im.convert('RGB').save(preview / f'week{week}_{image.stem}.jpg')
        html = folder / ('partA_dashboard.html' if week == 6 else 'partC_interactive.html')
        text = html.read_text(encoding='utf-8')
        assert 'Reset: all' in text and 'connectgaps' in text and 'rangeslider' in text
        assert html.stat().st_size > 1000000, 'Expected embedded offline Plotly runtime'
        # Keep each week independently runnable from its submission ZIP.
        archive = ROOT / f'week{week}_submission.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for p in folder.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('_') and not p.name.startswith('run_'):
                    z.write(p, p.relative_to(folder))
        summary = {'week': week, 'python': sys.version.split()[0], 'executed': scripts,
                   'raw_sha256': {str(p.relative_to(folder)): h for p, h in raw.items()},
                   'raw_unchanged': True, 'png_dpi': 300, 'pdf_pages': 1,
                   'offline_html': html.name, 'human_trials': 'not performed or claimed'}
        (folder / 'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('PASS: both weeks executed; raw hashes unchanged; PNGs 300 DPI; PDFs one page; offline HTML controls present.')


if __name__ == '__main__':
    main()
