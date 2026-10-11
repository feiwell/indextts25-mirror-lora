import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path('/root/voice-training/indextts25-mirror-20261010')
REPO = ROOT / 'repo'


def main():
    for name in ['gpt-training-report.json', 'cfm-training-report.json', 'evaluation-metrics.json']:
        report = json.loads((ROOT / name).read_text())
        if name.endswith('training-report.json') and not report['ok']:
            raise RuntimeError('Unsuccessful training: ' + name)
        if name == 'evaluation-metrics.json' and (len(report['rows']) != 30 or len(report['summaries']) != 5):
            raise RuntimeError('Incomplete evaluation')
    reproduced = json.loads((ROOT / 'inference-reproduction.json').read_text())
    if not reproduced['standalone_inference_matches_delivered_exact']:
        raise RuntimeError('Standalone inference was not verified')
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    with tarfile.open(ROOT / 'source-main.tar.gz') as source:
        provenance['archive_commit'] = source.pax_headers.get('comment')
    (ROOT / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    versions = sorted((dist.metadata['Name'], dist.version) for dist in importlib.metadata.distributions())
    (ROOT / 'installed-packages.json').write_text(json.dumps(versions, indent=2))
    hardware = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True).stdout
    (ROOT / 'gpu-final.txt').write_text(hardware)
    paths = [p for p in ROOT.iterdir() if p.is_file() and p.name != 'result-checksums.json'
             and p.suffix in {'.json', '.log', '.py', '.sh', '.txt'}]
    paths += [ROOT / 'listening', ROOT / 'reference.wav', REPO / 'training_runs',
              REPO / 'datasets/mirror-verified-asr', REPO / 'checkpoints/config.yaml',
              REPO / 'pyproject.toml', REPO / 'uv.lock']
    checksums = {}
    for path in paths:
        files = sorted(path.rglob('*')) if path.is_dir() else [path]
        for file in files:
            if file.is_file() and not file.is_symlink():
                checksums[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    (ROOT / 'result-checksums.json').write_text(json.dumps(checksums, indent=2))
    paths.append(ROOT / 'result-checksums.json')
    destination = ROOT / 'results.tar.gz'
    with tarfile.open(destination, 'w:gz') as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    print('RESULT PACKAGE', destination.stat().st_size,
          hashlib.sha256(destination.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
