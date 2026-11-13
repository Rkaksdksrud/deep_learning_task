"""Download and verify a versioned Task2 model ZIP using Python's standard library."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile

class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected is not None and urllib.parse.urlparse(request.full_url).hostname != urllib.parse.urlparse(newurl).hostname:
            redirected.remove_header('Authorization')
        return redirected

def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()

def download_model(run_id, output, archive=None):
    manifest = json.loads(Path(__file__).with_name('manifest.json').read_text(encoding='utf-8'))
    model = next((m for m in manifest['models'] if m['id'] == run_id), None)
    if model is None:
        raise ValueError('Unknown run ID; use --list')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.task2-download-', dir=output.parent) as tmp:
        work = Path(tmp)
        zip_path = work / model['asset_name']
        if archive:
            shutil.copyfile(archive, zip_path)
        else:
            opener = urllib.request.build_opener(SafeRedirect())
            request = urllib.request.Request(model['download_url'], headers={'User-Agent': 'Task2-model-downloader'})
            try:
                with opener.open(request, timeout=120) as response, zip_path.open('wb') as target:
                    shutil.copyfileobj(response, target, 1024*1024)
            except urllib.error.HTTPError as error:
                raise RuntimeError(f'HTTP {error.code}: release asset is unavailable. Check that the release has been published and the URL is accessible.') from error
        if zip_path.stat().st_size != model['archive_size_bytes'] or sha256(zip_path) != model['archive_sha256']:
            raise ValueError('Archive size/SHA256 mismatch; refusing extraction')
        extracted = work / 'model'
        extracted.mkdir()
        with zipfile.ZipFile(zip_path) as z:
            for item in z.infolist():
                name = Path(item.filename)
                if name.is_absolute() or '..' in name.parts or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError('Unsafe ZIP member')
            z.extractall(extracted)
        if sha256(extracted / 'resnet.pt') != model['checkpoint_sha256']:
            raise ValueError('Checkpoint SHA256 mismatch')
        checks = json.loads((extracted / 'SHA256SUMS.json').read_text(encoding='utf-8'))
        for name, expected in checks.items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or sha256(extracted / path) != expected:
                raise ValueError(f'Package file integrity mismatch: {name}')
        extracted.replace(output)
    print(f'Verified model package: {output}')
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--run')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--archive', type=Path, help='Verify/extract a previously downloaded ZIP instead of using the network')
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('manifest.json').read_text(encoding='utf-8'))
    if args.list:
        for model in manifest['models']:
            print(f"{model['id']:<24} BA={model['balanced_accuracy']:.4%}  {model['download_url']}")
        return
    if not args.run:
        parser.error('--run or --list is required')
    target = args.output or Path(__file__).parent / 'downloads' / args.run
    download_model(args.run, target, args.archive)

if __name__ == '__main__':
    main()
