"""Download one verified Task2 training-settings model package using the standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile

FILES = {"model.pt", "predict.py", "requirements.txt", "metadata.json", "README.md", "SHA256SUMS.json"}


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected is not None and urllib.parse.urlparse(request.full_url).hostname != urllib.parse.urlparse(newurl).hostname:
            redirected.remove_header("Authorization")
        return redirected


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest():
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("checkpoint_filename") != "model.pt":
        raise ValueError("Unsupported model manifest or weight filename")
    ids = [row["id"] for row in manifest["models"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate run IDs in model manifest")
    return manifest


def download_model(run_id, output, archive=None):
    manifest = read_manifest()
    model = next((row for row in manifest["models"] if row["id"] == run_id), None)
    if model is None:
        raise ValueError("Unknown or not-yet-published run ID; use --list")
    if model.get("checkpoint_filename") != "model.pt" or model["asset_name"] != f"{run_id}.zip":
        raise ValueError("Inconsistent manifest filenames")
    if not re.fullmatch(r"[a-z0-9_]+", run_id):
        raise ValueError("Unsafe run ID")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".task2-training-settings-download-", dir=output.parent) as temporary:
        work = Path(temporary)
        zip_path = work / model["asset_name"]
        if archive is not None:
            shutil.copyfile(archive, zip_path)
        else:
            opener = urllib.request.build_opener(SafeRedirect())
            request = urllib.request.Request(model["download_url"], headers={"User-Agent": "Task2-training-settings-downloader"})
            try:
                with opener.open(request, timeout=120) as response, zip_path.open("wb") as target:
                    shutil.copyfileobj(response, target, 1024 * 1024)
            except urllib.error.HTTPError as error:
                raise RuntimeError(f"HTTP {error.code}: check that the release is published and accessible") from error
        if zip_path.stat().st_size != model["archive_size_bytes"] or sha256(zip_path) != model["archive_sha256"]:
            raise ValueError("Archive size/SHA256 mismatch; refusing extraction")
        extracted = work / "model"
        extracted.mkdir()
        with zipfile.ZipFile(zip_path) as archive_file:
            members = archive_file.infolist()
            if len(members) != len(FILES) or {item.filename for item in members} != FILES:
                raise ValueError("Package must contain exactly the six documented files")
            for item in members:
                if item.is_dir() or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Unsafe ZIP member")
                # FILES contains only literal flat names; nested/absolute paths,
                # Windows path separators and duplicate entries are all rejected.
            archive_file.extractall(extracted)
        checks = json.loads((extracted / "SHA256SUMS.json").read_text(encoding="utf-8"))
        if set(checks) != FILES - {"SHA256SUMS.json"}:
            raise ValueError("Internal checksum list does not cover every package file")
        for name, expected in checks.items():
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256(extracted / name) != expected:
                raise ValueError(f"Package file integrity mismatch: {name}")
        if checks["model.pt"] != model["checkpoint_sha256"]:
            raise ValueError("Checkpoint SHA256 mismatch")
        metadata = json.loads((extracted / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("run_id") != run_id or metadata.get("checkpoint_filename") != "model.pt" or metadata.get("checkpoint_sha256") != checks["model.pt"]:
            raise ValueError("Package metadata disagrees with the manifest")
        extracted.replace(output)
    print(f"Verified model package: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path, help="Verify/extract a local ZIP without network access")
    args = parser.parse_args()
    manifest = read_manifest()
    if args.list:
        for model in manifest["models"]:
            print(f"{model['id']:<40} BA={model['balanced_accuracy']:.4%}  {model['download_url']}")
        for record in manifest.get("omitted_runs", []):
            print(f"{record['id']:<40} {record['status']}")
        return
    if not args.run:
        parser.error("--run or --list is required")
    target = args.output or Path(__file__).parent / "downloads" / args.run
    download_model(args.run, target, args.archive)


if __name__ == "__main__":
    main()
