#!/usr/bin/env python3
"""Publish the fixed Task2 release with Python's standard library only.

Usage from the repository root:
  python tools/publish_task2_release.py --artifacts ../task2_team_release/task2-resnet18-20261111-v1 --dry-run
  python tools/publish_task2_release.py --artifacts ../task2_team_release/task2-resnet18-20261111-v1 --commit HEAD

Dry-run checks local files only, without authentication or network requests.
Real publication requires GH_TOKEN / GITHUB_TOKEN, or a noninteractive Git HTTPS
credential helper. Push the intended commit separately before running this tool.
On failure the draft is retained. Re-running skips byte-identical uploaded assets;
conflicting/incomplete assets require manual inspection and are never overwritten.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request


REPO = "Rkaksdksrud/deep_learning_task"
OWNER = "Rkaksdksrud"
TAG = "task2-resnet18-20261111-v1"
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models/task2-resnet18-20261111/manifest.json"
API = "https://api.github.com"
EXPECTED_ZIPS = {
    "control_f0_s42.zip", "plain_ce_f0_s42.zip", "geometry_f0_s42.zip",
    "control_f0_s43.zip", "geometry_f0_s43.zip", "control_f1_s42.zip", "geometry_f1_s42.zip",
}
CHUNK_SIZE = 1024 * 1024


class PublishError(Exception):
    """Only controlled, non-sensitive messages may be shown to the user."""


@dataclass(frozen=True)
class Asset:
    name: str
    path: Path
    size: int
    sha256: str


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_assets(directory: Path) -> list[Asset]:
    """Validate the public manifest and every file before any authentication."""
    try:
        manifest_bytes = MANIFEST.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError):
        raise PublishError("Cannot read the repository's Task2 manifest.") from None
    if not isinstance(manifest, dict) or manifest.get("repository") != REPO or manifest.get("release_tag") != TAG:
        raise PublishError("The local manifest does not match the fixed repository and release tag.")
    models = manifest.get("models")
    if not isinstance(models, list) or len(models) != 7 or manifest.get("total_models") != 7:
        raise PublishError("The manifest must describe exactly seven model ZIPs.")
    if manifest.get("contains_training_data") is not False or manifest.get("contains_ensembles") is not False:
        raise PublishError("The manifest must declare that it contains no training data or ensembles.")
    if not all(isinstance(model, dict) for model in models):
        raise PublishError("Invalid model entries in the manifest.")
    names = [model.get("asset_name") for model in models]
    if len(set(name for name in names if isinstance(name, str))) != 7 or set(names) != EXPECTED_ZIPS:
        raise PublishError("The manifest's asset names differ from the seven expected ZIP names.")
    try:
        if not directory.is_dir() or {path.name for path in directory.iterdir()} != EXPECTED_ZIPS | {"manifest.json"}:
            raise PublishError("The artifact directory must contain only the seven ZIPs and manifest.json.")
        if any((directory / name).is_symlink() or not (directory / name).is_file()
               for name in EXPECTED_ZIPS | {"manifest.json"}):
            raise PublishError("Artifact entries must be regular files, not symbolic links.")
        if (directory / "manifest.json").read_bytes() != manifest_bytes:
            raise PublishError("The artifact manifest differs from the repository manifest.")
        assets = []
        for model in models:
            name = model["asset_name"]
            expected_size, expected_sha = model.get("archive_size_bytes"), model.get("archive_sha256")
            if type(expected_size) is not int or expected_size <= 0 or not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                raise PublishError("A model entry has an invalid size or SHA256.")
            path = directory / name
            size = path.stat().st_size
            sha = digest_file(path)
            if size != expected_size or sha != expected_sha:
                raise PublishError(f"Local size/SHA256 mismatch: {name}.")
            assets.append(Asset(name, path, size, sha))
        manifest_path = directory / "manifest.json"
        assets.append(Asset("manifest.json", manifest_path, len(manifest_bytes), hashlib.sha256(manifest_bytes).hexdigest()))
        return assets
    except OSError:
        raise PublishError("Cannot read the local release artifacts.") from None


def authentication_token() -> str:
    """Capture credentials without printing stdout/stderr or invoking a prompt."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        environment = os.environ.copy()
        environment["GIT_TERMINAL_PROMPT"] = "0"
        environment["GCM_INTERACTIVE"] = "never"
        environment["GIT_ASKPASS"] = ""
        environment["SSH_ASKPASS"] = ""
        try:
            completed = subprocess.run(
                ["git", "credential", "fill"], cwd=ROOT,
                input="protocol=https\nhost=github.com\n\n", text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment, timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is not None and completed.returncode == 0:
            fields = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
            token = fields.get("password")
    if not token or not token.strip():
        raise PublishError("No GitHub HTTPS authentication is available. No remote changes were made. Set GH_TOKEN/GITHUB_TOKEN or configure a Git credential helper, then rerun.")
    token = token.strip()
    if any(ord(char) < 32 or ord(char) == 127 for char in token):
        raise PublishError("Invalid GitHub authentication value; no remote changes were made.")
    return token


def resolve_commit(reference: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "--end-of-options", reference + "^{commit}"],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise PublishError("Cannot resolve the requested local commit.") from None
    sha = completed.stdout.strip()
    if completed.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise PublishError("The requested local commit is not a valid Git commit.")
    return sha


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        # JSON calls and uploads must not forward credentials or file bodies.
        return None


class DownloadRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        target = urllib.parse.urljoin(request.full_url, new_url)
        old, new = urllib.parse.urlsplit(request.full_url), urllib.parse.urlsplit(target)
        if request.get_method() not in {"GET", "HEAD"} or new.scheme != "https" or new.username or new.password:
            raise PublishError("Unsafe release-asset download redirect was refused.")
        same_origin = (old.scheme, old.hostname, old.port or 443) == (new.scheme, new.hostname, new.port or 443)
        clean = {key: value for key, value in request.headers.items()
                 if key.lower() not in {"host", "content-length", "content-type"}
                 and (same_origin or key.lower() not in {"authorization", "cookie", "proxy-authorization"})}
        return urllib.request.Request(target, headers=clean, method=request.get_method())


class GitHub:
    def __init__(self, token: str):
        self.headers = {
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "task2-release-publisher",
        }
        self.opener = urllib.request.build_opener(NoRedirect())
        self.download_opener = urllib.request.build_opener(DownloadRedirect())

    def request(self, method: str, path: str, payload=None, allow_missing: bool = False):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = dict(self.headers)
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(API + path, data=body, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=90) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404 and allow_missing:
                return None
            raise PublishError(f"GitHub API request failed (HTTP {error.code}). Any draft remains preserved.") from None
        except (OSError, ValueError, urllib.error.URLError):
            raise PublishError("GitHub API connection/response failed. Any draft remains preserved.") from None

    def list_all(self, path: str) -> list:
        result = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 101):
            values = self.request("GET", f"{path}{separator}per_page=100&page={page}")
            if not isinstance(values, list):
                raise PublishError("GitHub returned an invalid paginated response.")
            result.extend(values)
            if len(values) < 100:
                return result
        raise PublishError("GitHub pagination exceeded the bounded safety limit.")

    def remote_sha256(self, asset: dict) -> str:
        asset_id = positive_id(asset.get("id"))
        headers = {**self.headers, "Accept": "application/octet-stream"}
        request = urllib.request.Request(f"{API}/repos/{REPO}/releases/assets/{asset_id}", headers=headers)
        digest = hashlib.sha256()
        received = 0
        try:
            with self.download_opener.open(request, timeout=180) as response:
                for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                    received += len(chunk)
                    if received > asset["size"]:
                        raise PublishError("Remote asset exceeded its declared size; verification stopped.")
                    digest.update(chunk)
        except (OSError, urllib.error.URLError):
            raise PublishError("Cannot stream-verify an existing release asset. The draft was preserved.") from None
        if received != asset["size"]:
            raise PublishError("Remote asset download was incomplete; verification stopped.")
        return digest.hexdigest()

    def verify_asset(self, remote: dict, local: Asset) -> None:
        if remote.get("name") != local.name or remote.get("size") != local.size or remote.get("state") != "uploaded":
            raise PublishError(f"Remote asset name/size/state conflict: {local.name}. No overwrite was attempted.")
        digest = remote.get("digest")
        if isinstance(digest, str) and digest.startswith("sha256:"):
            remote_sha = digest.removeprefix("sha256:")
        else:
            remote_sha = self.remote_sha256(remote)
        if remote_sha != local.sha256:
            raise PublishError(f"Remote SHA256 conflict: {local.name}. No overwrite was attempted.")

    def upload(self, release_id: int, asset: Asset) -> dict:
        # Construct the known GitHub upload host instead of trusting a returned URL.
        url = f"https://uploads.github.com/repos/{REPO}/releases/{release_id}/assets?" + urllib.parse.urlencode({"name": asset.name})
        headers = {**self.headers, "Content-Type": "application/zip" if asset.name.endswith(".zip") else "application/json", "Content-Length": str(asset.size)}
        try:
            if asset.path.stat().st_size != asset.size:
                raise PublishError(f"Local asset changed after validation: {asset.name}.")
            with asset.path.open("rb") as stream:
                request = urllib.request.Request(url, data=stream, headers=headers, method="POST")
                with self.opener.open(request, timeout=300) as response:
                    return json.load(response)
        except urllib.error.HTTPError as error:
            raise PublishError(f"Asset upload failed (HTTP {error.code}); the draft is preserved. Rerun after inspecting its existing assets.") from None
        except (OSError, ValueError, urllib.error.URLError):
            raise PublishError("Asset upload failed; the draft is preserved. Rerun after inspecting its existing assets.") from None


def positive_id(value) -> int:
    if type(value) is not int or value <= 0:
        raise PublishError("GitHub returned an invalid object identifier.")
    return value


def assets_by_name(values: list, expected: set[str]) -> dict[str, dict]:
    result = {}
    for asset in values:
        if not isinstance(asset, dict) or asset.get("name") not in expected or asset["name"] in result:
            raise PublishError("The release contains unknown or duplicate assets. Nothing was overwritten.")
        result[asset["name"]] = asset
    return result


def asset_signature(asset: dict) -> tuple:
    return tuple(asset.get(key) for key in ("id", "name", "size", "digest", "state", "updated_at"))


def existing_release(api: GitHub):
    release = api.request("GET", f"/repos/{REPO}/releases/tags/{urllib.parse.quote(TAG, safe='')}", allow_missing=True)
    # Authenticated listing also finds drafts if tag lookup does not return one.
    if release is None:
        matches = [entry for entry in api.list_all(f"/repos/{REPO}/releases") if entry.get("tag_name") == TAG]
        if len(matches) > 1:
            raise PublishError("Multiple releases use the fixed tag; manual inspection is required.")
        release = matches[0] if matches else None
    if release is not None and release.get("tag_name") != TAG:
        raise PublishError("The returned release has an unexpected tag.")
    return release


def check_remote_commit_and_tag(api: GitHub, commit: str) -> None:
    remote = api.request("GET", f"/repos/{REPO}/commits/{commit}")
    if not isinstance(remote, dict) or remote.get("sha") != commit:
        raise PublishError("The intended commit is not available in the remote repository; push it separately first.")
    tag = api.request("GET", f"/repos/{REPO}/git/ref/tags/{urllib.parse.quote(TAG, safe='')}", allow_missing=True)
    if tag is None:
        return
    obj = tag.get("object", {})
    for _ in range(8):
        if obj.get("type") == "commit":
            if obj.get("sha") != commit:
                raise PublishError("The existing tag points to a different commit and will not be changed.")
            return
        if obj.get("type") != "tag" or not re.fullmatch(r"[0-9a-f]{40}", str(obj.get("sha", ""))):
            break
        obj = api.request("GET", f"/repos/{REPO}/git/tags/{obj['sha']}").get("object", {})
    raise PublishError("Cannot safely resolve the existing release tag.")


def publish(assets: list[Asset], token: str, commit: str) -> None:
    api = GitHub(token)
    user = api.request("GET", "/user")
    if not isinstance(user, dict) or not isinstance(user.get("login"), str) or not user["login"]:
        raise PublishError("GitHub authentication did not identify a user; no changes were made.")
    repository = api.request("GET", f"/repos/{REPO}")
    if repository.get("full_name", "").lower() != REPO.lower() or repository.get("owner", {}).get("login", "").lower() != OWNER.lower() or repository.get("fork") is not True:
        raise PublishError("Repository identity/fork ownership does not match the intended fork; no changes were made.")
    if repository.get("permissions", {}).get("push") is not True:
        raise PublishError("The authenticated user has no push permission on the intended fork; no changes were made.")
    check_remote_commit_and_tag(api, commit)
    remote_manifest = api.request(
        "GET", f"/repos/{REPO}/contents/models/task2-resnet18-20261111/manifest.json?ref={commit}")
    try:
        manifest_bytes = base64.b64decode(remote_manifest["content"], validate=False)
    except (KeyError, TypeError, ValueError):
        raise PublishError("The intended remote commit has no readable Task2 manifest; no changes were made.") from None
    manifest_asset = next(asset for asset in assets if asset.name == "manifest.json")
    if remote_manifest.get("encoding") != "base64" or hashlib.sha256(manifest_bytes).hexdigest() != manifest_asset.sha256:
        raise PublishError("The remote commit's manifest differs from the validated local artifacts; no changes were made.")
    release = existing_release(api)
    expected = {asset.name for asset in assets}
    if release is not None:
        release_id = positive_id(release.get("id"))
        published = release.get("draft") is False
        if release.get("draft") not in (True, False):
            raise PublishError("GitHub returned an invalid draft status.")
        target = release.get("target_commitish")
        if not isinstance(target, str) or not target:
            raise PublishError("The existing release has no verifiable target commit.")
        resolved_target = api.request("GET", f"/repos/{REPO}/commits/{urllib.parse.quote(target, safe='')}")
        if resolved_target.get("sha") != commit:
            raise PublishError("The existing release targets a different commit and will not be changed.")
        remote_assets = assets_by_name(api.list_all(f"/repos/{REPO}/releases/{release_id}/assets"), expected)
        if published:
            if set(remote_assets) != expected:
                raise PublishError("The published release is immutable and does not contain the exact expected asset set.")
            for asset in assets:
                api.verify_asset(remote_assets[asset.name], asset)
            print("The existing published release already matches all eight local assets; no changes made.")
            return
    else:
        # This is the first write. Identity, permissions, commit and local hashes were checked above.
        release = api.request("POST", f"/repos/{REPO}/releases", {
            "tag_name": TAG, "target_commitish": commit,
            "name": "Task2 ResNet18: seven patient-holdout experiments (2026-11-11)",
            "body": "Seven independently evaluated ResNet18 holdout checkpoints. These are not all-data final submission models. No training images, labels, or per-sample predictions are included. See manifest.json and the repository documentation for metrics, SHA256 hashes, and usage.",
            "draft": True, "prerelease": False,
        })
        release_id = positive_id(release.get("id"))
        if release.get("draft") is not True:
            raise PublishError("GitHub did not create a draft as requested; publication stopped.")
        remote_assets = {}
        print("Created a draft release. Errors will leave it available for inspection/resumption.")

    checked = {}
    for asset in assets:
        remote = remote_assets.get(asset.name)
        if remote is None:
            print(f"Uploading {asset.name} ({asset.size} bytes)...", flush=True)
            remote = api.upload(release_id, asset)
        else:
            print(f"Checking existing {asset.name}...", flush=True)
        api.verify_asset(remote, asset)
        checked[asset.name] = asset_signature(remote)
    # Fresh listing checks the complete set immediately before publication.
    final_assets = assets_by_name(api.list_all(f"/repos/{REPO}/releases/{release_id}/assets"), expected)
    if set(final_assets) != expected:
        raise PublishError("The draft does not contain all eight expected assets; it was left unpublished.")
    for asset in assets:
        if asset_signature(final_assets[asset.name]) != checked[asset.name]:
            api.verify_asset(final_assets[asset.name], asset)
    latest = api.request("GET", f"/repos/{REPO}/releases/{release_id}")
    if latest.get("draft") is not True or latest.get("tag_name") != TAG:
        raise PublishError("The draft status/tag changed externally; no publication update was attempted.")
    latest_target = latest.get("target_commitish")
    if not isinstance(latest_target, str) or not latest_target:
        raise PublishError("The draft target cannot be verified; no publication update was attempted.")
    latest_commit = api.request("GET", f"/repos/{REPO}/commits/{urllib.parse.quote(latest_target, safe='')}")
    if latest_commit.get("sha") != commit:
        raise PublishError("The draft target commit changed; no publication update was attempted.")
    check_remote_commit_and_tag(api, commit)
    result = api.request("PATCH", f"/repos/{REPO}/releases/{release_id}", {"draft": False})
    if result.get("draft") is not False or result.get("tag_name") != TAG:
        raise PublishError("Cannot confirm the final published release status; inspect GitHub before rerunning.")
    print(f"Published https://github.com/{REPO}/releases/tag/{TAG}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, required=True, help="Directory containing exactly seven ZIPs and manifest.json")
    parser.add_argument("--commit", default="HEAD", help="Already-pushed local Git commit or ref (default: HEAD)")
    parser.add_argument("--dry-run", action="store_true", help="Validate local sizes and SHA256 only; no auth or network")
    args = parser.parse_args()
    try:
        assets = local_assets(args.artifacts.expanduser().resolve())
        for asset in assets:
            print(f"OK {asset.name}: {asset.size} bytes; sha256={asset.sha256}")
        if args.dry_run:
            print("Dry-run passed: seven ZIPs plus manifest verified; no authentication or network used.")
            return 0
        token = authentication_token()
        commit = resolve_commit(args.commit)
        publish(assets, token, commit)
        return 0
    except PublishError as error:
        print(f"Stopped: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped by user. Any draft and uploaded assets are preserved.", file=sys.stderr)
        return 130
    except Exception as error:
        # Do not echo arbitrary exception text, URLs, credential-helper output, or local paths.
        print(f"Stopped after an unexpected {type(error).__name__}; inspect local inputs or the preserved draft. No automatic overwrite/delete was attempted.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
