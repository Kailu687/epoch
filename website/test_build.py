"""Tests for release-manifest gating. Run: python -m pytest test_build.py"""
import hashlib
import tempfile
from pathlib import Path

import pytest

import build


@pytest.fixture
def env(tmp_path, monkeypatch):
    rel = tmp_path / "releases"
    (rel / "artifacts").mkdir(parents=True)
    monkeypatch.setattr(build, "REL_DIR", rel)
    monkeypatch.setattr(build, "DIST", tmp_path / "dist")
    return rel


def art(rel, content=b"hello", **kw):
    (rel / "artifacts" / "a.zip").write_bytes(content)
    d = dict(platform="windows", arch="x64", package="zip", name="a.zip", file="a.zip",
             sha256=hashlib.sha256(content).hexdigest(), size=len(content))
    d.update(kw)
    return d


def test_verified_file_is_downloadable(env):
    ok, href, _ = build.verify_artifact(art(env), "1.0.0")
    assert ok and href == "downloads/1.0.0/a.zip"
    assert (build.DIST / "downloads/1.0.0/a.zip").is_file()


def test_missing_file_not_downloadable(env):
    a = art(env); (env / "artifacts" / "a.zip").unlink()
    assert build.verify_artifact(a, "1.0.0")[0] is False


def test_checksum_mismatch_not_downloadable(env):
    assert build.verify_artifact(art(env, sha256="0" * 64), "1.0.0")[0] is False


def test_missing_checksum_not_downloadable(env):
    a = art(env); a["sha256"] = None
    assert build.verify_artifact(a, "1.0.0")[0] is False


def test_path_traversal_rejected(env):
    a = art(env, file="../releases.json")
    assert build.verify_artifact(a, "1.0.0")[0] is False


def test_unverified_url_not_downloadable(env):
    a = dict(platform="windows", arch="x64", package="zip", name="a", url="https://example.invalid/a.zip")
    assert build.verify_artifact(a, "1.0.0")[0] is False
    a["url_verified_at"] = "2026-01-01T00:00:00Z"
    assert build.verify_artifact(a, "1.0.0")[0] is True


def test_semver_ordering():
    vs = ["0.9.0", "0.10.0", "1.0.0-beta.1", "1.0.0"]
    assert sorted(vs, key=build.semver_key, reverse=True) == ["1.0.0", "1.0.0-beta.1", "0.10.0", "0.9.0"]


def test_invalid_manifest_detected():
    errs = build.validate_manifest({"releases": [{"version": "x", "channel": "foo", "status": "bar", "artifacts": [{}]}]})
    assert len(errs) >= 4
