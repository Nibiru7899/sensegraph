import hashlib
import io
import zipfile

import pytest

from sensegraph.acquisition import BudgetExceeded, Downloader, RemoteZipReader


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.url = "https://example.org/asset"
        self.headers = headers or {"Content-Length": str(len(data))}


def test_download_resume_and_hash(tmp_path, monkeypatch):
    partial = tmp_path / "asset.bin.part"
    partial.write_bytes(b"abc")
    seen = []
    def response(request, timeout):
        seen.append(request.get_header("Range"))
        return Response(b"def", 206, {"Content-Length": "3", "Content-Range": "bytes 3-5/6"})
    monkeypatch.setattr("urllib.request.urlopen", response)
    downloader = Downloader(tmp_path, 100)
    path = downloader.fetch("https://example.org/asset", "asset.bin", hashlib.sha256(b"abcdef").hexdigest(), 6)
    assert path.read_bytes() == b"abcdef"
    assert seen == ["bytes=3-"]
    assert downloader.ledger["transferred_bytes"] == 3
    # A cached repeat performs no network request and no extra byte charge.
    downloader.fetch("https://example.org/asset", "asset.bin")
    assert seen == ["bytes=3-"]


def test_download_rejects_size_before_body(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(b"x" * 100))
    with pytest.raises(BudgetExceeded):
        Downloader(tmp_path, 50).fetch("https://example.org/asset", "large.bin")
    assert not (tmp_path / "large.bin").exists()


def test_download_hash_mismatch_not_published(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(b"bad"))
    with pytest.raises(IOError, match="SHA256"):
        Downloader(tmp_path).fetch("https://example.org/asset", "asset.bin", "0" * 64)
    assert not (tmp_path / "asset.bin").exists()
    assert (tmp_path / "asset.bin.part").exists()


def test_selective_zip_reads_only_chosen_member(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as zipped:
        zipped.writestr("train/chart.png", b"training")
        zipped.writestr("test/chart.png", b"x" * 200_000)
    payload = buffer.getvalue()
    ranges = []
    def response(request, timeout):
        if request.get_method() == "HEAD":
            return Response(b"", headers={"Content-Length": str(len(payload))})
        start, end = map(int, request.get_header("Range")[6:].split("-"))
        ranges.append((start, end))
        return Response(payload[start:end + 1], 206, {"Content-Range": f"bytes {start}-{end}/{len(payload)}"})
    monkeypatch.setattr("sensegraph.acquisition.archive_open", response)
    downloader = Downloader(tmp_path)
    with zipfile.ZipFile(RemoteZipReader(downloader, "https://example.org/archive.zip")) as zipped:
        assert zipped.read("train/chart.png") == b"training"
    assert downloader.ledger["transferred_bytes"] < 2000
