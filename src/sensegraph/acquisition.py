"""Serial HTTPS transfers with a persistent byte budget and resumable artifacts."""
import urllib.request
import urllib.error
import io
import http.client
import socket


def _connect_ipv4_first(address, timeout, source_address=None):
    """Keep TLS hostname validation while avoiding slow NAT64-first connections."""
    addresses = socket.getaddrinfo(address[0], address[1], 0, socket.SOCK_STREAM)
    addresses.sort(key=lambda item: item[0] != socket.AF_INET)
    last_error = None
    for family, kind, protocol, _, sockaddr in addresses:
        connection = socket.socket(family, kind, protocol)
        try:
            connection.settimeout(timeout)
            if source_address:
                connection.bind(source_address)
            connection.connect(sockaddr)
            return connection
        except OSError as error:
            last_error = error
            connection.close()
    raise last_error or OSError("No addresses found")


class _IPv4FirstHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = _connect_ipv4_first


class _IPv4FirstHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request):
        return self.do_open(_IPv4FirstHTTPSConnection, request, context=self._context, check_hostname=self._check_hostname)


def archive_open(request, timeout):
    for attempt in range(2):
        try:
            return urllib.request.build_opener(_IPv4FirstHTTPSHandler()).open(request, timeout=timeout)
        except (urllib.error.URLError, TimeoutError):
            if attempt:
                raise
from pathlib import Path

from .io import read_json, sha256, write_json


class BudgetExceeded(RuntimeError):
    pass


class Downloader:
    def __init__(self, root=".", limit=10_000_000_000):
        self.root = Path(root)
        self.path = self.root / "experiments/download_ledger.json"
        self.limit = limit
        self.ledger = read_json(self.path) if self.path.exists() else {
            "limit_bytes": limit, "transferred_bytes": 0, "files": {}, "allowances": []
        }
        self.ledger["limit_bytes"] = limit

    @property
    def remaining(self):
        return self.limit - self.ledger["transferred_bytes"]

    def charge(self, size):
        if size > self.remaining:
            raise BudgetExceeded(f"Transfer requires {size} bytes; {self.remaining} remain")
        self.ledger["transferred_bytes"] += size
        write_json(self.path, self.ledger)

    def allowance(self, name, size):
        if name not in [x["name"] for x in self.ledger["allowances"]]:
            self.charge(size)
            self.ledger["allowances"].append({"name": name, "bytes": size})
            write_json(self.path, self.ledger)

    def fetch(self, url, destination, expected_sha=None, expected_size=None):
        if not url.startswith("https://"):
            raise ValueError("Only HTTPS sources are supported")
        destination = self.root / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        key = str(destination.relative_to(self.root).as_posix())
        old = self.ledger["files"].get(key)
        if destination.exists() and old and sha256(destination) == old["sha256"]:
            if old["url"] != url or (expected_sha and old["sha256"] != expected_sha):
                raise ValueError(f"Artifact identity changed: {key}")
            return destination
        partial = destination.with_suffix(destination.suffix + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "SenseGraph-research/0.1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
            if not response.url.startswith("https://"):
                raise ValueError("Insecure redirect")
            if offset and response.status != 206:
                offset = 0
            elif offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Invalid resume range")
            length = response.headers.get("Content-Length")
            if length and int(length) > self.remaining:
                raise BudgetExceeded(f"{key} exceeds remaining download budget")
            with partial.open("ab" if offset else "wb") as stream:
                while True:
                    # Charge each received chunk before persistence, including failed attempts.
                    chunk = response.read(min(4 * 1024 * 1024, self.remaining + 1))
                    if not chunk:
                        break
                    self.charge(len(chunk))
                    stream.write(chunk)
        if length and partial.stat().st_size != offset + int(length):
            raise IOError(f"Truncated transfer: {key}")
        if expected_size is not None and partial.stat().st_size != expected_size:
            raise IOError(f"Size mismatch: {key}")
        digest = sha256(partial)
        if expected_sha and digest != expected_sha:
            raise IOError(f"SHA256 mismatch: {key}")
        partial.replace(destination)
        self.ledger["files"][key] = {
            "url": url, "bytes": destination.stat().st_size, "sha256": digest
        }
        write_json(self.path, self.ledger)
        return destination

    def json(self, url, name):
        path = self.fetch(url, f"data/metadata/{name}.json")
        return read_json(path)

    def range(self, url, start, end):
        size = end - start + 1
        if size > self.remaining:
            raise BudgetExceeded("ZIP range exceeds remaining budget")
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        with archive_open(request, timeout=10) as response:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                raise IOError("Server did not honor selective ZIP range")
            value = response.read(size)
            self.charge(len(value))
            if len(value) != size:
                raise IOError("Truncated ZIP range")
            return value


class RemoteZipReader(io.RawIOBase):
    """Seekable HTTP range reader: only ZIP metadata and chosen members are fetched."""
    def __init__(self, downloader, url):
        self.downloader, self.url, self.position = downloader, url, 0
        with archive_open(urllib.request.Request(url, method="HEAD"), timeout=10) as response:
            self.size = int(response.headers["Content-Length"])
            self.metadata = {"url": url, "bytes": self.size, "etag": response.headers.get("ETag"),
                             "last_modified": response.headers.get("Last-Modified")}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError("Negative ZIP seek")
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        value = self.downloader.range(self.url, self.position, self.position + size - 1)
        self.position += size
        return value
