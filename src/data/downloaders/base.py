import hashlib
import shutil
import ssl
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


class DatasetDownloader(ABC):
    name: str
    url: str
    archive_name: str
    dataset_dir_name: str | None = None
    sha256: str | None = None

    def __init__(
        self,
        raw_root: str | Path = "data/raw",
        force: bool = False,
    ):
        self.raw_root = Path(raw_root)
        self.force = force
        self._archive_path: Path | None = None

    @property
    def dataset_dir(self) -> Path:
        return self.raw_root / (self.dataset_dir_name or self.name)

    @property
    def archive_path(self) -> Path:
        if self._archive_path is None:
            raise RuntimeError("Archive path is only available during downloader.run().")
        return self._archive_path

    def run(self) -> Path:
        if self.is_available() and not self.force:
            self.validate()
            print(f"Dataset already available: {self.dataset_dir}")
            return self.dataset_dir

        with tempfile.TemporaryDirectory(prefix=f"{self.name}-download-") as tmp:
            self._archive_path = Path(tmp) / self.archive_name
            try:
                self.download()
                self.prepare()
                self.validate()
            finally:
                self._archive_path = None

        return self.dataset_dir

    def download(self) -> None:
        print(f"Downloading {self.name} from {self.url}")
        print("TLS certificate verification is disabled; archive SHA256 will be checked.")
        partial_path = self.archive_path.with_suffix(self.archive_path.suffix + ".part")
        try:
            self._download_url(self.url, partial_path)
        except (OSError, URLError) as exc:
            partial_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download {self.name} from {self.url}. "
                "Check your network connection."
            ) from exc

        partial_path.replace(self.archive_path)
        self._verify_archive()
        print(f"Saved archive: {self.archive_path}")

    def _download_url(self, url: str, output_path: Path) -> None:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        request = Request(
            url,
            headers={"User-Agent": "colonoscopy-segmentation-reproduction/0.1"},
        )

        with urlopen(request, context=ssl_context) as response:
            with open(output_path, "wb") as output_file:
                shutil.copyfileobj(response, output_file)

    def _verify_archive(self) -> None:
        if self.sha256 is None:
            return

        digest = hashlib.sha256()
        with open(self.archive_path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        actual = digest.hexdigest()
        if actual != self.sha256:
            raise ValueError(
                f"Checksum mismatch for {self.archive_path}. "
                f"Expected {self.sha256}, got {actual}."
            )

    @abstractmethod
    def prepare(self) -> None:
        ...

    @abstractmethod
    def validate(self) -> None:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...
