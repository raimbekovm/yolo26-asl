"""Dataset download utilities."""

import shutil
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

from loguru import logger
from tqdm import tqdm

from src.utils.constants import DATA_DIR, DATASET_URLS


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class DatasetDownloader:
    """
    Download and extract datasets for ASL recognition.

    Supports:
    - SignAlphaSet from Mendeley Data
    - Ultralytics Hand Keypoints dataset

    Example:
        >>> downloader = DatasetDownloader()
        >>> downloader.download_hand_keypoints()
        >>> downloader.download_all()
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize downloader.

        Args:
            data_dir: Root directory for datasets.
        """
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_hand_keypoints(self, force: bool = False) -> Path:
        """
        Download Ultralytics Hand Keypoints dataset.

        This dataset is auto-downloaded by Ultralytics during training,
        but can be pre-downloaded for faster setup.

        Args:
            force: Re-download even if exists.

        Returns:
            Path to downloaded dataset.
        """
        dataset_dir = self.raw_dir / "hand-keypoints"

        if dataset_dir.exists() and not force:
            logger.info(f"Hand keypoints dataset already exists at {dataset_dir}")
            return dataset_dir

        logger.info("Downloading Hand Keypoints dataset...")

        url = DATASET_URLS["hand_keypoints"]
        zip_path = self.raw_dir / "hand-keypoints.zip"

        try:
            self._download_file(url, zip_path)
            self._extract_zip(zip_path, self.raw_dir)
            zip_path.unlink()  # Remove zip after extraction
            logger.info(f"Hand keypoints dataset ready at {dataset_dir}")
        except Exception as e:
            logger.error(f"Failed to download hand keypoints: {e}")
            raise

        return dataset_dir

    def download_signalphaset(self) -> Path:
        """
        Provide instructions for SignAlphaSet download.

        SignAlphaSet requires manual download from Mendeley Data.

        Returns:
            Path where dataset should be placed.
        """
        dataset_dir = self.raw_dir / "signalphaset"
        dataset_dir.mkdir(exist_ok=True)

        if (dataset_dir / "images").exists():
            logger.info(f"SignAlphaSet already exists at {dataset_dir}")
            return dataset_dir

        instructions = f"""
╔══════════════════════════════════════════════════════════════════╗
║              SignAlphaSet Manual Download Required               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Visit: {DATASET_URLS['signalphaset']}            ║
║                                                                  ║
║  2. Click "Download" and accept terms                            ║
║                                                                  ║
║  3. Extract the downloaded archive to:                           ║
║     {dataset_dir}                                                ║
║                                                                  ║
║  Expected structure:                                             ║
║     signalphaset/                                                ║
║     ├── images/                                                  ║
║     │   ├── A/                                                   ║
║     │   ├── B/                                                   ║
║     │   └── ...                                                  ║
║     └── videos/                                                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
        logger.info(instructions)
        return dataset_dir

    def download_all(self, force: bool = False) -> dict[str, Path]:
        """
        Download all available datasets.

        Args:
            force: Re-download even if exists.

        Returns:
            Dictionary mapping dataset names to paths.
        """
        paths = {}

        # Auto-downloadable
        paths["hand_keypoints"] = self.download_hand_keypoints(force=force)

        # Manual download
        paths["signalphaset"] = self.download_signalphaset()

        return paths

    def _download_file(self, url: str, dest: Path) -> None:
        """Download file with progress bar."""
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=dest.name
        ) as t:
            urlretrieve(url, dest, reporthook=t.update_to)

    def _extract_zip(self, zip_path: Path, dest_dir: Path) -> None:
        """Extract zip file."""
        logger.info(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)

    def verify_dataset(self, name: str) -> bool:
        """
        Verify dataset integrity.

        Args:
            name: Dataset name ('hand_keypoints' or 'signalphaset').

        Returns:
            True if dataset is valid.
        """
        if name == "hand_keypoints":
            dataset_dir = self.raw_dir / "hand-keypoints"
            required = ["images/train", "images/val"]
        elif name == "signalphaset":
            dataset_dir = self.raw_dir / "signalphaset"
            required = ["images"]
        else:
            raise ValueError(f"Unknown dataset: {name}")

        for path in required:
            if not (dataset_dir / path).exists():
                logger.warning(f"Missing: {dataset_dir / path}")
                return False

        logger.info(f"Dataset '{name}' verified successfully")
        return True


def main():
    """CLI entry point for dataset download."""
    import argparse

    parser = argparse.ArgumentParser(description="Download ASL datasets")
    parser.add_argument(
        "--dataset",
        choices=["hand-keypoints", "signalphaset", "all"],
        default="all",
        help="Dataset to download",
    )
    parser.add_argument("--force", action="store_true", help="Re-download if exists")
    parser.add_argument("--data-dir", type=Path, help="Data directory")

    args = parser.parse_args()

    downloader = DatasetDownloader(data_dir=args.data_dir)

    if args.dataset == "all":
        downloader.download_all(force=args.force)
    elif args.dataset == "hand-keypoints":
        downloader.download_hand_keypoints(force=args.force)
    elif args.dataset == "signalphaset":
        downloader.download_signalphaset()


if __name__ == "__main__":
    main()
