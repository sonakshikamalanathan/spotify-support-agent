"""Get twcs.csv into data/raw.

Order of attempts:
  1. already extracted in data/raw
  2. a zip downloaded manually from the Kaggle website (data/raw or ~/Downloads)
  3. the Kaggle API (token in ~/.kaggle/kaggle.json, or KAGGLE_* variables in .env)
"""
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from config import DATA_RAW, KAGGLE_DATASET, ROOT


def find_raw_csv():
    matches = sorted(DATA_RAW.rglob("twcs.csv"))
    return matches[0] if matches else None


def find_manual_zip():
    for folder in (DATA_RAW, Path.home() / "Downloads"):
        for zip_path in sorted(folder.glob("*.zip")):
            with zipfile.ZipFile(zip_path) as zf:
                if any(name.endswith("twcs.csv") for name in zf.namelist()):
                    return zip_path
    return None


def main():
    existing = find_raw_csv()
    if existing:
        print(f"Found {existing}, skipping download.")
        return
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    manual_zip = find_manual_zip()
    if manual_zip:
        print(f"Extracting {manual_zip} ...")
        with zipfile.ZipFile(manual_zip) as zf:
            zf.extractall(DATA_RAW)
        print(f"Done: {find_raw_csv()}")
        return

    load_dotenv(ROOT / ".env")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {KAGGLE_DATASET} ...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_RAW), unzip=True, quiet=False)
    print(f"Done: {find_raw_csv()}")


if __name__ == "__main__":
    main()
