"""
dataset_loader.py — Real Dataset Integration for Underwater Image Restoration
==============================================================================

DATASET 1 — EUVP (Enhancing Underwater Visual Perception)
  Source  : http://irvlab.cs.umn.edu/resources/euvp-dataset
  Authors : Islam et al. (2020), IEEE RA-L
  License : Creative Commons (Non-Commercial)
  Size    : ~12,000 paired (distorted, enhanced) underwater images
  URL     : https://drive.google.com/drive/folders/1LWiaro7XivHBg4sMOAIHuGANM6pOg0gy
  Why     : Gold-standard paired underwater image dataset — exact training
            target for CycleGAN underwater-to-clear domain translation.

DATASET 2 — UIEB (Underwater Image Enhancement Benchmark)
  Source  : https://li-chongyi.github.io/proj_benchmark.html
  Authors : Li et al. (2019), IEEE TIP
  License : Research use
  Size    : 950 raw + 890 reference underwater images
  Direct  : https://github.com/Li-Chongyi/UIEB-dataset
  Why     : Industry benchmark for underwater enhancement evaluation.
            PSNR/SSIM computed against UIEB reference images.

DATASET 3 — OceanDark Dataset (Low-light underwater)
  Source  : https://github.com/tunai/OceanDark
  License : MIT
  Size    : 189 paired dark/enhanced underwater images
  Why     : Covers low-light conditions not in EUVP/UIEB.

DATASET 4 — RUIE (Real-world Underwater Image Enhancement)
  Source  : https://github.com/dlut-dimt/RealworldUnderwater-task1-UIQS
  License : Research use
  Size    : 4,000+ real underwater images with quality scores

NOTE: These are image datasets. This loader handles download + path organisation.
      The C++ training code reads images from the paths set here.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

DATASETS = {
    "uieb": {
        "description": "Underwater Image Enhancement Benchmark — 950 raw + 890 reference images",
        "url":         "https://github.com/Li-Chongyi/UIEB-dataset/archive/refs/heads/master.zip",
        "target_dir":  DATA_DIR / "uieb",
        "paper":       "Li et al. (2019), IEEE TIP",
        "license":     "Research use",
    },
    "oceandark": {
        "description": "OceanDark — 189 paired dark/enhanced underwater images",
        "url":         "https://github.com/tunai/OceanDark/archive/refs/heads/master.zip",
        "target_dir":  DATA_DIR / "oceandark",
        "paper":       "Porto Marques et al. (2019)",
        "license":     "MIT",
    },
}

EUVP_MANUAL_URL = "https://drive.google.com/drive/folders/1LWiaro7XivHBg4sMOAIHuGANM6pOg0gy"


def print_dataset_info() -> None:
    """Print information about all datasets and how to obtain them."""
    print("=" * 65)
    print("  Underwater Image Restoration — Dataset Information")
    print("=" * 65)

    print("""
DATASET 1 — EUVP (PRIMARY — 12,000 paired images)
  URL     : http://irvlab.cs.umn.edu/resources/euvp-dataset
  Download: Requires manual registration (Google Drive link after):
            {EUVP_MANUAL_URL}
  Place files in: data/euvp/
    ├── trainA/   (distorted underwater images)
    ├── trainB/   (enhanced/clear counterparts)
    ├── testA/
    └── testB/

DATASET 2 — UIEB (Benchmark — 950 raw + 890 reference)
  Auto-downloadable via this script.
  Run: python dataset_loader.py --download uieb

DATASET 3 — OceanDark (189 paired dark images)
  Auto-downloadable via this script.
  Run: python dataset_loader.py --download oceandark

After placing/downloading datasets, update configs/dataset.yaml:
  train_data_root: data/euvp
  test_data_root:  data/uieb
""")


def download_dataset(name: str, force: bool = False) -> Path:
    """Download a named dataset (uieb or oceandark)."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS.keys())}")

    info       = DATASETS[name]
    target_dir = info["target_dir"]

    if target_dir.exists() and not force:
        print(f"[{name}] Already downloaded at {target_dir}. Use --force to re-download.")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / f"{name}.zip"

    print(f"[{name}] Downloading from {info['url']} …")
    try:
        urllib.request.urlretrieve(info["url"], zip_path)
        print(f"[{name}] Extracting …")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(target_dir)
        zip_path.unlink()
        print(f"[{name}] Extracted to {target_dir}")
    except Exception as e:
        print(f"[{name}] Download failed: {e}")
        print(f"  Manual: {info['url']}")
    return target_dir


def get_image_paths(dataset: str = "euvp") -> dict:
    """
    Return train/test image paths for the specified dataset.
    Used by the C++ CMake build system via dataset.yaml.
    """
    roots = {
        "euvp":      DATA_DIR / "euvp",
        "uieb":      DATA_DIR / "uieb",
        "oceandark": DATA_DIR / "oceandark",
    }
    root = roots.get(dataset, DATA_DIR / dataset)

    paths = {
        "train_distorted": str(root / "trainA"),
        "train_enhanced":  str(root / "trainB"),
        "test_distorted":  str(root / "testA"),
        "test_enhanced":   str(root / "testB"),
    }

    # Check existence
    found = sum(1 for p in paths.values() if Path(p).exists())
    print(f"[{dataset}] {found}/{len(paths)} directories found.")
    if found < 2:
        print(f"  Run: python dataset_loader.py --download {dataset}")

    return paths


def generate_dataset_yaml(output: str = "configs/dataset.yaml") -> None:
    """Generate a dataset config YAML consumed by the C++ training code."""
    import yaml
    euvp_paths = get_image_paths("euvp")
    uieb_paths = get_image_paths("uieb")

    config = {
        "train": {
            "distorted_dir": euvp_paths["train_distorted"],
            "enhanced_dir":  euvp_paths["train_enhanced"],
            "image_size":    256,
            "batch_size":    4,
            "shuffle":       True,
        },
        "test": {
            "distorted_dir": uieb_paths["test_distorted"],
            "enhanced_dir":  uieb_paths["test_enhanced"],
            "image_size":    256,
        },
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[Config] Dataset YAML written to {output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Underwater Image Dataset Loader")
    parser.add_argument("--info",     action="store_true", help="Show dataset info")
    parser.add_argument("--download", choices=list(DATASETS.keys()), help="Download a dataset")
    parser.add_argument("--force",    action="store_true", help="Force re-download")
    parser.add_argument("--yaml",     action="store_true", help="Generate dataset.yaml")
    args = parser.parse_args()

    if args.info or not any([args.download, args.yaml]):
        print_dataset_info()
    if args.download:
        download_dataset(args.download, force=args.force)
    if args.yaml:
        try:
            generate_dataset_yaml()
        except ImportError:
            print("Run: pip install pyyaml   to generate YAML config")
