#!/usr/bin/env python3
"""
Utility helpers that download spatial assets required by the pipeline when Git LFS
objects are unavailable (e.g., in CI where the LFS quota is exceeded).

Currently ensures the HDX COD-AB ADM2 shapefile is present under data/.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import json
from urllib import request, error

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ADM2_DIR = DATA_DIR / "mex_admbnda_govmex_20210618_SHP"
HDX_PACKAGE_ID = "cod-ab-mex"
HDX_API_URL = f"https://data.humdata.org/api/3/action/package_show?id={HDX_PACKAGE_ID}"
HDX_DIRECT_URL = (
    "https://data.humdata.org/dataset/9721eaf0-5663-4137-b3a2-c21dc8fac15a/resource/"
    "f151b1c1-1353-4f57-bdb2-b1b1c18a1fd1/download/mex_admbnda_govmex_20210618_shp.zip"
)
MIN_SHAPEFILE_BYTES = 5_000_000  # pointer files are only a few hundred bytes


def _needs_download(shp_path: Path, force: bool) -> bool:
    if force:
        return True
    return not shp_path.exists() or shp_path.stat().st_size < MIN_SHAPEFILE_BYTES


def _request_json(url: str) -> dict:
    headers = {"User-Agent": "BrigadasPipeline/1.0 (+https://github.com/robertclayh/Brigadas-SaludMaterna)"}
    req = request.Request(url, headers=headers)
    with request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected response from {url}")
    return payload


def _find_hdx_resource(tokens: Iterable[str]) -> str:
    tokens = [t.lower() for t in tokens]
    try:
        payload = _request_json(HDX_API_URL)
        result = payload.get("result") or {}
        resources = result.get("resources") or []
        for resource in resources:
            blob = " ".join(
                str(resource.get(k) or "")
                for k in ("name", "description", "format", "url", "download_url")
            ).lower()
            if all(token in blob for token in tokens):
                url = resource.get("download_url") or resource.get("url")
                if url:
                    return url
        raise RuntimeError(f"Unable to locate ADM2 shapefile download in HDX dataset '{HDX_PACKAGE_ID}'.")
    except Exception as exc:
        if HDX_DIRECT_URL:
            print(f"[adm2] Falling back to direct HDX download URL due to error: {exc}")
            return HDX_DIRECT_URL
        raise


def _download_file(url: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    tmp_path = Path(tmp.name)
    try:
        headers = {"User-Agent": "BrigadasPipeline/1.0 (+https://github.com/robertclayh/Brigadas-SaludMaterna)"}
        req = request.Request(url, headers=headers)
        with request.urlopen(req, timeout=600) as resp:
            shutil.copyfileobj(resp, tmp)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        tmp.close()
    return tmp_path


def _extract_zip(zip_path: Path) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="adm2_zip_"))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmpdir)
    return tmpdir


def _locate_shapefile(root: Path) -> Optional[Path]:
    shapefiles = sorted(root.rglob("*.shp"))
    if not shapefiles:
        return None
    for shp in shapefiles:
        if "adm2" in shp.stem.lower():
            return shp
    return shapefiles[0]


def _replace_dir(dest: Path):
    if dest.exists():
        for child in dest.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                shutil.rmtree(child)
    else:
        dest.mkdir(parents=True, exist_ok=True)


def _copy_shapefile_family(source_shp: Path, dest_dir: Path):
    stem = source_shp.stem
    source_dir = source_shp.parent
    relevant = list(source_dir.glob(f"{stem}.*"))
    if not relevant:
        raise RuntimeError(f"No shapefile components found for stem '{stem}'.")
    for src in relevant:
        target = dest_dir / src.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


def ensure_adm2_shapefile(force: bool = False):
    target_shp = ADM2_DIR / "mex_admbnda_adm2_govmex_20210618.shp"
    if not _needs_download(target_shp, force):
        print(f"[adm2] Existing shapefile looks good at {target_shp}.")
        return
    print("[adm2] Downloading ADM2 shapefile from HDX…")
    url = _find_hdx_resource(tokens=("adm2", "shp"))
    tmp_file = _download_file(url)
    extract_dir = None
    try:
        extract_dir = _extract_zip(tmp_file)
        shp_path = _locate_shapefile(extract_dir)
        if not shp_path:
            raise RuntimeError("Zip archive did not contain a .shp file.")
        print(f"[adm2] Using shapefile '{shp_path.name}' extracted from archive.")
        ADM2_DIR.mkdir(parents=True, exist_ok=True)
        _replace_dir(ADM2_DIR)
        _copy_shapefile_family(shp_path, ADM2_DIR)
        print(f"[adm2] Shapefile saved under {ADM2_DIR}.")
    finally:
        tmp_file.unlink(missing_ok=True)
        if extract_dir and extract_dir.exists():
            shutil.rmtree(extract_dir)


def main(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(description="Fetch required spatial assets for the pipeline.")
    parser.add_argument(
        "--force-adm2",
        action="store_true",
        help="Re-download the ADM2 shapefile even if the current file looks valid.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        ensure_adm2_shapefile(force=args.force_adm2)
    except Exception as exc:
        print(f"ERROR: failed to fetch ADM2 shapefile - {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
