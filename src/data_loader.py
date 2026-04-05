from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
import requests

# NHANES has changed URL patterns over time. The legacy Continuous NHANES path
# can now return HTML "Page Not Found" while still responding with HTTP 200.
# The "Public/{BeginYear}/DataFiles" endpoint currently serves the real XPTs.
NHANES_PUBLIC_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
NHANES_LEGACY_BASE = "https://wwwn.cdc.gov/Nchs/Nhanes"

XPT_HEADER_PREFIX = b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!"


@dataclass(frozen=True)
class Cycle:
    years: str   # e.g. "2011-2012"
    suffix: str  # e.g. "G"


CYCLES_2011_2018: tuple[Cycle, ...] = (
    Cycle("2011-2012", "G"),
    Cycle("2013-2014", "H"),
    Cycle("2015-2016", "I"),
    Cycle("2017-2018", "J"),
)


class NHANESDownloadError(RuntimeError):
    pass


def _is_valid_xpt_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < len(XPT_HEADER_PREFIX):
        return False
    with open(path, "rb") as f:
        head = f.read(max(256, len(XPT_HEADER_PREFIX)))
    return XPT_HEADER_PREFIX in head


def _begin_year(cycle_years: str) -> str:
    # "2011-2012" -> "2011"
    return cycle_years.split("-")[0]


def _download(url: str, dst: Path, timeout: int = 60) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        # Validate cached file: CDC sometimes serves HTML error pages at XPT URLs.
        if _is_valid_xpt_file(dst):
            return dst
        dst.unlink(missing_ok=True)

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code == 404:
                raise NHANESDownloadError(f"NHANES file not found (404): {url}")
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except requests.RequestException as e:
        raise NHANESDownloadError(f"Failed downloading {url}: {e}") from e

    # Validate: if we accidentally downloaded HTML, fail fast and don't poison cache.
    if not _is_valid_xpt_file(dst):
        dst.unlink(missing_ok=True)
        raise NHANESDownloadError(
            "Downloaded file is not a valid XPT (likely HTML error page). "
            f"URL: {url}"
        )

    return dst


def fetch_xpt(cycle: Cycle, file_code: str, cache_dir: Path) -> Path:
    """Download an NHANES XPT for a cycle into cache.

    file_code examples: DEMO, DIQ, BMX, BPX, SMQ, ALQ, PAQ, GHB
    """
    # Cache filename uses .XPT but remote endpoints may use .xpt.
    filename = f"{file_code}_{cycle.suffix}.XPT"
    begin = _begin_year(cycle.years)
    remote_name = f"{file_code}_{cycle.suffix}.xpt"

    # Try current public endpoint first, then legacy as fallback.
    candidate_urls = [
        f"{NHANES_PUBLIC_BASE}/{begin}/DataFiles/{remote_name}",
        f"{NHANES_LEGACY_BASE}/{cycle.years}/{filename}",
    ]

    dst = cache_dir / cycle.years / filename
    last_err: Exception | None = None
    for url in candidate_urls:
        try:
            return _download(url, dst)
        except Exception as e:
            last_err = e
            continue

    raise NHANESDownloadError(f"All download attempts failed for {filename}: {last_err}")


def read_xpt(path: Path) -> pd.DataFrame:
    df = pd.read_sas(path, format="xport")
    df.columns = [str(c).upper() for c in df.columns]
    return df


def load_cycle(cycle: Cycle, cache_dir: Path, components: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for comp in components:
        xpt_path = fetch_xpt(cycle, comp, cache_dir=cache_dir)
        frames.append(read_xpt(xpt_path))

    # Merge all frames on SEQN
    base = next((f for f in frames if "SEQN" in f.columns and "RIDAGEYR" in f.columns), None)
    if base is None:
        base = next(f for f in frames if "SEQN" in f.columns)

    out = base.copy()
    for f in frames:
        if f is base:
            continue
        if "SEQN" not in f.columns:
            continue
        out = out.merge(f, on="SEQN", how="left")

    out["NHANES_CYCLE"] = cycle.years
    out["NHANES_SUFFIX"] = cycle.suffix
    return out


def default_components(with_labs: bool) -> list[str]:
    components = [
        "DEMO",  # demographics + weights
        "DIQ",   # diabetes questionnaire (target)
        "BMX",   # body measures (BMI, waist)
        "BPX",   # blood pressure
        "SMQ",   # smoking
        "ALQ",   # alcohol
        "PAQ",   # physical activity
    ]

    # Optional lab-enhanced model: HbA1c (generally available and does not require fasting subsample weights).
    if with_labs:
        components.append("GHB")

    return components


def load_nhanes_2011_2018(cache_dir: Path, *, with_labs: bool = False) -> pd.DataFrame:
    components = default_components(with_labs=with_labs)

    all_cycles: list[pd.DataFrame] = []
    for c in CYCLES_2011_2018:
        all_cycles.append(load_cycle(c, cache_dir=cache_dir, components=components))

    df = pd.concat(all_cycles, ignore_index=True, sort=False)

    # Normalize column casing + types
    for col in df.columns:
        if col in {"NHANES_CYCLE", "NHANES_SUFFIX"}:
            continue
        if df[col].dtype == "object":
            # pandas 3.x removed errors='ignore'; we only coerce if it produces
            # at least some numeric values.
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted

    return df
