"""Step 1: collect raw data.

1. Weekly dengue cases per district (Sri Lanka Epidemiology Unit weekly
   epidemiological reports, compiled in the CRAN package `denguedatahub`).
2. Daily weather per district from the Open-Meteo historical archive API
   (ERA5 reanalysis), aggregated later to weekly.

Run:  py src/collect_data.py
"""
import io
import re
import tarfile
import time
from pathlib import Path

import pandas as pd
import pyreadr
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

CRAN_PKG = "https://cran.r-project.org/src/contrib/"

# Approximate district centroids (lat, lon) used to query weather.
DISTRICTS = {
    "Colombo": (6.90, 79.95), "Gampaha": (7.09, 80.01), "Kalutara": (6.58, 80.11),
    "Kandy": (7.29, 80.63), "Matale": (7.67, 80.62), "NuwaraEliya": (6.95, 80.78),
    "Galle": (6.12, 80.22), "Matara": (6.05, 80.54), "Hambanthota": (6.20, 81.12),
    "Jaffna": (9.66, 80.02), "Kilinochchi": (9.39, 80.40), "Mannar": (8.98, 79.91),
    "Vavuniya": (8.75, 80.50), "Mullaitivu": (9.27, 80.81), "Batticaloa": (7.73, 81.69),
    "Ampara": (7.30, 81.67), "Trincomalee": (8.59, 81.21), "Kurunegala": (7.49, 80.36),
    "Puttalam": (8.04, 79.83), "Anuradhapura": (8.31, 80.40), "Polonnaruwa": (7.94, 81.00),
    "Badulla": (6.99, 81.06), "Monaragala": (6.87, 81.35), "Ratnapura": (6.68, 80.40),
    "Kegalle": (7.25, 80.35), "Kalmune": (7.41, 81.83),
}


def download_dengue() -> pd.DataFrame:
    """Download denguedatahub source tarball and read srilanka_weekly_data."""
    index = requests.get(CRAN_PKG, timeout=60).text
    versions = re.findall(r"denguedatahub_([\d.]+)\.tar\.gz", index)
    url = f"{CRAN_PKG}denguedatahub_{versions[-1]}.tar.gz"
    print("Downloading", url)
    blob = requests.get(url, timeout=120).content
    out = RAW / "srilanka_weekly_data.rda"
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        member = next(m for m in tar.getmembers()
                      if m.name.endswith("data/srilanka_weekly_data.rda"))
        out.write_bytes(tar.extractfile(member).read())
    df = next(iter(pyreadr.read_r(str(out)).values()))
    df.to_csv(RAW / "dengue_weekly_district.csv", index=False)
    print(f"Dengue: {len(df):,} rows, districts={df['district'].nunique()}")
    return df


def download_weather(start: str, end: str) -> pd.DataFrame:
    """Daily rainfall, temperature and humidity per district from Open-Meteo."""
    frames = []
    for name, (lat, lon) in DISTRICTS.items():
        params = dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                      daily="precipitation_sum,temperature_2m_mean,"
                            "relative_humidity_2m_mean",
                      timezone="Asia/Colombo")
        for attempt in range(5):
            r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                             params=params, timeout=120)
            if r.status_code == 200:
                break
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()
        d = pd.DataFrame(r.json()["daily"])
        d["district"] = name
        frames.append(d)
        print(f"  weather: {name} ({len(d)} days)")
        time.sleep(1)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RAW / "weather_daily_district.csv", index=False)
    return df


if __name__ == "__main__":
    dengue = download_dengue()
    print(dengue.head())
    start = pd.to_datetime(dengue["start.date"], format="mixed").min() - pd.Timedelta(days=60)
    end = pd.to_datetime(dengue["end.date"], format="mixed").max()
    download_weather(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
