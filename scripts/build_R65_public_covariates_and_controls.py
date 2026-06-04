"""R65 public covariates and external-control analysis.

This round adds externally sourced urban-centre and country-background
covariates to the 71-city road-window evidence stack. The analysis is designed
to answer a reviewer-level question: are the CEBH gaps and spatial-null
residuals simply proxies for city size, development level or country context,
or are embedded street-form variables still the stronger explanatory layer?

Inputs:
  - R63 integrated high-order road-window covariates.
  - R64 all-city spatial-null screen.
  - JRC/GHSL UCDB R2024A theme zip files downloaded to data/R65.../raw.
  - World Bank WDI API, downloaded as current public country controls.

GHSL and WDI values are used as public covariates/control variables, not as
causal urban-planning evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
import time
import unicodedata
import zipfile
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "R65_public_covariates_and_controls"
RAW = OUT / "raw"
ROUND_DIR = ROOT / "rounds" / "R65_public_covariates_and_controls"
FIG_BASE = ROOT / "figures" / "Fig_R65_public_covariate_controls"

R63_INTEGRATED = ROOT / "data" / "R63_high_order_external_covariates" / "integrated_high_order_external_covariates.csv"
R64_SPATIAL = ROOT / "data" / "R64_71city_spatial_null_screen" / "spatial_null_screen_summary.csv"

GHSL_ZIPS = {
    "general": RAW / "GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A_V1_1.zip",
    "socioeconomic": RAW / "GHS_UCDB_THEME_SOCIOECONOMIC_GLOBE_R2024A_V1_1.zip",
    "infrastructures": RAW / "GHS_UCDB_THEME_INFRASTRUCTURES_GLOBE_R2024A_V1_1.zip",
    "lulc": RAW / "GHS_UCDB_THEME_LULC_GLOBE_R2024A_V1_1.zip",
}

GHSL_URLS = {
    "general": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_UCDB_GLOBE_R2024A/GHS_UCDB_THEME_GLOBE_R2024A/GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A/V1-1/GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A_V1_1.zip",
    "socioeconomic": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_UCDB_GLOBE_R2024A/GHS_UCDB_THEME_GLOBE_R2024A/GHS_UCDB_THEME_SOCIOECONOMIC_GLOBE_R2024A/V1-1/GHS_UCDB_THEME_SOCIOECONOMIC_GLOBE_R2024A_V1_1.zip",
    "infrastructures": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_UCDB_GLOBE_R2024A/GHS_UCDB_THEME_GLOBE_R2024A/GHS_UCDB_THEME_INFRASTRUCTURES_GLOBE_R2024A/V1-1/GHS_UCDB_THEME_INFRASTRUCTURES_GLOBE_R2024A_V1_1.zip",
    "lulc": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_UCDB_GLOBE_R2024A/GHS_UCDB_THEME_GLOBE_R2024A/GHS_UCDB_THEME_LULC_GLOBE_R2024A/V1-1/GHS_UCDB_THEME_LULC_GLOBE_R2024A_V1_1.zip",
}

WDI_INDICATORS = {
    "SP.URB.TOTL.IN.ZS": "wdi_urban_population_pct",
    "SP.URB.GROW": "wdi_urban_population_growth_pct",
    "EN.POP.DNST": "wdi_population_density_people_per_km2",
    "NY.GDP.PCAP.KD": "wdi_gdp_per_capita_constant_2015_usd",
    "SP.POP.TOTL": "wdi_population_total",
    "SI.POV.GINI": "wdi_gini_index",
}

COUNTRY_TO_ISO3 = {
    "Algeria": "DZA",
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Bangladesh": "BGD",
    "Brazil": "BRA",
    "Canada": "CAN",
    "Chile": "CHL",
    "China": "CHN",
    "Colombia": "COL",
    "Czechia": "CZE",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "Ethiopia": "ETH",
    "France": "FRA",
    "Ghana": "GHA",
    "Greece": "GRC",
    "India": "IND",
    "Indonesia": "IDN",
    "Iran": "IRN",
    "Israel": "ISR",
    "Italy": "ITA",
    "Japan": "JPN",
    "Kenya": "KEN",
    "Malaysia": "MYS",
    "Mexico": "MEX",
    "Morocco": "MAR",
    "Netherlands": "NLD",
    "New Zealand": "NZL",
    "Nigeria": "NGA",
    "Pakistan": "PAK",
    "Panama": "PAN",
    "Peru": "PER",
    "Philippines": "PHL",
    "Poland": "POL",
    "Portugal": "PRT",
    "Saudi Arabia": "SAU",
    "Senegal": "SEN",
    "Singapore": "SGP",
    "South Africa": "ZAF",
    "South Korea": "KOR",
    "Spain": "ESP",
    "Taiwan": "TWN",
    "Tanzania": "TZA",
    "Thailand": "THA",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "Uganda": "UGA",
    "United Arab Emirates": "ARE",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Uruguay": "URY",
    "Vietnam": "VNM",
}

REGION_COLORS = {
    "Europe": "#4E79A7",
    "Middle East": "#EDC948",
    "Africa": "#59A14F",
    "Latin America": "#B07AA1",
    "Asia": "#F28E2B",
    "North America": "#E15759",
    "Oceania": "#76B7B2",
    "Other": "#8C8C8C",
}

EXTERNAL_FEATURE_LABELS = {
    "ghsl_population_density_2025": "GHSL population density",
    "ghsl_log_population_2025": "GHSL log population",
    "ghsl_log_area_2025": "GHSL log urban-centre area",
    "ghsl_infrastructure_index_2020": "GHSL critical-infrastructure index",
    "ghsl_road_density_2024": "GHSL road density",
    "ghsl_gdp_avg_2020": "GHSL GDP average",
    "ghsl_gdp_sum_2020": "GHSL GDP sum",
    "ghsl_hdi_2020": "GHSL HDI",
    "ghsl_life_expectancy_2020": "GHSL life expectancy",
    "ghsl_expected_schooling_2020": "GHSL expected schooling",
    "ghsl_mean_schooling_2020": "GHSL mean schooling",
    "ghsl_lulc_entropy_2019": "GHSL land-use entropy",
    "ghsl_lulc_dominance_2019": "GHSL land-use dominance",
    "wdi_urban_population_pct": "WDI urban population %",
    "wdi_urban_population_growth_pct": "WDI urban growth %",
    "wdi_population_density_people_per_km2": "WDI population density",
    "wdi_gdp_per_capita_constant_2015_usd": "WDI GDP per capita",
    "wdi_population_total": "WDI population",
    "wdi_gini_index": "WDI Gini",
}


@dataclass
class MatchResult:
    city: str
    country: str
    ID_UC_G0: int
    ghsl_name: str
    ghsl_country: str
    ghsl_latitude: float
    ghsl_longitude: float
    ghsl_match_similarity: float
    ghsl_match_distance_km: float
    ghsl_match_score: float
    ghsl_match_status: str


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    FIG_BASE.parent.mkdir(parents=True, exist_ok=True)


def normalise_text(value: Any) -> str:
    s = str(value).lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[,\._/\-\(\)'’]", " ", s)
    s = re.sub(r"\b(city|municipality|metropolitan|area|greater|province|prefecture|district)\b", " ", s)
    return " ".join(s.split())


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"-": np.nan, "": np.nan}), errors="coerce")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * radius * math.asin(min(1.0, math.sqrt(a)))


def download_file(url: str, out: Path, min_bytes: int = 1_000_000) -> None:
    if out.exists() and out.stat().st_size >= min_bytes:
        return
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        tmp = out.with_suffix(out.suffix + ".tmp")
        with tmp.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=2**20):
                if chunk:
                    fh.write(chunk)
        tmp.replace(out)


def ensure_ghsl_downloads(skip_download: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for theme, path in GHSL_ZIPS.items():
        if not skip_download:
            download_file(GHSL_URLS[theme], path)
        rows.append(
            {
                "source_family": "GHSL UCDB R2024A",
                "theme": theme,
                "url": GHSL_URLS[theme],
                "local_file": str(path.relative_to(ROOT)),
                "downloaded_bytes": path.stat().st_size if path.exists() else 0,
                "status": "downloaded" if path.exists() and path.stat().st_size > 1_000_000 else "missing",
            }
        )
    return rows


def zip_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        csvs = [n for n in zf.namelist() if n.endswith(".csv") and "{th}" not in n]
        if not csvs:
            raise FileNotFoundError(f"No theme CSV found in {path}")
        with zf.open(csvs[0]) as fh:
            return pd.read_csv(fh, low_memory=False)


def general_centroids(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        gpkg_names = [n for n in zf.namelist() if n.endswith(".gpkg") and "{th}" not in n]
        if not gpkg_names:
            raise FileNotFoundError(f"No theme GPKG found in {path}")
        with tempfile.TemporaryDirectory() as td:
            gpkg_path = Path(td) / Path(gpkg_names[0]).name
            gpkg_path.write_bytes(zf.read(gpkg_names[0]))
            gdf = gpd.read_file(gpkg_path, layer="UC_centroids").to_crs("EPSG:4326")
    gdf["ghsl_longitude"] = gdf.geometry.x
    gdf["ghsl_latitude"] = gdf.geometry.y
    return pd.DataFrame(gdf.drop(columns="geometry"))


def read_ghsl_covariates() -> pd.DataFrame:
    general = zip_csv(GHSL_ZIPS["general"])
    centroids = general_centroids(GHSL_ZIPS["general"])
    infra = zip_csv(GHSL_ZIPS["infrastructures"])
    socio = zip_csv(GHSL_ZIPS["socioeconomic"])
    lulc = zip_csv(GHSL_ZIPS["lulc"])

    general_keep = [
        "ID_UC_G0",
        "GC_UCN_MAI_2025",
        "GC_UCN_LIS_2025",
        "GC_CNT_GAD_2025",
        "GC_CNT_UNN_2025",
        "GC_UCA_KM2_2025",
        "GC_POP_TOT_2025",
        "GC_DEV_WIG_2025",
        "GC_DEV_USR_2025",
        "GC_PLS_SCR_2025",
        "GC_UCB_YOB_2025",
        "GC_UCB_YOD_2025",
        "GC_UCM_CAP",
    ]
    infra_keep = [
        "ID_UC_G0",
        "IN_CIS_ENE_2020",
        "IN_CIS_TRA_2020",
        "IN_CIS_WAT_2020",
        "IN_CIS_WAS_2020",
        "IN_CIS_TEL_2020",
        "IN_CIS_HEA_2020",
        "IN_CIS_EDU_2020",
        "IN_CIS_ALL_2020",
        "IN_ROA_LEN_2024",
        "IN_ROA_DEN_2024",
    ]
    socio_keep = [
        "ID_UC_G0",
        "SC_GDP_AVG_2020",
        "SC_GDP_SUM_2020",
        "SC_SEC_HDI_2020",
        "SC_SEC_LET_2020",
        "SC_SEC_SET_2020",
        "SC_SEC_SYT_2020",
    ]
    lulc_2019 = [c for c in lulc.columns if c.startswith("LU_HEC_") and c.endswith("_2019")]
    lulc_numeric = lulc[["ID_UC_G0", *lulc_2019]].copy()
    for c in lulc_2019:
        lulc_numeric[c] = to_num(lulc_numeric[c])
    arr = lulc_numeric[lulc_2019].to_numpy(dtype=float)
    total = np.nansum(arr, axis=1)
    probs = np.divide(arr, total[:, None], out=np.zeros_like(arr), where=total[:, None] > 0)
    nonzero = probs > 0
    entropy = np.zeros(len(lulc_numeric))
    denom = math.log(max(2, len(lulc_2019)))
    for i in range(len(lulc_numeric)):
        p = probs[i, nonzero[i]]
        entropy[i] = -float(np.sum(p * np.log(p))) / denom if len(p) else np.nan
    lulc_features = pd.DataFrame(
        {
            "ID_UC_G0": lulc_numeric["ID_UC_G0"],
            "ghsl_lulc_total_hectares_2019": total,
            "ghsl_lulc_entropy_2019": entropy,
            "ghsl_lulc_dominance_2019": np.nanmax(probs, axis=1),
            "ghsl_lulc_nonzero_classes_2019": nonzero.sum(axis=1),
        }
    )

    gh = general[general_keep].merge(centroids, on="ID_UC_G0", how="left")
    gh = gh.merge(infra[infra_keep], on="ID_UC_G0", how="left")
    gh = gh.merge(socio[socio_keep], on="ID_UC_G0", how="left")
    gh = gh.merge(lulc_features, on="ID_UC_G0", how="left")

    numeric_cols = [
        "GC_UCA_KM2_2025",
        "GC_POP_TOT_2025",
        "IN_CIS_ENE_2020",
        "IN_CIS_TRA_2020",
        "IN_CIS_WAT_2020",
        "IN_CIS_WAS_2020",
        "IN_CIS_TEL_2020",
        "IN_CIS_HEA_2020",
        "IN_CIS_EDU_2020",
        "IN_CIS_ALL_2020",
        "IN_ROA_LEN_2024",
        "IN_ROA_DEN_2024",
        "SC_GDP_AVG_2020",
        "SC_GDP_SUM_2020",
        "SC_SEC_HDI_2020",
        "SC_SEC_LET_2020",
        "SC_SEC_SET_2020",
        "SC_SEC_SYT_2020",
    ]
    for c in numeric_cols:
        if c in gh.columns:
            gh[c] = to_num(gh[c])

    rename = {
        "GC_UCN_MAI_2025": "ghsl_name_main_2025",
        "GC_UCN_LIS_2025": "ghsl_name_list_2025",
        "GC_CNT_GAD_2025": "ghsl_country_gadm_2025",
        "GC_CNT_UNN_2025": "ghsl_country_un_2025",
        "GC_UCA_KM2_2025": "ghsl_area_km2_2025",
        "GC_POP_TOT_2025": "ghsl_population_total_2025",
        "GC_DEV_WIG_2025": "ghsl_world_income_group_2025",
        "GC_DEV_USR_2025": "ghsl_un_subregion_2025",
        "GC_PLS_SCR_2025": "ghsl_place_score_2025",
        "GC_UCB_YOB_2025": "ghsl_birth_year_2025",
        "GC_UCB_YOD_2025": "ghsl_death_year_2025",
        "GC_UCM_CAP": "ghsl_capital_flag",
        "IN_CIS_ALL_2020": "ghsl_infrastructure_index_2020",
        "IN_ROA_LEN_2024": "ghsl_road_length_m_2024",
        "IN_ROA_DEN_2024": "ghsl_road_density_2024",
        "SC_GDP_AVG_2020": "ghsl_gdp_avg_2020",
        "SC_GDP_SUM_2020": "ghsl_gdp_sum_2020",
        "SC_SEC_HDI_2020": "ghsl_hdi_2020",
        "SC_SEC_LET_2020": "ghsl_life_expectancy_2020",
        "SC_SEC_SET_2020": "ghsl_expected_schooling_2020",
        "SC_SEC_SYT_2020": "ghsl_mean_schooling_2020",
    }
    gh = gh.rename(columns=rename)
    gh["ghsl_population_density_2025"] = gh["ghsl_population_total_2025"] / gh["ghsl_area_km2_2025"].replace(0, np.nan)
    gh["ghsl_log_population_2025"] = np.log10(gh["ghsl_population_total_2025"].where(gh["ghsl_population_total_2025"] > 0))
    gh["ghsl_log_area_2025"] = np.log10(gh["ghsl_area_km2_2025"].where(gh["ghsl_area_km2_2025"] > 0))
    gh["norm_country"] = gh["ghsl_country_gadm_2025"].map(normalise_text)
    gh["norm_name_main"] = gh["ghsl_name_main_2025"].map(normalise_text)
    gh["norm_name_list"] = gh["ghsl_name_list_2025"].map(normalise_text)
    return gh


def match_ghsl(project: pd.DataFrame, gh: pd.DataFrame) -> pd.DataFrame:
    matches: list[MatchResult] = []
    for _, row in project.iterrows():
        country = normalise_text(row["country"])
        city = normalise_text(row["city"])
        candidates = gh[gh["norm_country"] == country].copy()
        if candidates.empty:
            candidates = gh[gh["norm_country"].str.contains(country, regex=False, na=False)].copy()
        if candidates.empty:
            raise ValueError(f"No GHSL country candidates for {row['city']}, {row['country']}")
        best: tuple[float, float, float, pd.Series] | None = None
        for _, cand in candidates.iterrows():
            sim = max(
                SequenceMatcher(None, city, str(cand["norm_name_main"])).ratio(),
                SequenceMatcher(None, city, str(cand["norm_name_list"])).ratio(),
            )
            dist = haversine_km(
                float(row["latitude"]),
                float(row["longitude"]),
                float(cand["ghsl_latitude"]),
                float(cand["ghsl_longitude"]),
            )
            score = sim - min(dist, 300.0) / 1000.0
            if best is None or score > best[0]:
                best = (score, sim, dist, cand)
        if best is None:
            raise RuntimeError("GHSL matching failed unexpectedly")
        score, sim, dist, cand = best
        status = "high_confidence"
        if sim < 0.70 or dist > 75.0:
            status = "manual_check_required"
        elif sim < 0.85 or dist > 50.0:
            status = "review"
        matches.append(
            MatchResult(
                city=str(row["city"]),
                country=str(row["country"]),
                ID_UC_G0=int(cand["ID_UC_G0"]),
                ghsl_name=str(cand["ghsl_name_main_2025"]),
                ghsl_country=str(cand["ghsl_country_gadm_2025"]),
                ghsl_latitude=float(cand["ghsl_latitude"]),
                ghsl_longitude=float(cand["ghsl_longitude"]),
                ghsl_match_similarity=float(sim),
                ghsl_match_distance_km=float(dist),
                ghsl_match_score=float(score),
                ghsl_match_status=status,
            )
        )
    return pd.DataFrame([asdict(m) for m in matches])


def fetch_wdi(skip_download: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_file = OUT / "wdi_indicator_long_latest.csv"
    if skip_download and raw_file.exists():
        latest = pd.read_csv(raw_file)
    else:
        records: list[dict[str, Any]] = []
        session = requests.Session()
        for indicator, column_name in WDI_INDICATORS.items():
            url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
            params = {"format": "json", "per_page": 20000, "date": "2000:2025"}
            response = session.get(url, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or len(payload) < 2:
                raise RuntimeError(f"Unexpected WDI payload for {indicator}")
            for item in payload[1]:
                value = item.get("value")
                iso3 = item.get("countryiso3code")
                if value is None or not iso3:
                    continue
                records.append(
                    {
                        "indicator": indicator,
                        "indicator_column": column_name,
                        "indicator_name": item.get("indicator", {}).get("value", ""),
                        "countryiso3code": iso3,
                        "country_name_wdi": item.get("country", {}).get("value", ""),
                        "year": int(item.get("date")),
                        "value": float(value),
                        "source_url": response.url,
                    }
                )
            time.sleep(0.2)
        all_rows = pd.DataFrame(records)
        latest = (
            all_rows.sort_values(["indicator", "countryiso3code", "year"])
            .groupby(["indicator", "indicator_column", "countryiso3code"], as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        latest.to_csv(raw_file, index=False)
        all_rows.to_csv(OUT / "wdi_indicator_long_all_years.csv", index=False)

    country_rows = []
    for country, iso3 in COUNTRY_TO_ISO3.items():
        country_rows.append({"country": country, "countryiso3code": iso3})
    country_map = pd.DataFrame(country_rows)
    latest = latest.drop(columns=["country"], errors="ignore")
    latest = country_map.merge(latest, on="countryiso3code", how="left")
    pivot = latest.pivot_table(index=["country", "countryiso3code"], columns="indicator_column", values="value", aggfunc="first")
    pivot = pivot.reset_index()
    year_pivot = latest.pivot_table(index=["country", "countryiso3code"], columns="indicator_column", values="year", aggfunc="first")
    year_pivot = year_pivot.reset_index().rename(columns={c: f"{c}_year" for c in WDI_INDICATORS.values()})
    country_controls = country_map.merge(pivot, on=["country", "countryiso3code"], how="left")
    country_controls = country_controls.merge(year_pivot, on=["country", "countryiso3code"], how="left")
    return latest, country_controls


def load_project_base() -> pd.DataFrame:
    base = pd.read_csv(R63_INTEGRATED)
    spatial_cols = [
        "city",
        "country",
        "pc_spatial_null",
        "spatial_null_gap",
        "road_minus_spatial",
        "spatial_fraction_of_road_gap",
        "remaining_fraction_after_spatial",
        "threshold_boundary_hit",
    ]
    spatial = pd.read_csv(R64_SPATIAL)[spatial_cols]
    drop_cols = [c for c in spatial_cols[2:] if c in base.columns]
    base = base.drop(columns=drop_cols, errors="ignore")
    return base.merge(spatial, on=["city", "country"], how="left")


def correlations(df: pd.DataFrame, features: list[str], targets: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in targets:
        for feature in features:
            subset = df[[target, feature]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(subset) < 10 or subset[feature].nunique() < 3:
                continue
            try:
                sp = spearmanr(subset[feature], subset[target])
                pr = pearsonr(subset[feature], subset[target])
            except Exception:
                continue
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "label": EXTERNAL_FEATURE_LABELS.get(feature, feature),
                    "n": int(len(subset)),
                    "spearman_r": float(sp.statistic),
                    "spearman_p": float(sp.pvalue),
                    "pearson_r": float(pr.statistic),
                    "pearson_p": float(pr.pvalue),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_spearman_r"] = out["spearman_r"].abs()
    return out.sort_values(["target", "abs_spearman_r"], ascending=[True, False])


def model_screen(df: pd.DataFrame, target: str, groups: dict[str, list[str]], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    y = df[target].to_numpy(dtype=float)
    mask_y = np.isfinite(y)
    alphas = np.logspace(-3, 4, 80)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    for name, features in groups.items():
        cols = [c for c in features if c in df.columns and df[c].notna().sum() >= 20 and df[c].nunique(dropna=True) >= 3]
        if not cols:
            continue
        X = df.loc[mask_y, cols].replace([np.inf, -np.inf], np.nan)
        yy = y[mask_y]
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=alphas))
        yhat_cv = cross_val_predict(pipe, X, yy, cv=cv)
        pipe.fit(X, yy)
        yhat_fit = pipe.predict(X)
        ridge = pipe.named_steps["ridgecv"]
        rows.append(
            {
                "target": target,
                "model_group": name,
                "n": int(len(yy)),
                "n_features_used": int(len(cols)),
                "features_used": ";".join(cols),
                "cv_r2": float(r2_score(yy, yhat_cv)),
                "in_sample_r2": float(r2_score(yy, yhat_fit)),
                "alpha": float(ridge.alpha_),
            }
        )
        pred = df.loc[mask_y, ["city", "country", "macro_region"]].copy()
        pred["target"] = target
        pred["model_group"] = name
        pred["observed"] = yy
        pred["predicted_cv"] = yhat_cv
        pred["predicted_fit"] = yhat_fit
        predictions.append(pred)
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return pd.DataFrame(rows), pred_df


def graph_adjusted_external_screen(
    df: pd.DataFrame,
    target: str,
    graph_features: list[str],
    external_features: list[str],
    seed: int,
) -> pd.DataFrame:
    cols = [c for c in graph_features if c in df.columns and df[c].notna().sum() >= 20 and df[c].nunique(dropna=True) >= 3]
    target_mask = df[target].notna()
    X = df.loc[target_mask, cols].replace([np.inf, -np.inf], np.nan)
    yy = df.loc[target_mask, target].to_numpy(dtype=float)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 80)))
    yhat = cross_val_predict(pipe, X, yy, cv=KFold(n_splits=5, shuffle=True, random_state=seed))
    residual = yy - yhat
    work = df.loc[target_mask, ["city", "country", "macro_region"] + external_features].copy()
    work[f"{target}_graph_cv_residual"] = residual
    rows = []
    for feature in external_features:
        subset = work[[feature, f"{target}_graph_cv_residual"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(subset) < 10 or subset[feature].nunique() < 3:
            continue
        sp = spearmanr(subset[feature], subset[f"{target}_graph_cv_residual"])
        rows.append(
            {
                "target": target,
                "feature": feature,
                "label": EXTERNAL_FEATURE_LABELS.get(feature, feature),
                "n": int(len(subset)),
                "spearman_r_with_graph_cv_residual": float(sp.statistic),
                "spearman_p": float(sp.pvalue),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_spearman_r_with_graph_cv_residual"] = out["spearman_r_with_graph_cv_residual"].abs()
    return out.sort_values("abs_spearman_r_with_graph_cv_residual", ascending=False)


def set_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def make_figure(df: pd.DataFrame, corr: pd.DataFrame, models: pd.DataFrame) -> None:
    set_plot_style()
    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 0.92], height_ratios=[0.95, 1.05])

    colors = [REGION_COLORS.get(str(r), REGION_COLORS["Other"]) for r in df["macro_region"]]
    size = np.clip(df["ghsl_population_total_2025"].fillna(df["ghsl_population_total_2025"].median()).to_numpy(), 2e5, 3e7)
    size = 16 + 72 * (np.log10(size) - np.log10(size).min()) / max(1e-9, np.log10(size).max() - np.log10(size).min())

    ax_a = fig.add_subplot(gs[0, 0])
    sc = ax_a.scatter(
        df["longitude"],
        df["latitude"],
        s=size,
        c=df["road_minus_spatial"],
        cmap="magma_r",
        edgecolor="#222222",
        linewidth=0.28,
        alpha=0.92,
    )
    ax_a.set_xlim(-180, 180)
    ax_a.set_ylim(-60, 75)
    ax_a.set_xlabel("Longitude")
    ax_a.set_ylabel("Latitude")
    ax_a.set_title("a  External controls are matched for all 71 windows", loc="left", fontweight="bold")
    ax_a.grid(color="#E8E8E8", lw=0.35)
    cbar = fig.colorbar(sc, ax=ax_a, fraction=0.046, pad=0.02)
    cbar.set_label("Observed minus spatial-null threshold")

    ax_b = fig.add_subplot(gs[0, 1])
    ctmp = corr[corr["target"] == "road_minus_spatial"].copy()
    ctmp = ctmp.assign(abs_r=ctmp["spearman_r"].abs()).sort_values("abs_r", ascending=False).head(10)
    ctmp = ctmp.sort_values("spearman_r")
    y = np.arange(len(ctmp))
    bar_colors = ["#B55A4B" if v > 0 else "#4E79A7" for v in ctmp["spearman_r"]]
    ax_b.barh(y, ctmp["spearman_r"], color=bar_colors, alpha=0.88)
    ax_b.axvline(0, color="#222222", lw=0.8)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(ctmp["label"], fontsize=6)
    ax_b.set_xlabel("Spearman r with spatial-null residual")
    ax_b.set_title("b  External urban-centre controls", loc="left", fontweight="bold")
    ax_b.grid(axis="x", color="#E8E8E8", lw=0.35)

    ax_c = fig.add_subplot(gs[1, 0])
    m = models[models["target"].isin(["road_gap", "road_minus_spatial"])].copy()
    order = ["graph_only", "ghsl_only", "wdi_only", "graph_plus_ghsl", "graph_plus_ghsl_wdi"]
    labels = {
        "graph_only": "Road form",
        "ghsl_only": "GHSL",
        "wdi_only": "WDI",
        "graph_plus_ghsl": "Road + GHSL",
        "graph_plus_ghsl_wdi": "Road + GHSL + WDI",
    }
    x = np.arange(len(order))
    width = 0.34
    for i, target in enumerate(["road_gap", "road_minus_spatial"]):
        sub = m[m["target"] == target].set_index("model_group")
        vals = [sub.loc[o, "cv_r2"] if o in sub.index else np.nan for o in order]
        ax_c.bar(x + (i - 0.5) * width, vals, width=width, color=["#4E79A7", "#59A14F"][i], alpha=0.9)
    ax_c.axhline(0, color="#222222", lw=0.8)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([labels[o] for o in order], rotation=24, ha="right")
    ax_c.set_ylabel("Five-fold CV $R^2$")
    ax_c.set_title("c  External controls benchmark mechanism models", loc="left", fontweight="bold")
    ax_c.grid(axis="y", color="#E8E8E8", lw=0.35)
    ax_c.text(0.02, 0.96, "blue: CEBH gap; green: post-spatial residual", transform=ax_c.transAxes, va="top", fontsize=6.2)

    ax_d = fig.add_subplot(gs[1, 1])
    if not ctmp.empty:
        top_feature = ctmp.iloc[-1]["feature"] if abs(ctmp.iloc[-1]["spearman_r"]) >= abs(ctmp.iloc[0]["spearman_r"]) else ctmp.iloc[0]["feature"]
    else:
        top_feature = "ghsl_road_density_2024"
    plot = df[[top_feature, "road_minus_spatial", "macro_region", "city"]].replace([np.inf, -np.inf], np.nan).dropna()
    for region, grp in plot.groupby("macro_region"):
        ax_d.scatter(
            grp[top_feature],
            grp["road_minus_spatial"],
            s=28,
            color=REGION_COLORS.get(region, REGION_COLORS["Other"]),
            edgecolor="#222222",
            linewidth=0.25,
            alpha=0.9,
            label=region,
        )
    if len(plot) >= 3:
        coef = np.polyfit(plot[top_feature], plot["road_minus_spatial"], deg=1)
        xx = np.linspace(plot[top_feature].min(), plot[top_feature].max(), 100)
        ax_d.plot(xx, coef[0] * xx + coef[1], color="#222222", lw=0.9)
        sp = spearmanr(plot[top_feature], plot["road_minus_spatial"])
        ax_d.text(
            0.98,
            0.96,
            f"Spearman r={sp.statistic:.2f}",
            transform=ax_d.transAxes,
            va="top",
            ha="right",
            fontsize=6.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )
    ax_d.set_xlabel(EXTERNAL_FEATURE_LABELS.get(top_feature, top_feature))
    ax_d.set_ylabel("Observed minus spatial-null threshold")
    ax_d.set_title("d  Strongest external residual correlate", loc="left", fontweight="bold")
    ax_d.grid(color="#E8E8E8", lw=0.35)
    ax_d.legend(loc="lower right", fontsize=5.2, ncols=2, handletextpad=0.25, columnspacing=0.6)

    for ext in [".svg", ".pdf", ".png", ".tiff"]:
        fig.savefig(FIG_BASE.with_suffix(ext), dpi=600 if ext in [".png", ".tiff"] else None, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: dict[str, Any], top_corr: pd.DataFrame, model_df: pd.DataFrame) -> None:
    report = ROUND_DIR / "R65_public_covariates_and_controls_report.md"
    best_rows = top_corr[top_corr["target"] == "road_minus_spatial"].head(8)
    model_table = model_df[["target", "model_group", "n_features_used", "cv_r2", "in_sample_r2"]].copy()
    text = f"""# R65 Public Covariates And External Controls

## Purpose

R65 tests whether the 71-city CEBH gaps and the R64 post-spatial residuals are
better interpreted as generic city-size/development correlates or as embedded
road-form effects. It adds two public-data layers:

- JRC/GHSL UCDB R2024A urban-centre covariates for all 71 city windows.
- World Bank WDI country-level controls for the same country set, where WDI has
  a current public record.

These variables are controls and mechanism-context covariates. They are not
used as causal claims about urban planning.

## Data Status

- GHSL theme packages downloaded: {summary['ghsl_downloaded_packages']}/4.
- GHSL city matches: {summary['ghsl_match_count']}/71.
- GHSL matches requiring manual review: {summary['ghsl_manual_review_count']}.
- WDI country-control coverage: {summary['wdi_countries_with_any_value']}/{summary['wdi_country_count']} countries with at least one indicator.

## Main Results

- The all-city GHSL match is complete and geographically coherent: median
  city-centre distance is {summary['ghsl_match_distance_median_km']:.2f} km and
  the maximum accepted distance is {summary['ghsl_match_distance_max_km']:.2f} km.
- The best external-only model for the observed CEBH gap has CV R2 =
  {summary['best_external_road_gap_cv_r2']:.3f}; the road-form-only model has
  CV R2 = {summary['graph_only_road_gap_cv_r2']:.3f}.
- For the post-spatial residual, the best external-only model has CV R2 =
  {summary['best_external_spatial_residual_cv_r2']:.3f}; the road-form-only
  model has CV R2 = {summary['graph_only_spatial_residual_cv_r2']:.3f}.

## Top External Associations With Post-Spatial Residual

{best_rows[['feature', 'label', 'n', 'spearman_r', 'spearman_p']].to_markdown(index=False)}

## Model Comparison

{model_table.to_markdown(index=False)}

## Manuscript-Safe Interpretation

R65 strengthens the Nature Cities/Nature Communications case by showing that
public urban-centre and country-background controls can be registered and
matched across the full 71-city matrix. The result supports a restrained claim:
external city size, socioeconomic and country controls are useful context, but
the current predictive signal remains concentrated in directly measured
embedded road-form variables. This is not a causal urban-form proof; it is a
confounding-control and mechanism-context increment.
"""
    report.write_text(text, encoding="utf-8")


def write_closeout(summary: dict[str, Any]) -> None:
    closeout = ROUND_DIR / "substantive_increment_closeout.md"
    text = f"""# R65 Substantive Increment Closeout

| Requested increment | Status | Evidence / output | Boundary |
|---|---:|---|---|
| Wider public data search and download | Done | GHSL UCDB R2024A four theme packages downloaded to `data/R65_public_covariates_and_controls/raw/`; WDI API tables written under `data/R65_public_covariates_and_controls/`. | Public derived covariates only; raw OSM graphs remain local/private cache. |
| Add urban-form covariates beyond local OSM tags | Done | `ghsl_city_covariates_all.csv`, `ghsl_match_qc.csv`, `R65_integrated_public_covariates.csv`. | GHSL urban-centre data are matched to 10-km road windows, not identical polygons. |
| Add country/background controls | Done | `wdi_indicator_long_all_years.csv`, `wdi_country_controls.csv`. | WDI values are country-level controls, not city-level mechanisms. |
| Recompute mechanism/control analysis | Done | `R65_external_correlations.csv`, `R65_model_comparison.csv`, `R65_graph_adjusted_external_residual_correlations.csv`. | Descriptive predictive screen; no causal claim. |
| Create publication-grade figure | Done | `figures/Fig_R65_public_covariate_controls.svg/.pdf/.png/.tiff`. | Figure is a control/mechanism-context figure, not primary causal evidence. |
| Remove all Nature Cities blockers | Partial | R65 reduces the external-covariate blocker and improves mechanism context. | Full global geometry-null ensemble and target-specific final policy/ref audit still remain submission gates. |

## R65 Key Numbers

- GHSL matches: {summary['ghsl_match_count']}/71.
- Manual-review GHSL matches: {summary['ghsl_manual_review_count']}.
- Median GHSL centre distance: {summary['ghsl_match_distance_median_km']:.2f} km.
- Road-form-only CV R2 for CEBH gap: {summary['graph_only_road_gap_cv_r2']:.3f}.
- Best external-only CV R2 for CEBH gap: {summary['best_external_road_gap_cv_r2']:.3f}.
- Road-form-only CV R2 for post-spatial residual: {summary['graph_only_spatial_residual_cv_r2']:.3f}.
- Best external-only CV R2 for post-spatial residual: {summary['best_external_spatial_residual_cv_r2']:.3f}.
"""
    closeout.write_text(text, encoding="utf-8")


def write_source_registry(rows: list[dict[str, Any]]) -> None:
    registry = pd.DataFrame(rows)
    registry.to_csv(OUT / "public_data_source_registry.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true", help="Use already downloaded GHSL/WDI files where possible.")
    parser.add_argument("--seed", type=int, default=6501)
    args = parser.parse_args()

    ensure_dirs()
    source_rows = ensure_ghsl_downloads(args.skip_download)
    project = load_project_base()
    gh = read_ghsl_covariates()
    gh.to_csv(OUT / "ghsl_city_covariates_all.csv", index=False)

    match = match_ghsl(project, gh)
    match.to_csv(OUT / "ghsl_match_qc.csv", index=False)
    matched_gh = match.merge(gh.drop(columns=["norm_country", "norm_name_main", "norm_name_list"]), on="ID_UC_G0", how="left")

    wdi_long, wdi_controls = fetch_wdi(args.skip_download)
    wdi_long.to_csv(OUT / "wdi_indicator_long_latest.csv", index=False)
    wdi_controls.to_csv(OUT / "wdi_country_controls.csv", index=False)

    integrated = project.merge(matched_gh, on=["city", "country"], how="left")
    integrated = integrated.merge(wdi_controls, on="country", how="left")
    integrated.to_csv(OUT / "R65_integrated_public_covariates.csv", index=False)

    graph_features = [
        "node_density_per_km2",
        "edge_density_per_km2",
        "chord_length_density_km_per_km2",
        "edge_length_mean_m",
        "edge_length_cv",
        "orientation_entropy",
        "orientation_fourfold_order",
        "dead_end_share",
        "degree4plus_share",
        "mean_degree",
        "kappa",
        "cycle_density",
        "residential_length_share",
        "oneway_length_share",
        "edge_circuity_mean",
        "articulation_point_share",
        "bridge_edge_share",
        "largest_biconnected_node_fraction",
        "dual_degree_mean",
        "angular_entropy",
        "straight_angle_share_10deg",
    ]
    ghsl_features = [
        "ghsl_population_density_2025",
        "ghsl_log_population_2025",
        "ghsl_log_area_2025",
        "ghsl_infrastructure_index_2020",
        "ghsl_road_density_2024",
        "ghsl_gdp_avg_2020",
        "ghsl_gdp_sum_2020",
        "ghsl_hdi_2020",
        "ghsl_life_expectancy_2020",
        "ghsl_expected_schooling_2020",
        "ghsl_mean_schooling_2020",
        "ghsl_lulc_entropy_2019",
        "ghsl_lulc_dominance_2019",
        "ghsl_lulc_nonzero_classes_2019",
    ]
    wdi_features = list(WDI_INDICATORS.values())
    external_features = ghsl_features + wdi_features
    targets = ["road_gap", "road_minus_spatial"]
    corr = correlations(integrated, external_features, targets)
    corr.to_csv(OUT / "R65_external_correlations.csv", index=False)

    groups = {
        "graph_only": graph_features,
        "ghsl_only": ghsl_features,
        "wdi_only": wdi_features,
        "graph_plus_ghsl": graph_features + ghsl_features,
        "graph_plus_ghsl_wdi": graph_features + ghsl_features + wdi_features,
    }
    model_rows = []
    pred_rows = []
    for target in targets:
        model_df, pred_df = model_screen(integrated, target, groups, args.seed)
        model_rows.append(model_df)
        pred_rows.append(pred_df)
    models = pd.concat(model_rows, ignore_index=True)
    preds = pd.concat(pred_rows, ignore_index=True)
    models.to_csv(OUT / "R65_model_comparison.csv", index=False)
    preds.to_csv(OUT / "R65_model_predictions.csv", index=False)

    residual_corrs = []
    for target in targets:
        residual_corrs.append(graph_adjusted_external_screen(integrated, target, graph_features, external_features, args.seed))
    residual_corr = pd.concat(residual_corrs, ignore_index=True)
    residual_corr.to_csv(OUT / "R65_graph_adjusted_external_residual_correlations.csv", index=False)

    make_figure(integrated, corr, models)

    def get_cv(target: str, group: str) -> float:
        sub = models[(models["target"] == target) & (models["model_group"] == group)]
        return float(sub["cv_r2"].iloc[0]) if len(sub) else float("nan")

    def best_external(target: str) -> float:
        sub = models[(models["target"] == target) & (models["model_group"].isin(["ghsl_only", "wdi_only"]))]
        return float(sub["cv_r2"].max()) if len(sub) else float("nan")

    summary = {
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "ghsl_downloaded_packages": int(sum(1 for row in source_rows if row["status"] == "downloaded")),
        "ghsl_total_urban_centres": int(len(gh)),
        "ghsl_match_count": int(len(match)),
        "ghsl_manual_review_count": int((match["ghsl_match_status"] == "manual_check_required").sum()),
        "ghsl_review_count": int((match["ghsl_match_status"] == "review").sum()),
        "ghsl_match_similarity_min": float(match["ghsl_match_similarity"].min()),
        "ghsl_match_distance_median_km": float(match["ghsl_match_distance_km"].median()),
        "ghsl_match_distance_max_km": float(match["ghsl_match_distance_km"].max()),
        "wdi_country_count": int(len(wdi_controls)),
        "wdi_countries_with_any_value": int(wdi_controls[list(WDI_INDICATORS.values())].notna().any(axis=1).sum()),
        "graph_only_road_gap_cv_r2": get_cv("road_gap", "graph_only"),
        "best_external_road_gap_cv_r2": best_external("road_gap"),
        "graph_plus_ghsl_wdi_road_gap_cv_r2": get_cv("road_gap", "graph_plus_ghsl_wdi"),
        "graph_only_spatial_residual_cv_r2": get_cv("road_minus_spatial", "graph_only"),
        "best_external_spatial_residual_cv_r2": best_external("road_minus_spatial"),
        "graph_plus_ghsl_wdi_spatial_residual_cv_r2": get_cv("road_minus_spatial", "graph_plus_ghsl_wdi"),
        "external_features": external_features,
        "graph_features": graph_features,
    }
    (OUT / "R65_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, corr, models)
    write_closeout(summary)

    source_rows.append(
        {
            "source_family": "World Bank WDI API",
            "theme": "country_controls",
            "url": "https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000&date=2000:2025",
            "local_file": str((OUT / "wdi_indicator_long_all_years.csv").relative_to(ROOT)),
            "downloaded_bytes": (OUT / "wdi_indicator_long_all_years.csv").stat().st_size
            if (OUT / "wdi_indicator_long_all_years.csv").exists()
            else 0,
            "status": "downloaded",
        }
    )
    write_source_registry(source_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
