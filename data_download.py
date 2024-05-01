import requests
from zipfile import ZipFile
import io
import pandas as pd

def download_and_extract_zip(url, endswith=".csv", skiprows=9):
    response = requests.get(url)
    response.raise_for_status()
    with ZipFile(io.BytesIO(response.content)) as zip_file:
        for name in zip_file.namelist():
            if name.endswith(endswith):
                with zip_file.open(name) as csv_file:
                    df = pd.read_csv(csv_file, skiprows=skiprows, low_memory=False)
                    return df
                

def organize_returns(df, keep_columns=None, lowercase_names=False):
    df = df.rename(columns={"Unnamed: 0": "date"})
    df = df.loc[lambda df: ~df.date.isin(["Copyright 2024 Kenneth R. French"])]
    marker_idx = df[df["date"] == "  Average Equal Weighted Returns -- Daily"].index
    if len(marker_idx) > 0:
        df = df.loc[:marker_idx[0] - 1]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace(-99.99, pd.NA)
    df = df / 100
    if keep_columns is not None:
        if isinstance(keep_columns, str):
            keep_columns = [keep_columns]
        df = df[keep_columns]
    if lowercase_names:
        df.columns = [col.lower() for col in df.columns]
    return df

URLS = {
    "risk_free": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
    "industry_portfolios_49_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip",
    "industry_portfolios_48_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/48_Industry_Portfolios_daily_CSV.zip",
    "industry_portfolios_38_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/38_Industry_Portfolios_daily_CSV.zip",
    "industry_portfolios_30_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_daily_CSV.zip",
    "industry_portfolios_17_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/17_Industry_Portfolios_daily_CSV.zip",
    "industry_portfolios_12_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_daily_CSV.zip",
    "industry_portfolios_10_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Industry_Portfolios_daily_CSV.zip",
    "industry_portfolios_05_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/5_Industry_Portfolios_daily_CSV.zip",
}

def download_industry_portfolio_returns():
    for name, url in URLS.items():
        if name == "risk_free":
            returns = download_and_extract_zip(url, endswith=".CSV", skiprows=4)
            returns = organize_returns(returns, keep_columns="RF", lowercase_names=True)
        else:
            returns = download_and_extract_zip(url)
            returns = organize_returns(returns)
        returns.to_csv(f"data/{name}.csv")


if __name__ == "__main__":
    download_industry_portfolio_returns()