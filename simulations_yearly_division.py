import pandas as pd
import numpy as np



def calc_yearly_performance_metrics(integrated_returns):
    integrated_returns = integrated_returns.copy()
    integrated_returns.index = integrated_returns.index
    integrated_returns["year"] = integrated_returns.index.year
    yearly_performance_metrics = (
        integrated_returns
        .groupby("year")
        .agg(["mean", "std", "count"])
        .stack(0)
        .reset_index()
        .rename(columns={"level_1": "id"})
        .assign(id=lambda df: df.id.astype(int))
        .sort_values(["id", "year"])
        .assign(sharpe=lambda df: df["mean"] / df["std"] * np.sqrt(252))
        .assign(annualized_mean=lambda df: df["mean"] * 252)
        .assign(annualized_std=lambda df: df["std"] * np.sqrt(252))
        .drop(columns=["mean", "std"])
    )
    return yearly_performance_metrics

if __name__ == "__main__":
    integrated_testing_returns = (
        pd.read_csv("output/simulations_integrated_testing_returns.csv", index_col=0, parse_dates=True)
        .loc[:"2021-12-31"]
    )
    yearly_performance_metrics = calc_yearly_performance_metrics(integrated_testing_returns)
    integrated_ids = pd.read_csv("output/simulations_ids.csv")
    simulations_yearly_performance_metrics = integrated_ids.merge(yearly_performance_metrics, on="id", how="outer")
    simulations_yearly_performance_metrics.to_csv("output/simulations_yearly_performance_metrics.csv")





