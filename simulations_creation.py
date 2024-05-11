import pandas as pd
import numpy as np
import cvxpy as cp
import sqlite3
import json
import itertools
from portfolio import Portfolio

L1_OPTS = [l / 10 for l in list(range(12, 31, 2))]
L1_OPTS = [0]
L2_OPTS = [l / 10 for l in list(range(0, 11, 1))] + [l / 10 for l in list(range(12, 31, 2))]


def get_unique_results(table="portfolio"):
    with sqlite3.connect("portfolio.db") as conn:
        query = f"""
            SELECT DISTINCT
                n_assets,
                training_window,
                testing_window,
                n_tscv,
                tscv_size,
                tscv_metric,
                testing_metric,
                n_days
            FROM {table}
        """
        portfolio_results = pd.read_sql_query(query, conn)
        return portfolio_results
    

def params_already_calculated(params, table="portfolio", from_sql=True):
    if from_sql:
        if isinstance(table, str):
            with sqlite3.connect("portfolio.db") as conn:
                query = f"""
                    SELECT *
                    FROM {table}
                    WHERE
                        n_assets = {params["n_assets"]}
                        AND training_window = {params["training_window"]}
                        AND testing_window = {params["testing_window"]}
                        AND n_tscv = {params["n_tscv"]}
                        AND tscv_size = {params["tscv_size"]}
                        AND tscv_metric = '{params["tscv_metric"]}'
                        AND testing_metric = '{params["testing_metric"]}'
                        AND n_days = {params["n_days"]}
                """
                portfolio_results = pd.read_sql_query(query, conn)
                return not portfolio_results.empty
        else:
            raise ValueError("If `from_sql=True` table must be a string.")
    else:
        if isinstance(table, pd.DataFrame):
            return not table[
                (table["n_assets"] == params["n_assets"])
                & (table["training_window"] == params["training_window"])
                & (table["testing_window"] == params["testing_window"])
                & (table["n_tscv"] == params["n_tscv"])
                & (table["tscv_size"] == params["tscv_size"])
                & (table["tscv_metric"] == params["tscv_metric"])
                & (table["testing_metric"] == params["testing_metric"])
                & (table["n_days"] == params["n_days"])
            ].empty
        else:
            raise ValueError("If `from_sql=False` table must be a DataFrame.")
            
def params_to_string(params, date_training_end):
    params_str = [
        f"{k}: {v:.1f}"
        if k not in ("tscv_metric", "testing_metric")
        else f"{k}: {v}"
        for k, v in params.items()
    ]
    params_str = ", ".join(params_str) + f", date_training_end: {date_training_end}"
    return params_str

parameters = {
    "n_assets": [5, 10, 12, 17, 30],
    "training_window": [63, 126, 252, 504],
    "testing_window": [5, 10, 21, 42, 63, 126, 252],
    "n_tscv": [1, 2, 3, 4, 5, 6, 7, 8],
    "tscv_size_multiple": [.5, .75, 1, 1.25, 1.5],
    "tscv_metric": ["sharpe"],
    "testing_metric": ["sharpe"],
    "n_days": [252, 504, 756, 1008, 1260, 12600],
}

parameter_combinations = [
    dict(zip(parameters.keys(), combination))
    for combination in itertools.product(*parameters.values())
]

parameter_combinations = [
    p for p in parameter_combinations
    if (
        (p["training_window"] != False and p["n_days"] == 12600)
        or (p["testing_window"] == False and p["n_days"] != 12600)
    )
]

if __name__ == "__main__":
    for params in parameter_combinations:
        params["tscv_size"] = round(params["training_window"] * params["tscv_size_multiple"])
        if params_already_calculated(params):
            print(f"SKIP: {params_to_string(params, '2000-01-01')}")
            continue
        date_training_end = "2000-01-01"
        while pd.to_datetime(date_training_end) < pd.to_datetime("2022-01-01"):
            try:
                portfolio = Portfolio(
                    n_assets=params["n_assets"],
                    l1_opts=L1_OPTS,
                    l2_opts=L2_OPTS,
                    date_training_end=date_training_end,
                    n_days=params["n_days"],
                    tscv_size=params["tscv_size"],
                    testing_window=params["testing_window"],
                    training_window=params["training_window"],
                    n_tscv=params["n_tscv"],
                    tscv_metric=params["tscv_metric"],
                    testing_metric=params["testing_metric"],
                )
                portfolio.tune_hyperparameters()
                portfolio.get_best_hyperparameters()
                portfolio.calculate_training_weights()
                portfolio.calculate_testing_returns()
                portfolio.calculate_testing_performance()
                portfolio.calculate_testing_optimal_l()
                portfolio.calculate_testing_weights_optimal_l()
                portfolio.calculate_testing_optimal_weights_performance()
            except Exception as e:
                portfolio.add_exception(str(e))
                print(f"ERROR: {portfolio}: {str(e)}")
            try:
                portfolio.save()
            except Exception as e:
                print(f"ERROR SAVING: {params_to_string(params, date_training_end)}: {str(e)}")
                break
            print(f"COMPLETED: {portfolio}")
            date_training_end = portfolio.get_testing_end_date()