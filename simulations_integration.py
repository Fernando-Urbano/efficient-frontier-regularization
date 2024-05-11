import pandas as pd
import time
import numpy as np
import cvxpy as cp
import sqlite3
import json
from portfolio import Portfolio



def get_unique_results():
    with sqlite3.connect("portfolio.db") as conn:
        query = """
            SELECT DISTINCT
                n_assets,
                training_window,
                testing_window,
                n_tscv,
                tscv_size,
                tscv_metric,
                testing_metric,
                n_days
            FROM portfolio
            ORDER BY date_training_end DESC
        """
        unique_results = pd.read_sql_query(query, conn)
    unique_results.reset_index(drop=True, inplace=True)
    unique_results = unique_results.reset_index().rename(columns={"index": "id"})
    return unique_results


def join_testing_returns(testing_returns_dicts, id):
    testing_returns = {}
    for returns_dict in testing_returns_dicts:
        returns_dict = {
            date: ret for date, ret in json.loads(returns_dict).items()
            if ret not in [np.nan, np.inf, -np.inf, None, "NaN", "nan", "inf", "-inf"]
        }
        testing_returns.update(returns_dict)
    return pd.DataFrame(testing_returns, index=[id]).transpose()


if __name__ == "__main__":
    unique_results = get_unique_results()
    start_time = time.time()
    portfolio_testing_returns_entire_sample = pd.DataFrame()
    for i in range(len(unique_results.index)):
        with sqlite3.connect("portfolio.db") as conn:
            query = f"""
                SELECT * FROM portfolio
                WHERE
                    n_assets = {unique_results.loc[i, "n_assets"]}
                    AND training_window = {unique_results.loc[i, "training_window"]}
                    AND testing_window = {unique_results.loc[i, "testing_window"]}
                    AND n_tscv = {unique_results.loc[i, "n_tscv"]}
                    AND tscv_size = {unique_results.loc[i, "tscv_size"]}
                    AND tscv_metric = '{unique_results.loc[i, "tscv_metric"]}'
                    AND testing_metric = '{unique_results.loc[i, "testing_metric"]}'
                    AND n_days = {unique_results.loc[i, "n_days"]}
            """
            try:
                calculations_params_combination = pd.read_sql_query(query, conn)
            except Exception as e:
                print(f"Error occurred: {e}")
            new_portfolio_testing_returns = join_testing_returns(
                calculations_params_combination.portfolio_testing_returns, unique_results.loc[i, "id"]
            )
            portfolio_testing_returns_entire_sample = (
                portfolio_testing_returns_entire_sample
                .join(new_portfolio_testing_returns, how="outer")
            )
            if (i % 100 == 0 or i == len(unique_results.index) - 1) and i != 0:
                current_time = time.time()
                elapsed_time = current_time - start_time
                remaining_combinations = len(unique_results.index) - i
                remaining_time = elapsed_time * remaining_combinations / i / 60
                print(f"COMPLETED {i}/{(len(unique_results.index) - 1)}; REMAINING TIME: {remaining_time:.0f} minutes")
    unique_results.to_csv("output/simulations_ids.csv")
    portfolio_testing_returns_entire_sample.to_csv("output/simulations_integrated_testing_returns.csv")
    
    