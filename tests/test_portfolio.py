import pytest
import re
import pandas as pd
import numpy as np
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from portfolio import Portfolio


def check_dates_are_sequential(data, first_date, second_date):
    data_filtered = data.loc[lambda df: df.index >= first_date]
    return data_filtered.index[0] == first_date and data_filtered.index[1] == second_date


def init_portfolio():
    return Portfolio(n_assets=49, date_training_end="2023-04-10", n_days=70)

def test_initialization():
    portfolio = Portfolio(
        n_assets=49,
        date_training_end="2023-04-10",
        n_days=70,
        testing_window=10,
        n_tscv=3,
    )
    assert portfolio.date_training_end == "2023-04-10"
    assert portfolio.n_days == 70
    assert portfolio.training_window is False
    assert portfolio.testing_window == 10
    assert portfolio.n_tscv == 3
    assert portfolio.tscv_size == 10
    assert portfolio.tscv_metric == "sharpe"
    assert portfolio.testing_metric == "sharpe"
    assert portfolio.data is not None
    assert len(portfolio.available_traning_data.index) == 70
    assert len(portfolio.available_traning_data.columns) == 49
    assert len(portfolio.testing_data.columns) == 49
    assert portfolio.n_assets == 49
    assert portfolio.n_tscv == 3
    assert portfolio.testing_window == 10
    assert len(portfolio.testing_data.index) == 10
    assert len(portfolio.training_tscv_data.keys()) == 3
    assert len(portfolio.testing_tscv_data.keys()) == 3
    assert len(portfolio.tscv_l_performance.keys()) == 3

def test_upload_data():
    n_assets_opts = [5, 10, 12, 17, 30, 38, 48, 49]
    for n in n_assets_opts:
        _ = Portfolio(n_assets=n, date_training_end="2023-04-10", n_days=70)
    assert len(Portfolio.databases.keys()) == len(n_assets_opts)
    for n in n_assets_opts:
        _ = Portfolio(n_assets=n, date_training_end="2023-04-10", n_days=70)
    assert len(Portfolio.databases.keys()) == len(n_assets_opts)

def test_cross_validation_data():
    portfolio = portfolio = Portfolio(
        n_assets=49,
        date_training_end="2023-04-10",
        n_days=70,
        training_window=False,
        testing_window=15,
        n_tscv=3,
        tscv_size=20
    )
    assert portfolio.available_traning_data is not None
    assert portfolio.testing_data is not None
    for tscv_i, data in portfolio.testing_tscv_data.items():
        assert tscv_i[:4] == "tscv"
        assert len(data.index) == 20
        assert len(data.columns) == 49
    previous_tscv_size = -np.inf
    for tscv_i, data in portfolio.training_tscv_data.items():
        assert tscv_i[:4] == "tscv"
        assert len(data.columns) == 49
        assert len(data.index) > previous_tscv_size
        previous_tscv_size = len(data.index)

def test_cross_validation_rolling_window_data():
    portfolio = portfolio = Portfolio(
        n_assets=49,
        date_training_end="2023-04-10",
        n_days=800,
        training_window=50,
        testing_window=5,
        n_tscv=6,
        tscv_size=15
    )
    assert portfolio.available_traning_data is not None
    assert portfolio.testing_data is not None
    for tscv_i, data in portfolio.testing_tscv_data.items():
        assert tscv_i[:4] == "tscv"
        assert len(data.index) == 15
        assert len(data.columns) == 49
    for tscv_i, data in portfolio.training_tscv_data.items():
        assert tscv_i[:4] == "tscv"
        assert len(data.columns) == 49
        assert len(data.index) == 50

def test_cross_validation_dates():
    p1 = Portfolio(
        n_assets=49,
        date_training_end="2023-04-10",
        n_days=800,
        training_window=50,
        testing_window=5,
        n_tscv=6,
        tscv_size=15
    )
    p2 = Portfolio(
        n_assets=38,
        date_training_end="2022-01-01",
        n_days=800,
        training_window=30,
        testing_window=20,
        n_tscv=10,
        tscv_size=20
    )
    p3 = Portfolio(
        n_assets=30,
        date_training_end="2020-01-01",
        n_days=5000,
        training_window=252,
        testing_window=252,
        n_tscv=2,
        tscv_size=252
    )
    for portfolio in [p1, p2, p3]:
        assert portfolio.available_traning_data is not None
        assert portfolio.testing_data is not None
        for tscv_i in portfolio.testing_tscv_data.keys():
            training_tscv_data = portfolio.training_tscv_data[tscv_i]
            testing_tscv_data = portfolio.testing_tscv_data[tscv_i]
            threshold_dates = portfolio.data.loc[lambda df: df.index >= training_tscv_data.index[-1]].iloc[:2]
            assert threshold_dates.index[0] == training_tscv_data.index[-1]
            assert threshold_dates.index[1] == testing_tscv_data.index[0]

def test_diff_cross_validation_dates():
    p1 = Portfolio(
        n_assets=49,
        date_training_end="2023-04-10",
        n_days=800,
        training_window=50,
        testing_window=5,
        n_tscv=6,
        tscv_size=15
    )
    p2 = Portfolio(
        n_assets=38,
        date_training_end="2022-01-01",
        n_days=800,
        training_window=30,
        testing_window=20,
        n_tscv=10,
        tscv_size=20
    )
    p3 = Portfolio(
        n_assets=30,
        date_training_end="2020-01-01",
        n_days=5000,
        training_window=252,
        testing_window=252,
        n_tscv=2,
        tscv_size=252
    )
    for portfolio in [p1, p2, p3]:
        initial_final_dates = {}
        for tscv_i in portfolio.testing_tscv_data.keys():
            training_tscv_data = portfolio.training_tscv_data[tscv_i]
            testing_tscv_data = portfolio.testing_tscv_data[tscv_i]
            initial_final_dates[tscv_i] = [
                training_tscv_data.index[0], training_tscv_data.index[-1],
                testing_tscv_data.index[0], testing_tscv_data.index[-1]
            ]
        for i in range(1, portfolio.n_tscv-1):
            assert initial_final_dates[f"tscv_{i:02}"][3] == initial_final_dates[f"tscv_{(i+1):02}"][1]
            assert check_dates_are_sequential(
                portfolio.data, initial_final_dates[f"tscv_{i:02}"][3],
                initial_final_dates[f"tscv_{(i+1):02}"][2]
            )

def test_calculate_performance():
    data = (
        pd.read_csv("data/industry_portfolios_49_daily.csv", index_col=0, parse_dates=True)
        .sort_index()
        .loc["2015-01-01":"2023-01-01"]
    )
    weights = np.ones(49) / 49
    sharpe = Portfolio.calculate_performance(data, weights, "sharpe")
    mean = Portfolio.calculate_performance(data, weights, "mean")
    volatility = Portfolio.calculate_performance(data, weights, "volatility")
    assert round(sharpe, 5) == round(mean / volatility, 5)

def test_calculate_min_var_weights():
    data = (
        pd.read_csv("data/industry_portfolios_49_daily.csv", index_col=0, parse_dates=True)
        .sort_index()
        .loc["2015-01-01":"2023-01-01"]
    )
    weights = Portfolio.calculate_min_var_weights(data, l1=0, l2=0)
    assert len(weights) == 49
    assert round(weights.sum(), 5) == 1

def test_calculate_l2_min_var_weights():
    data = (
        pd.read_csv("data/industry_portfolios_49_daily.csv", index_col=0, parse_dates=True)
        .sort_index()
        .loc["2015-01-01":"2023-01-01"]
    )
    weights = Portfolio.calculate_min_var_weights(data, l1=0, l2=0)
    previous_weights_var = weights.var()
    for l2_opt in [.1, .3, .5, .75, 1, 1.5, 2, 3, 5, 10]:
        l2_weights = Portfolio.calculate_min_var_weights(data, l1=0, l2=l2_opt)
        assert round(l2_weights.sum(), 5) == 1
        assert round(l2_weights.var(), 5) <= round(previous_weights_var, 5)
        previous_weights_var = l2_weights.var()
    extreme_l2_weights = Portfolio.calculate_min_var_weights(data, l1=0, l2=10e12)
    avg_extreme_l2_weights = extreme_l2_weights.mean()
    diff_avg_extreme_l2_weights = [round(w - avg_extreme_l2_weights, 2) for w in list(extreme_l2_weights)]
    assert all([diff == 0 for diff in diff_avg_extreme_l2_weights])

def test_calculate_l1_min_var_weights():
    data = (
        pd.read_csv("data/industry_portfolios_49_daily.csv", index_col=0, parse_dates=True)
        .sort_index()
        .loc["2015-01-01":"2023-01-01"]
    )
    weights = Portfolio.calculate_min_var_weights(data, l1=0, l2=0)
    previous_weights_abs_sum = abs(weights).sum()
    for l1_opt in [.1, .3, .5, .75, 1, 1.5, 2, 3, 5, 10] + list(range(20, 501, 10)):
        print(l1_opt)
        l1_weights = Portfolio.calculate_min_var_weights(data, l1=l1_opt, l2=0)
        assert round(l1_weights.sum(), 4) == 1
        assert round(abs(l1_weights).sum(), 4) <= round(previous_weights_abs_sum, 4)
        previous_weights_abs_sum = abs(l1_weights).sum()

def test_get_best_hyperparameters():
    portfolio = Portfolio(
        n_assets=49,
        date_training_end="2023-04-10",
        l2_opts=[0, 0.1, 0.5, 1],
        n_days=800,
        training_window=50,
        testing_window=5,
        n_tscv=6,
        tscv_size=15
    )
    portfolio.tune_hyperparameters()
    assert portfolio.best_l == {}
    portfolio.get_best_hyperparameters()
    assert "l1" in portfolio.best_l.keys() and "l2" in portfolio.best_l.keys()
    assert portfolio.best_l["l1"] >= 0 and portfolio.best_l["l2"] >= 0

def test_calculate_training_weights():
    portfolio = Portfolio(
        n_assets=49,
        date_training_end="2023-04-10",
        l2_opts=[0, 0.1, 0.5, 1],
        n_days=800,
        training_window=50,
        testing_window=5,
        n_tscv=6,
        tscv_size=15
    )
    try:
        portfolio.calculate_training_weights()
        assert False, "Should raise an exception"
    except:
        pass
    portfolio.tune_hyperparameters()
    portfolio.get_best_hyperparameters()
    portfolio.calculate_training_weights()
    assert portfolio.training_weights is not None
    assert len(portfolio.training_weights) == 49
    assert round(portfolio.training_weights.sum(), 5) == 1

def test_testing_returns():
    portfolio = Portfolio(
        n_assets=49,
        date_training_end="2023-04-10",
        l2_opts=[0, 0.1, 0.5, 1],
        n_days=800,
        training_window=50,
        testing_window=5,
        n_tscv=6,
        tscv_size=15
    )
    portfolio.tune_hyperparameters()
    portfolio.get_best_hyperparameters()
    portfolio.calculate_training_weights()
    portfolio.calculate_testing_returns()
    assert portfolio.portfolio_testing_returns is not None
    assert len(portfolio.portfolio_testing_returns) == 5
    assert check_dates_are_sequential(
        portfolio.data,
        portfolio.available_traning_data.index[-1],
        portfolio.portfolio_testing_returns.index[0]
    )

def test_save_portfolio():
    portfolio = Portfolio(
        n_assets=49,
        date_training_end="2023-04-10",
        l1_opts=[0, 0.1, 0.5, 1],
        l2_opts=[0, 0.1, 0.5, 1],
        n_days=800,
        training_window=50,
        testing_window=15,
        n_tscv=6,
        tscv_size=15
    )
    portfolio.tune_hyperparameters()
    portfolio.get_best_hyperparameters()
    portfolio.calculate_training_weights()
    portfolio.calculate_testing_returns()
    portfolio.save()
    portfolio.delete_last_line()

def test_possibly_singular_value_matrix():
    portfolio = Portfolio(
        n_assets=10,
        date_training_end="2003-08-25",
        l1_opts=[0],
        l2_opts=[0],
        n_days=12600,
        training_window=126,
        testing_window=5,
        n_tscv=5,
        tscv_size=32
    )
    portfolio.tune_hyperparameters()
    portfolio.get_best_hyperparameters()
    portfolio.calculate_training_weights()
    portfolio.calculate_testing_returns()

if __name__ == "__main__":
    test_possibly_singular_value_matrix()
    test_initialization()
    test_save_portfolio()
    test_calculate_l1_min_var_weights()
    test_get_best_hyperparameters()
    test_calculate_performance()
    test_calculate_l2_min_var_weights()
    test_testing_returns()
    test_calculate_training_weights()
    test_cross_validation_data()
    test_cross_validation_rolling_window_data()
    test_cross_validation_dates()
    test_diff_cross_validation_dates()
    test_calculate_min_var_weights()
    test_upload_data()