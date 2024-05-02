import pandas as pd
import numpy as np
import cvxpy as cp
import sqlite3
import json
import itertools
import datetime

def round_weights(weights, n=4):
    if not isinstance(weights, list):
        weights = list(weights)
    weights = [round(w, n) for w in weights]
    weights /= sum(weights)
    return weights

class Portfolio():
    databases = {}

    def __init__(
            self, l1_opts=[0], l2_opts=[0], n_assets=49,
            date_training_end=None, n_days=None, date_training_start=None,
            training_window=False,
            testing_window=10, n_tscv=3, tscv_size=None, tscv_metric="sharpe",
            testing_metric="sharpe",
            inialize_data=True
        ) -> None:
        # Initialize dates
        self.date_training_end = date_training_end # in dataclass
        self.n_days = n_days # in dataclass
        self.date_training_start = date_training_start # in dataclass
        # Initialize hyperparameters
        self.n_assets = n_assets # in dataclass
        self.l1_opts = l1_opts # in dataclass
        self.l2_opts = l2_opts # in dataclass
        self.n_tscv = n_tscv # in dataclass
        if tscv_metric not in ["sharpe", "mean", "volatility"]:
            raise ValueError("Invalid time-series cross validaion metric")
        self.tscv_metric = tscv_metric # in dataclass
        self.tscv_size = testing_window if tscv_size is None else tscv_size # in dataclass
        self.testing_window = testing_window # in dataclass
        self.training_window = training_window if training_window not in [None, False, 0] else False # in dataclass
        self.training_tscv_data = {f"tscv_{i:02}": None for i in range(1, n_tscv+1)}
        self.testing_tscv_data = {f"tscv_{i:02}": None for i in range(1, n_tscv+1)}
        self.tscv_l_performance = {f"tscv_{i:02}": {} for i in range(1, n_tscv+1)}
        self.best_l = {} # in dataclass
        self.available_traning_data = None
        self.testing_data = None
        self.data = None
        # Results
        self.testing_metric = testing_metric # in dataclass
        self.training_weights = None # in dataclass
        self.portfolio_testing_returns = None # in dataclass
        self.testing_performance = {} # in dataclass
        self.testing_optimal_weights_performance = {} # in dataclass
        self.testing_optimal_l = {} # in dataclass
        self.exception = None # in dataclass
        if inialize_data:
            self.csv_name = f"industry_portfolios_{self.n_assets:02}_daily.csv"
            self.upload_data()
            self._define_training_testing_data(date_training_start, n_days, date_training_end)
            self._define_tscv_data()

    def _define_training_testing_data(self, date_training_start, n_days, date_training_end):
        if sum([n_days is None, date_training_start is None, date_training_end is None]) != 1:
            raise ValueError("Exactly two of end_date, n_days, and start_date must be specified")
        if date_training_end and n_days:
            self.available_traning_data = (
                self.data
                .loc[lambda df: df.index <= date_training_end]
                .iloc[-n_days:]
            )
        else:
            raise Exception("Not implemented yet")
        self.testing_data = (
            self.data
            .loc[lambda df: df.index > date_training_end]
            .iloc[:self.testing_window]
        )

    def _define_tscv_data(self):
        curr_index = len(self.available_traning_data.index)
        for i in range(self.n_tscv, 0, -1):
            initial_testing_index = curr_index - self.tscv_size
            new_tscv_testing_data = (
                self.available_traning_data
                .iloc[initial_testing_index:curr_index]
            )
            if initial_testing_index <= 0:
                raise ValueError("Not enough data to create a time-series cross validation sets")
            final_training_index = initial_testing_index - 1
            curr_index = initial_testing_index
            initial_training_index = (
                final_training_index - self.training_window + 1 if self.training_window else 0
            )
            if initial_training_index < 0:
                raise ValueError("Not enough data to create a time-series cross validation sets")
            new_tscv_training_data = (
                self.available_traning_data
                .iloc[initial_training_index:(final_training_index+1)]
            )
            if f"tscv_{i:02}" in self.training_tscv_data.keys() and f"tscv_{i:02}" in self.testing_tscv_data.keys():
                self.training_tscv_data[f"tscv_{i:02}"] = new_tscv_training_data
                self.testing_tscv_data[f"tscv_{i:02}"] = new_tscv_testing_data
            else:
                raise ValueError(f"Invalid time-series cross validation index: {i:02}")

    @staticmethod
    def calculate_performance(data, weights, metric="sharpe"):
        returns = data @ weights
        avg = returns.mean()
        std = returns.std()
        sharpe = avg / std
        if metric == "sharpe":
            return sharpe
        elif metric == "mean":
            return avg
        elif metric == "volatility":
            return std
        elif metric == "all":
            return {"sharpe": sharpe, "mean": avg, "volatility": std}
        else:
            raise ValueError("Invalid metric")
        
    @staticmethod
    def calculate_min_var_weights(data, **kwargs):
        l1 = kwargs.get("l1", 0)
        l2 = kwargs.get("l2", 0)
        n = data.shape[1]
        cov = data.cov()
        if l1 == 0:
            ones = np.ones(n)
            inv = np.linalg.inv(cov + l2 * np.eye(n))
            weights = inv @ ones / (ones @ inv @ ones)
            return round_weights(weights)
        else:
            cov = cov.values
            weights = cp.Variable(n)
            portfolio_variance = cp.quad_form(weights, cov + l2 * np.eye(n))
            l1_penalty = l1 * cp.norm(weights, 1)
            constraints = [cp.sum(weights) == 1]
            problem = cp.Problem(cp.Minimize(portfolio_variance + l1_penalty), constraints)
            problem.solve(solver=cp.SCS, verbose=False, eps=1e-6)
            if problem.status not in ["infeasible", "unbounded"]:
                return round_weights(weights.value)
            else:
                raise ValueError("Error in minimum variance optimization with L1 > 0")
            
    @staticmethod
    def calculate_max_sharpe_weights(data, **kwargs):
        raise Exception("Not implemented yet")
        l1 = kwargs.get("l1", 0)
        l2 = kwargs.get("l2", 0)
        n = data.shape[1]
        returns = data.mean()
        cov = data.cov()

        if l1 == 0:
            inv_cov = np.linalg.inv(cov + l2 * np.eye(n))
            weights = inv_cov @ returns
            return weights / weights.sum()
        else:
            cov = cov.values
            weights = cp.Variable(n)
            expected_return = returns @ weights
            portfolio_variance = cp.quad_form(weights, cov + l2 * np.eye(n))
            l1_penalty = l1 * cp.norm(weights, 1)
            constraints = [cp.sum(weights) == 1]
            problem = cp.Problem(cp.Maximize(expected_return - 0.5 * portfolio_variance - l1_penalty), constraints)
            problem.solve(solver=cp.SCS, verbose=False, eps=1e-8)
            if problem.status not in ["infeasible", "unbounded"]:
                return weights.value
            else:
                raise ValueError("Error in maximum Sharpe ratio optimization with L1 > 0")

    def tune_hyperparameters(self):
        for i in range(1, self.n_tscv+1):
            training_data = self.training_tscv_data[f"tscv_{i:02}"]
            testing_data = self.testing_tscv_data[f"tscv_{i:02}"]
            for l1, l2 in list(itertools.product(self.l1_opts, self.l2_opts)):
                weights = self.calculate_min_var_weights(training_data, l1=l1, l2=l2)
                weights = [round(w, 5) for w in list(weights)]
                weights /= sum(weights)
                performance = self.calculate_performance(testing_data, weights, self.tscv_metric)
                self.tscv_l_performance[f"tscv_{i:02}"][(l1, l2)] = performance

    def get_best_hyperparameters(self):
        best_hyperparameters = {}
        best_func = min if self.tscv_metric == "volatility" else max
        for i in range(1, self.n_tscv+1):
            best_hyperparameters[f"tscv_{i:02}"] = best_func(
                self.tscv_l_performance[f"tscv_{i:02}"],
                key=self.tscv_l_performance[f"tscv_{i:02}"].get
            )
        self.best_l = {
            "l1": np.mean([l[0] for l in list(best_hyperparameters.values())]),
            "l2": np.mean([l[1] for l in list(best_hyperparameters.values())])
        }

    def upload_data(self):
        if self.csv_name not in self.databases:
            self.__class__.databases[self.csv_name] = (
                pd.read_csv(f"data/{self.csv_name}", index_col=0, parse_dates=True)
                .sort_index()
            )
        self.data = self.__class__.databases[self.csv_name]

    def calculate_training_weights(self):
        if self.best_l == {}:
            raise ValueError("You must tune hyperparameters and get the best hyperparameters before calculating training weights")
        if self.training_window:
            training_data = self.available_traning_data.iloc[-self.training_window:]
        else:
            training_data = self.available_traning_data.copy()
        self.date_training_start = training_data.index[0]
        self.training_weights = self.calculate_min_var_weights(training_data, **self.best_l)

    def calculate_testing_returns(self):
        self.portfolio_testing_returns = self.testing_data @ self.training_weights

    def calculate_testing_performance(self):
        self.testing_performance = self.calculate_performance(
            self.testing_data,
            self.training_weights, metric="all"
        )

    def calculate_testing_optimal_l(self):
        testing_optimal_l = {"l1": 0, "l2": 0}
        # best_performance = -np.inf
        # for l1, l2 in list(itertools.product(self.l1_opts, self.l2_opts)):
        #     weights = self.calculate_min_var_weights(self.testing_data, l1=l1, l2=l2)
        #     performance = self.calculate_performance(self.testing_data, weights, self.testing_metric)
        #     if self.testing_metric == "volatility":
        #         performance *= -1
        #     if performance > best_performance:
        #         testing_optimal_l["l1"], testing_optimal_l["l2"] = l1, l2
        #         best_performance = performance
        self.testing_optimal_l = testing_optimal_l

    def calculate_testing_weights_optimal_l(self):
        if self.testing_optimal_l == {}:
            raise ValueError("You must calculate the optimal hyperparameters before calculating testing weights")
        self.testing_weights_optimal_l = self.calculate_min_var_weights(self.testing_data, **self.testing_optimal_l)

    def calculate_testing_optimal_weights_performance(self):
        self.testing_optimal_weights_performance = self.calculate_performance(
            self.testing_data,
            self.testing_weights_optimal_l, metric="all"
        )

    def add_exception(self, exception):
        if isinstance(exception, str):
            self.exception = exception
        self.exception = str(exception)

    def get_testing_end_date(self):
        testing_end_date = self.testing_data.index[-1]
        if isinstance(testing_end_date, (pd.Timestamp, (datetime.date, datetime.datetime))):
            testing_end_date = testing_end_date.strftime("%Y-%m-%d")
        return testing_end_date

    def save(self, db_name="portfolio.db"):
        # Convert necessary data into appropriate formats
        l1_opts_str = json.dumps(self.l1_opts)
        l2_opts_str = json.dumps(self.l2_opts)
        best_l_str = json.dumps(self.best_l)
        training_weights_str = json.dumps(self.training_weights.tolist() if self.training_weights is not None else None)
        portfolio_testing_returns_dict = self.portfolio_testing_returns.to_dict() if self.portfolio_testing_returns is not None else None
        if portfolio_testing_returns_dict is not None:
            portfolio_testing_returns_dict = {
                key.strftime("%Y-%m-%d"): value for key, value in portfolio_testing_returns_dict.items()
            }
        if isinstance(self.date_training_end, (pd.Timestamp, datetime.date, datetime.datetime)):
            date_training_end_str = self.date_training_end.strftime("%Y-%m-%d")
        elif not self.date_training_end:
            date_training_end_str = None
        elif isinstance(self.date_training_end, str):
            date_training_end_str = self.date_training_end
        else:
            raise ValueError("Invalid date_training_end")
        if isinstance(self.date_training_start, (pd.Timestamp, datetime.date, datetime.datetime)):
            date_training_start_str = self.date_training_start.strftime("%Y-%m-%d")
        elif not self.date_training_start:
            date_training_start_str = None
        elif isinstance(self.date_training_start, str):
            date_training_start_str = self.date_training_start
        else:
            raise ValueError("Invalid date_training_start")
        portfolio_testing_returns_str = json.dumps(portfolio_testing_returns_dict)
        testing_performance_str = json.dumps(self.testing_performance)
        testing_optimal_weights_performance_str = json.dumps(self.testing_optimal_weights_performance)
        testing_optimal_l_str = json.dumps(self.testing_optimal_l)
        exception_str = self.exception if self.exception else None
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO portfolio (
                n_assets, l1_opts, l2_opts,
                n_days, n_tscv, date_training_start, date_training_end, best_l,
                tscv_metric, tscv_size, testing_window, testing_metric,
                training_window, training_weights, portfolio_testing_returns, testing_performance,
                testing_optimal_weights_performance, exception
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.n_assets, l1_opts_str, l2_opts_str,
            self.n_days, self.n_tscv, date_training_start_str, date_training_end_str, best_l_str,
            self.tscv_metric, self.tscv_size, self.testing_window, self.testing_metric,
            self.training_window, training_weights_str, portfolio_testing_returns_str, testing_performance_str,
            testing_optimal_weights_performance_str, exception_str
        ))
        conn.commit()
        conn.close()

    def delete_last_line(self):
        conn = sqlite3.connect("portfolio.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM portfolio WHERE id = (SELECT MAX(id) FROM portfolio)")
        conn.commit()
        conn.close()
    