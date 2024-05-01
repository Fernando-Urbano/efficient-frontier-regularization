import pandas as pd
import numpy as np
import itertools

class Portfolio():
    databases = {}

    def __init__(
            self, l1_opts=[0], l2_opts=[0], n_assets=49,
            date_training_end=None, n_days=None, date_training_start=None,
            training_window=False,
            testing_window=10, n_tscv=3, tscv_size=None, tscv_metric="sharpe",
            inialize_data=True
        ) -> None:
        # Initialize hyperparameters
        self.l1_opts = l1_opts
        self.l2_opts = l2_opts
        self.n_tscv = n_tscv
        if tscv_metric not in ["sharpe", "mean", "volatility"]:
            raise ValueError("Invalid time-series cross validaion metric")
        self.tscv_metric = tscv_metric
        self.tscv_size = testing_window if tscv_size is None else tscv_size
        self.testing_window = testing_window
        self.training_window = training_window if training_window not in [None, False, 0] else False
        self.training_tscv_data = {f"tscv_{i:02}": None for i in range(1, n_tscv+1)}
        self.testing_tscv_data = {f"tscv_{i:02}": None for i in range(1, n_tscv+1)}
        self.tscv_l_performance = {f"tscv_{i:02}": {} for i in range(1, n_tscv+1)}
        self.best_l = {}
        self.available_traning_data = None
        self.testing_data = None
        self.data = None
        # Results
        self.training_weights = None
        self.portfolio_testing_returns = None
        self.testing_performance = {}
        self.testing_optimal_l = {}
        # Initialize data
        if inialize_data:
            self.n_assets = n_assets
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
        if metric == "sharpe":
            return avg / std
        elif metric == "mean":
            return avg
        elif metric == "volatility":
            return std
        else:
            raise ValueError("Invalid metric")
        
    @staticmethod
    def calculate_min_var_weights(data, **kwargs):
        l1 = kwargs.get("l1", 0)
        l2 = kwargs.get("l2", 0)
        if l1 == 0:
            n = data.shape[1]
            cov = data.cov()
            ones = np.ones(n)
            inv = np.linalg.inv(cov + l2 * np.eye(n))
            weights = inv @ ones / (ones @ inv @ ones)
            return weights

    def tune_hyperparameters(self):
        for i in range(1, self.n_tscv+1):
            training_data = self.training_tscv_data[f"tscv_{i:02}"]
            testing_data = self.testing_tscv_data[f"tscv_{i:02}"]
            for l1, l2 in list(itertools.product(self.l1_opts, self.l2_opts)):
                weights = self.calculate_min_var_weights(training_data, l1=l1, l2=l2)
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
        self.training_weights = self.calculate_min_var_weights(training_data, **self.best_l)

    def calculate_testing_returns(self):
        self.portfolio_testing_returns = self.testing_data @ self.training_weights

    def calculate_testing_performance(self):
        self.testing_performance = {
            "sharpe": self.calculate_performance(self.testing_data, self.training_weights, "sharpe"),
            "mean": self.calculate_performance(self.testing_data, self.training_weights, "mean"),
            "volatility": self.calculate_performance(self.testing_data, self.training_weights, "volatility")
        }

    def calculate_testing_optimal_l(self):
        testing_optimal_l = {"l1": 0, "l2": 0}
        best_performance = -np.inf
        for l1, l2 in list(itertools.product(self.l1_opts, self.l2_opts)):
            weights = self.calculate_min_var_weights(self.testing_data, l1=l1, l2=l2)
            performance = self.calculate_performance(self.testing_data, weights, self.tscv_metric)
            if performance > best_performance:
                testing_optimal_l["l1"], testing_optimal_l["l2"] = l1, l2
                best_performance = performance
        self.testing_optimal_l = testing_optimal_l

    def calculate_testing_weights_optimal_l(self):
        if self.testing_optimal_l == {}:
            raise ValueError("You must calculate the optimal hyperparameters before calculating testing weights")
        self.testing_weights_optimal_l = self.calculate_min_var_weights(self.testing_data, **self.testing_optimal_l)