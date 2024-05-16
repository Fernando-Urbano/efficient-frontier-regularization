# Exploring Hyperparameter Spaces: A Comprehensive Study of Ridge Regularization in Mean-Variance Optimization

**Fernando Urbano**

Detailed report available in [reports/report.pdf](reports/report.pdf)

## Code
### Create Virtual Environment
Open your terminal and run the following command:
```bash
conda create --name markowitz python=3.12.2
```

Deactivate any existing virtual environment:
```bash
conda deactivate
```

Install packages for the virtual environment:
```bash
pip install -r requirements.txt
```

### Run the following scripts in order
Downloads data from the Kenneth French website and saves into data:
```bash
python data_download.py
```

Create the simulations for the portfolio (*3 to 4 days to complete in a normal computer*):
```bash
python simulations_creation.py
```

Integrate the results into time-series:
```bash
python simulation_integration.py
```

Divide the time-series into annual portions and calculate the yearly performance metrics:
```bash
python simulation_yearly_division.py
```

Create analysis by running the following jupyter notebooks:
```bash
jupyter nbconvert --to notebook --execute simulations_analysis_anova.ipynb
```

```bash
jupyter nbconvert --to notebook --execute simulations_analysis_distributions.ipynb
```

```bash
jupyter nbconvert --to notebook --execute simulations_analysis_time_series.ipynb
```

## Introduction

Markovitz's construction of the Efficient Frontier in 1952 is among the most significant improvements in quantitative finance and set the start of the Modern Portfolio Theory [Markowitz, 1952].

The idea behind the Efficient Frontier provided an algorithm to choose optimal weights for the assets in the portfolios of hedge funds, banks, and other financial institutions.

It is part of the mean-variance optimization framework and served as the foundation for the CAPM theory of Willian Sharpe.

The original algorithm was highly praised for its simplicity accompanied, nonetheless, by a powerful conclusion: portfolio optimization depends on expected returns and risk (measured by variance) and investors aim to maximize expected returns given their levels of risk aversion. With mean-variance optimization, investors can transform their views into investment bets.

With time, the original mean-variance optimization started facing criticism due to estimation error and extremely aggressive allocation as a consequence of mathematical instability [Schmid, 2018]. Nowadays, improvements in the field of Machine Learning found paths to mitigate the problem through the use of Regularization (L1 and L2) [Britten, 2013] and Resampling (Monte Carlo and Bootstrapping) [Bruder, 2013].

While the original algorithm provides a concise set of hyperparameters, modern applications with regularization techniques require extensive hyperparameter tuning and a gigantic number of possible combinations on how to address the problem. Practitioners often choose one possible set among the vast poll, given limited time to train and analyze results.

The goal of our paper is to dive deeper into the tuning of hyperparameters for RIDGE regularization, tirelessly testing possible combinations of different parameters to arrive at general guidelines on how to approach the problem and which sets generate more favorable results. We aim to provide a comprehensive study of the hyperparameter space of the RMVP, exploring the impact of the number of assets, training size, days until rebalancing (viewed as testing size), number of time-series cross-validation, and cross-validation size.

## Research Methodology

### Mean-Variance Optimization

The optimization process is based on finding the weights for assets in a given portfolio that maximize the expected return of the portfolio given a level of risk or minimize the risk of a portfolio given a level of expected return:

$$
\max_{w} \quad \mu^{T} w \quad \quad
\text{s.t.} \quad w^{T} \Sigma w \leq \sigma^{2}, \quad
\sum_{i=1}^{n} w_{i} = 1
$$

In the equation, $w$ is the vector of weights, $\mu$ is the vector of expected returns, $\Sigma$ is the covariance matrix of the returns, and $\sigma^{2}$ is the tolerated level of risk. The first constraint ensures that the risk of the portfolio is below a certain threshold, and the second constraint ensures that the investor uses 100% (not more or less) of their capital in the allocation.

The Efficient Frontier gives the best allocation for every given risk (variance) level. The curved shape is a consequence of diversification: less than perfect correlation between assets allows for a reduction in the overall risk of the portfolio.

The Efficient Frontier has two points worth of special attention: the Global Minimum Variance Portfolio (GMV), the leftmost point, and the Tangency Portfolio.

### Global Minimum Variance Portfolio

The Global Minimum Variance Portfolio (GMV) is the portfolio with the lowest possible variance.

Given the convex nature of the optimization problem, it is possible to find the $w_{\text{GMV}}$ with a closed form solution:

$$
w_{\text{GMV}} = \argmin_{w} \quad w^{T} \Sigma w \quad \quad
\text{s.t.} \sum_{i=1}^{n} w_{i} = 1
$$

$$
w_{\text{GMV}} = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^{T} \Sigma^{-1} \mathbf{1}}
$$

In the equation, $\mathbf{1}$ is a vector of ones and $\Sigma^{-1}$ is the inverse of the covariance matrix of returns.

### Tangency Portfolio

The Tangency Portfolio is the portfolio that maximizes the Sharpe Ratio [Sharpe, 1964], a measure of risk-adjusted return, defined as the ratio of the excess return of the portfolio over the risk-free rate to the standard deviation of the portfolio (square root of the variance):

$$
\text{Sharpe Ratio (SR)} = \frac{\tilde{\mu}^{T} w}{\sqrt{w^{T} \Sigma w}}
$$

Where $\tilde{\mu} = \mu - r_f \mathbf{1}$ is the vector of excess returns, with $r_f$ being the risk-free rate.

Again, given the convex nature of the optimization problem, it is possible to find the $w_{\text{TAN}}$ with a closed form solution:

$$
w_{\text{TAN}} = \argmax_{w} \quad \frac{\mu^{T} w - r_f}{\sqrt{w^{T} \Sigma w}} \quad \quad
\text{s.t.} \sum_{i=1}^{n} w_{i} = 1
$$

$$
w_{\text{TAN}} = \frac{\Sigma^{-1} \tilde{\mu}}{\mathbf{1}^{T} \Sigma^{-1} \tilde{\mu}}
$$

### Instability and Unpredictability

The two formulas give us the optimal weights for the given levels of risk. Any other point in the efficient frontier can be obtained by a linear combination of the GMV and the Tangency Portfolio.

$$
w_{\text{optimal}} = \alpha w_{\text{GMV}} + (1 - \alpha) w_{\text{TAN}} \quad \quad \alpha > 0
$$

However, the process of estimating expected returns and the covariance matrix from historical data encounters significant challenges. These estimates are inherently prone to inaccuracies. Specifically, returns exhibit minimal or no autocorrelation, resulting in a high level of unpredictability. While the covariance matrix's estimates are marginally more reliable, the high dimensionality of portfolio optimization exacerbates the impact of even minor estimation errors, leading to substantial discrepancies in the optimization outcomes and final portfolio allocation.

More specifically, the errors are magnified by the $\Sigma^{-1}$ term in the formulas. The inverse of the covariance matrix is highly sensitive to small changes in the estimates, given its high condition number (ratio of the largest to the smallest eigenvalue of the matrix), which serves as a measure of how much the output of the matrix changes due to small changes in the input. This results in extreme optimal weights, with large allocations in few assets (long and short).

In simpler words, a small difference between the historical and future excess returns ($\tilde{\mu}$) or the covariance matrix ($\Sigma$) leads to a large difference in the optimal weights between the training period and the future period.

### Ridge Regularization

Ridge Regularization is a technique used to mitigate the problem of instability in the optimization process, by adding a penalty term. The optimized point of minimum variance with RIDGE is called the Regularized Minimum Variance Portfolio (RMVP).

$$
w_{\text{RMVP}} = \argmin_{w} \quad w^{T} \Sigma w + \lambda \|w\|^{2} \quad \quad
\text{s.t.} \sum_{i=1}^{n} w_{i} = 1
$$

The $\lambda$ term is the hyperparameter that controls the trade-off between the variance of the portfolio and the penalty term. As in regression problems, Ridge has the advantage of providing a closed-form solution:

$$
w_{\text{RMVP}} = \argmin_{w} \quad w^{T} \Sigma w + \lambda w^T w
$$

$$
w^{T} \Sigma w + w^T \lambda I w
$$

$$
w^{T} (\Sigma + \lambda I) w
$$

The problem is now again a quadratic form, and the solution is:

$$
w_{\text{RMVP}} = \frac{(\Sigma + \lambda I)^{-1} \mathbf{1}}{\mathbf{1}^{T} (\Sigma + \lambda I)^{-1} \mathbf{1}}
$$

The algorithm reduces the extreme allocations in few assets and decreases the sensitivity of the optimization process to small changes in the estimates.

In practice, the solution vector $w_{\text{RMVP}}$ penalty value is defined by how much the individual weights differ from the equal weight allocation of $1/n$, where $n$ is the number of assets in the portfolio. Namely, The penalty term in RIDGE is a function of the Euclidean Norm of the vector of weights, which is minimized when $w_i = 1/n$ for all $i$.

### Methodology

In this paper, we explore the hyperparameter space of the RMVP. Using the point of Minimum Variance instead of the Tangency Portfolio is viewed as a better approach, given that historical returns are not good predictors of future returns and we intend to remain agnostic about other prediction models' results, given that the focus should remain in the exploration of the hyperparameters of the mean-variance optimization [Campbell, 2008].

In our tests, we use the conventional format of the regularized optimization problem. Given the selected assets and training data, we use time-series cross-validation to find the optimal $\lambda$ for the RMVP selecting based on the Sharpe Ratio performance. After choosing the optimal $\lambda$, we calculate $w_{\text{RMVP}}$ in the training sample and see their performance in the testing sample.

The hyperparameter space tested is defined by the following parameters:

- Choice of number of assets $n$: 5, 10, 12, 17, 30.
- Choice of training size $T$ in days: 63, 126, 252, 504.
- Choice of testing size $t$ in days (how long the portfolio will run until a new optimization is calculated): 5, 10, 21, 42, 126, 252.
- Choice of number of time-series cross-validation $n_{cv}$: 1 to 8.
- Choice of cross-validation size $t_{cv}$ as a percentage of the training size: 50%, 75%, 100%, 125%, 150%.

The $\lambda$'s tested are always in the range of $[0, 3]$ with a step of $0.10$ (where $\lambda = 0$ is the GMV). Before running the simulations, we tested the range in which the optimal $\lambda$'s are found, and limit the search until $3$ given that the optimal is always below this threshold. Convex optimization approaches were also tested for $\lambda$ with gradient descent, but led to unsatisfactory results, given the non-convex nature of the loss function with respect to $\lambda$.

Each of 5,600 combinations in the simulation runs from January 1st 2000 to December 31st 2022, totaling 4.5 million calculations. As a result, we have 5,600 testing portfolios, each with 23 years of data.

The simulations provide answers to questions such as:

- How does the optimal training size $T$ change as a function of the number of assets $n$ and testing size $t$?
- What is the optimal number of time-series cross-validation ($n_{cv}$)?
- What is the optimal size of the cross-validation ($t_{cv}$)?
- What is the optimal training window for the RMVP?
- How often should the portfolio be rebalanced?
- How do moments of financial stress affect the previous questions?

The testing portfolios for each of the combinations are aggregated yearly (123,200 data points), allowing for comparison of Sharpe Ratio, volatility, and returns between the portfolios. The results are analyzed using multivariate ANOVA and Tukey HSD pairwise tests.

## Data

The data is collected from the [Kenneth French Website](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). The website provides daily industry returns in tables dividing the industries in 5, 10, 12, 17, 30, 38, 48, 49 assets from 1926 to 2024 (updated regularly). Each of those tables is used as one of the possible $n$ (number of assets) in the portfolio. The risk-free rate is also provided on the website, inside the "Fama/French 3 Factors" table, and used to subtract the returns.

## Results

Every factor is significant in explaining yearly annualized Sharpe Ratios in the ANOVA without interactions. As expected, the results are highly dependent on the number of assets and year since those factors dictate which returns are being used.

Training Window and Number of Time-Series Cross-Validation are the factors among the ones that can be optimized by practitioners, which represent the most significant impact on the Sharpe Ratio.

The comparison between the smallest training window tested, 63 days (3 months), and the largest two training windows tested, 252 and 504 days (1 and 2 years), show a significant difference in every Sharpe ratio favorable to the larger training windows.

Using 1 time-series cross-validation fold shows a significantly higher average performance when compared to any other number of folds. Nonetheless, the result is accompanied by a higher standard deviation of the forecasted Sharpe Ratio, which showcases a trade-off between performance and stability.

The pairwise comparison of year is omitted since it does not provide useful information, given that the results are highly dependent on the returns of the assets in the year, and the comparison is not meaningful.

The pairwise comparison is not significant for testing sample (days until rebalancing) and cross-validation size as a percentage of training days. The absence of significance for testing sample shows practitioners that, unless changes in expected return or risk are provided, continuous rebalancing is not favorable accounting for transaction costs.

In the Appendix, detailed results are provided by year and in aggregate of the Sharpe Ratio statistics.
