import numpy as np
import pandas as pd


for dataset in ['DJIA', 'NASDAQ100', 'FTSE100', 'SP500_new']:
    print(dataset)

    ESTIMATION_START = 469
    ESTIMATION_END = 938
    ANNUAL_MULTIPLIER = np.sqrt(938/18)

    if dataset in ['FTSE', 'HSCI', 'HDAX', 'SP500']:
        data = pd.read_csv(f'data/wk_price_{dataset}.txt', sep="   ", header=None, engine='python').iloc[:ESTIMATION_END+1]
        RISK_FREE = pd.read_csv(f'data/rf_00_17.txt', header=None, engine='python').T[0]
        RISK_FREE = RISK_FREE.iloc[ESTIMATION_START:ESTIMATION_END+1]
        returns = (data / data.shift(1) - 1.)
    else:
        RISK_FREE = pd.read_csv(f'data/rf_{dataset}.txt', header=None, engine='python').T[0]
        T = len(RISK_FREE)
        if T % 2 == 1:
            T -= 1
        ESTIMATION_START = T//2
        ESTIMATION_END = T
        RISK_FREE = RISK_FREE.iloc[:ESTIMATION_END]
        RISK_FREE = RISK_FREE.iloc[ESTIMATION_START:ESTIMATION_END+1]
        returns = pd.read_excel(f'data/{dataset}.xlsx', index_col=None, header=None).iloc[:ESTIMATION_END]
        returns = pd.concat([pd.DataFrame([np.NaN]), returns], ignore_index=True)

    STOCKS_NUM = returns.shape[1]

    x = np.ones(STOCKS_NUM)/STOCKS_NUM
    portfolio_returns = (
        returns[ESTIMATION_START:].shift(-1)@x
        ).loc[:ESTIMATION_END-1]

    excess_portfolio_returns = portfolio_returns - RISK_FREE
    R = excess_portfolio_returns.mean()

    # Sharpe
    SR = ANNUAL_MULTIPLIER*R/np.sqrt(
        np.sum((excess_portfolio_returns - R)**2)/(len(portfolio_returns)-1)
    )
    print(f'Sharpe: {SR:.10f}')

    # Sortino
    excess_portfolio_returns_dsr = excess_portfolio_returns.copy()
    excess_portfolio_returns_dsr[excess_portfolio_returns_dsr > 0] = 0
    DSR = np.sum(excess_portfolio_returns_dsr**2)/len(excess_portfolio_returns_dsr)
    print(f'Sortino: {ANNUAL_MULTIPLIER*R/np.sqrt(DSR):.10f}\n')