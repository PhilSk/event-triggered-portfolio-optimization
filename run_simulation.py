import numpy as np
import cvxopt
from pymongo import MongoClient


client = MongoClient(host="localhost", port=27017)
db = client.cno
runs = db.runs
runs.remove({})
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cvxopt.solvers.options['maxiters'] = 1000
cvxopt.solvers.options['show_progress'] = False

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
        # assert sigma is symmetric
        args = [cvxopt.matrix(P), cvxopt.matrix(q)]
        if G is not None:
            args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
            if A is not None:
                args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
        sol = cvxopt.solvers.qp(*args)
        if 'optimal' not in sol['status']:
            return None
        return np.array(sol['x']).reshape((P.shape[1],))

def run_simulation(theta=1,
                   beta=0.75,
                   period=50,
                   rebalancing_rule='EVENT',
                   dataset='HDAX'):

    data = np.genfromtxt(f'data/{dataset}', delimiter='   ')

    BUDGET = 1

    x = np.ones(data.shape[1])/data.shape[1]
    # x = np.zeros(data.shape[1])
    # x[2] = 1
    assets_num = (x*BUDGET)/data[0]

    # No possibility to compute return rate for the last time point, so ignore it
    return_arr = np.empty((data.shape[0]-1, data.shape[1]))
    return_arr[:] = np.nan
    portfolio_returns = np.empty(data.shape[0]-1)
    portfolio_returns[:] = np.nan
    events = []

    for i in range(len(return_arr)):
        # print(f'\nDay {i}')
        if i > 0:
            # Make this as in portfolio returns (i-1)
            return_arr[i] = (data[i] - data[i-1])/data[i-1]
            # portfolio instead of x - should calculate number of assets
            portfolio_price_prev = (data[i-1]*assets_num).sum()
            portfolio_price = (data[i]*assets_num).sum()
            # Move this line to the end of script
            portfolio_returns[i] = (portfolio_price - portfolio_price_prev)/portfolio_price_prev
      
            event_happened = np.abs(return_arr[i].sum() - return_arr[i-1].sum()) > theta
            events.append(int(event_happened))
            # print(event_happened)
            if (rebalancing_rule != 'DISABLE' and
                (
                    (rebalancing_rule == 'EVENT' and event_happened) or
                    (rebalancing_rule == 'PERIODIC' and i % period == 0)
                )):
                # [1:i+1] to ignore the first day and include day i
                mu = return_arr[1:i+1].mean(axis=0)
                sigma = np.cov(return_arr[1:i+1].T, ddof=1)
                A = np.ones((1, len(x)))
                b = np.ones(1)
                G = -np.eye(len(x))
                h = np.zeros(len(x))
                x = cvxopt_solve_qp(2*beta*sigma, (beta-1)*mu, A=A, b=b, G=G, h=h)
                assets_num = (x*portfolio_price)/data[i]
    risk_free = 0
    adj_portfolio_returns = portfolio_returns[1:] - risk_free
    sharpe_ratio = (adj_portfolio_returns.mean() * np.sqrt(252))/adj_portfolio_returns.std()
    # print(f'{experiment_name}\nSharpe Ratio: {sharpe_ratio}\n{np.array(events).sum()}\n')

    run_data = {
        "dataset": dataset,
        "rebalancing_rule": rebalancing_rule,
        "beta": beta,
        "sharpe_ratio": sharpe_ratio,
        "portfolio_returns": portfolio_returns[1:].tolist()
    }
    if rebalancing_rule == 'EVENT':
        run_data['theta'] = theta
    if rebalancing_rule == 'PERIODIC':
        run_data['period'] = period
    runs.insert_one(run_data)

for dataset in ['FTSE', 'HDAX', 'HSI', 'SP500']:
    run_simulation(rebalancing_rule='DISABLE', dataset=dataset)
    for beta in np.arange(0., 1.01, 0.05):
        beta = np.around(beta, decimals=2)
        print(beta)
        for i in range (50, 550, 100):
            run_simulation(beta=beta, rebalancing_rule='PERIODIC', period=i, dataset=dataset)
        run_simulation(beta=beta, theta=0.01, dataset=dataset)
        run_simulation(beta=beta, theta=0.05, dataset=dataset)
        run_simulation(beta=beta, theta=0.1, dataset=dataset)
        run_simulation(beta=beta, theta=0.15, dataset=dataset)
        run_simulation(beta=beta, theta=0.2, dataset=dataset)
        run_simulation(beta=beta, theta=0.25, dataset=dataset)
        run_simulation(beta=beta, theta=0.3, dataset=dataset)
        run_simulation(beta=beta, theta=0.5, dataset=dataset)
        for theta in range(1, 20):
            run_simulation(beta=beta, theta=theta, dataset=dataset)

