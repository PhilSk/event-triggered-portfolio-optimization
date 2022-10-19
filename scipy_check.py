import numpy as np
import pandas as pd
import time
import argparse
import cvxpy as cp
import warnings
from scipy.optimize import minimize
from pymongo import MongoClient
from utils import none_or_int


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-data', type=str, default='HSCI')
parser.add_argument('--data_partition', '-dp', type=float, default=1/2)
parser.add_argument('--rebalancing_rule', '-rr', type=str, default='PERIODIC', choices=['DISABLE', 'PERIODIC', 'EVENT'])
parser.add_argument('--event_type', '-et', type=str, default='DELTA_SUM_RETURN', choices=['DELTA_SUM_RETURN', 'DELTA_COV'])
parser.add_argument('--theta', '-t', type=float, default=None)
parser.add_argument('--period', '-p', type=int, default=None)
parser.add_argument('--return', '-r', type=str, default='SIMPLE')
parser.add_argument('--estimation_window', '-ew', type=none_or_int, default=None)
parser.add_argument('--note', '-n', type=str, default="17.02.22_sor")
parser.add_argument('--test', action='store_true')
args_parsed = parser.parse_args()
args = vars(args_parsed)

start_t = time.time()

client = MongoClient(host="localhost", port=27017)
db = client.cno
runs = db.runs

if args['data_partition'] == 1/3:
    ESTIMATION_START = 313
elif args['data_partition'] == 1/2:
    ESTIMATION_START = 469
elif args['data_partition'] == 2/3:
    ESTIMATION_START = 625
ESTIMATION_END = 938

ANNUAL_MULTIPLYER = np.sqrt(938/18)

data = pd.read_csv(f'data/wk_price_{args["dataset"]}.txt', sep="   ", header=None, engine='python').iloc[:ESTIMATION_END+1]
columns = [f'Asset {i}' for i in range(data.shape[1])]
data.columns = columns
index = [f'Week {i}' for i in range(data.shape[0])]
data.index = index

if args['return'] == 'SIMPLE':
    returns = data / data.shift(1) - 1.
elif args['return'] == 'LOG':
    returns = np.log(data/data.shift(1))

# [0] is used to convert DataFrame to Series
RISK_FREE = pd.read_csv(f'data/rf_00_17.txt', header=None, engine='python').T[0].iloc[:ESTIMATION_END+1]
RISK_FREE.index = index[:-1]

x = np.ones(data.shape[1])/data.shape[1]
portfolio = pd.DataFrame(np.nan, index=index, columns=['price', 'return'])
gamma_log = []
sr_gamma_log = []
sr_log = []
sor_log = []
x_arr = []
event_happened = np.nan
rebalance_count = 0


for i in range(data.shape[0])[ESTIMATION_START: ESTIMATION_END+1]:

    # Estimate porfolio performance
    # We choose portfolio today, but estimate its performance tomorrow.
    if i > ESTIMATION_START:
        portfolio['return'].iloc[i-1] = returns.iloc[i] @ x
        portfolio_returns = portfolio['return'].iloc[ESTIMATION_START:i]
        excess_portfolio_returns = portfolio_returns - RISK_FREE.iloc[ESTIMATION_START:i]
        R = excess_portfolio_returns.mean()
        sr_log.append(
            R/excess_portfolio_returns.std()
        )
        excess_portfolio_returns[excess_portfolio_returns > 0] = 0
        risk = np.sum(excess_portfolio_returns**2)/len(excess_portfolio_returns)
        if risk != 0:
            sor_log.append(R/np.sqrt(risk))

    # We have estimated last portfolio performance, exit
    if i == ESTIMATION_END: continue

    # Choose portfolio
    if args['rebalancing_rule'] == 'EVENT':
        if args['event_type'] == 'DELTA_SUM_RETURN':
            event_happened = np.abs(
                    (returns.iloc[i]*x).sum() - (returns.iloc[i-1]*x).sum()
                ) > args['theta']

    if (args['rebalancing_rule'] != 'DISABLE' and
            (
                (args['rebalancing_rule'] == 'EVENT' and event_happened) or
                (args['rebalancing_rule'] == 'PERIODIC' and i % args['period'] == 0)
            )
        ):

        r = returns[1:i+1].to_numpy()
        mu = returns[1:i+1].mean().to_numpy()
        # sigma = returns[1:i+1].cov().to_numpy()
        # _sigma = cp.atoms.affine.wraps.psd_wrap(sigma)

        rebalance_count += 1
        gamma_dynamics: List[float] = []
        gamma_prev = 0

        for t in range(100):
            time_s = time.time()

            numerator = mu.T@x - RISK_FREE.iloc[i]
            excess_x_returns = r@x - RISK_FREE.iloc[i]
            excess_x_returns[excess_x_returns > 0] = 0
            dsr = ((excess_x_returns)**2).mean()
            # dsr = excess_x_returns[excess_x_returns <= 0].var(ddof=1)
            # dsr = excess_x_returns.var(ddof=1)
            # dsr = x@sigma@x
            # assert np.allclose(dsr, dsr1)
            gamma = np.divide(numerator, dsr)

            print(f'{i}\t{t}\t{gamma:.5f}\t{numerator:.2f}\t{dsr}\n\n')

            if (t > 5):
                if (np.abs(gamma-gamma_prev) < 1e-3) or (len(set(np.around(gamma_dynamics[-5:], 3))) < 5):
                    break
            assert gamma > 0

            x_ = cp.Variable(shape=len(x), nonneg=True)
            x_.value = x
            excess_r = r @ x_ - RISK_FREE.iloc[i]
            excess_r = cp.atoms.elementwise.minimum.minimum(excess_r, 0)
            r3 = (excess_r)**2
            risk = cp.sum(r3)/r3.size

            prob = cp.Problem(cp.Minimize(0.5*(gamma**2)*risk - gamma*mu.T @ x_ - RISK_FREE.iloc[i]),
                        [cp.sum(x_) == 1])

            # prob = cp.Problem(cp.Minimize(0.5*(gamma**2)*cp.quad_form(x_, _sigma) - gamma*mu.T @ x_ - RISK_FREE.iloc[i]),
            #              [G @ x_ <= h,
            #               A @ x_ == b])
            prob.solve(solver=cp.OSQP, warm_start=True)

            # Perform solution testing
            def f(x):
                excess_x_returns = r@x - RISK_FREE.iloc[i]
                excess_x_returns[excess_x_returns >= 0] = 0
                dsr = np.sum(excess_x_returns**2)/len(excess_x_returns)
                return 0.5*(gamma**2)*dsr - gamma*mu.T@x - RISK_FREE.iloc[i]

            # optimizer = ng.optimizers.NGOpt(parametrization=len(x), budget=100)
            # optimizer.parametrization.register_cheap_constraint(lambda x: abs(sum(x) - 1.) <= 1e-2)
            # optimizer.parametrization.register_cheap_constraint(lambda x: np.all(x >= 0))
            # optimizer.register_callback("tell", ng.callbacks.ProgressBar())
            # recommendation = optimizer.minimize(f)  # best value
            # print(recommendation.value)


            cons = ({'type': 'eq', 'fun': lambda x:  abs(sum(x) - 1.)})
            bnds = [(0, None) for i in range(len(x))]
            res = minimize(f, 
                np.ones(data.shape[1])/data.shape[1],
                bounds=bnds,
                constraints=cons,
                options={'ftol': 1e-07, 'disp': True},
                )
            print(
                f'x 0 val: {f(np.ones(data.shape[1])/data.shape[1])}\n'
                f'x 1: {res.x}\n'
                f'Val 1: {f(res.x)}\n'
                f'x 2: {x_.value}\n'
                f'Val 2: {f(x_.value)}\n'
                f'x diff: {x_.value-res.x}\n'
            )
            print(res)
            exit()
            # val = f(x)

            # for pert_step in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            #     for x_i in range(len(x)):
            #         x_pert = x.copy()
            #         x_pert[x_i] += pert_step
            #         x_pert = x_pert/x_pert.sum()
            #         assert abs(x_pert.sum() - 1.0) < 1e-9
            #         assert np.all(x_pert >= 0)
            #         print(f(x_pert))
            #         assert f(x_pert) >= val
            

            gamma_dynamics.append(gamma)
            gamma_prev = gamma
            x = x_.value
        gamma_log.append(gamma_dynamics)

    else:
        sr_gamma_log.append((np.nan, np.nan))
    
    x_arr.append(x)
        
run_data = {
    "dataset": args["dataset"],
    "data_partition": args["data_partition"],
    "rebalancing_rule": args["rebalancing_rule"],
    "return": args["return"],
    "estimation_window": args["estimation_window"],
    "SR": sr_log,
    "SoR": sor_log,
    "portfolio_returns": portfolio['return'].tolist(),
    "gamma_log": gamma_log,
    "ESTIMATION_START": ESTIMATION_START,
    "rebalance_count": rebalance_count,
    "note": args["note"],
    "run_time": time.time() - start_t
}
if args["rebalancing_rule"] == 'EVENT':
    run_data['theta'] = args['theta']
    run_data['event_type'] = args['event_type']
if args["rebalancing_rule"] == 'PERIODIC':
    run_data['period'] = args['period']

if not args['test']:
    runs.insert_one(run_data)

np.save('results/x_arr.npy', np.array(x_arr))

run_sharpe_print = run_data['sr_log'][-1]
run_sortino_print = run_data['sor_log'][-1]

for key in ['portfolio_returns', 'gamma_log', 'sr_log', 'sor_log']:
    del run_data[key]

print(
         f'Sharpe ratio: {run_sharpe_print*ANNUAL_MULTIPLYER}\n'
         f'Sortino ratio: {run_sortino_print*ANNUAL_MULTIPLYER}\n'
         f'{rebalance_count} rebalancings.\n'
         f'{run_data["run_time"]:.2f} sec.\n'
         f'Cumulative return: {(1+portfolio_returns).prod()}\n'
    )
