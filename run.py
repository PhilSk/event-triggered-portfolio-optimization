import numpy as np
import pandas as pd
import time
import argparse
import cvxpy as cp
import warnings
from pymongo import MongoClient
from utils import none_or_int


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-data', type=str, default='HSCI')
parser.add_argument('--data_partition', '-dp', type=float, default=1/2)
parser.add_argument('--objective', '-obj', type=str, default='SR', choices=['SR', 'SoR'])
parser.add_argument('--rebalancing_rule', '-rr', type=str, default='PERIODIC', choices=['DISABLE', 'PERIODIC', 'EVENT'])
parser.add_argument('--event_type', '-et', type=str, default='DELTA_SUM_RETURN', choices=['DELTA_SUM_RETURN', 'PORTFOLIO_RETURN_MU', 'PORTFOLIO_RETURN_RF', 'PORTFOLIO_RETURN_MU_NWEEKS', 'PORTFOLIO_RETURN_RF_NWEEKS'])
parser.add_argument('--theta', '-t', type=float, default=None)
parser.add_argument('--period', '-p', type=int, default=1)
parser.add_argument('--estimation_window', '-ew', type=none_or_int, default=None)
parser.add_argument('--note', '-n', type=str, default="17.08.22")
parser.add_argument('--test', action='store_true')
args_parsed = parser.parse_args()
args = vars(args_parsed)

start_t = time.time()

client = MongoClient(host="localhost", port=27017)
db = client.cno
runs = db.runs

duplicate_dict = {
    'dataset': args['dataset'],
    'objective': args['objective'],
    'note': args['note'],
    'data_partition': args['data_partition'],
    'rebalancing_rule': args['rebalancing_rule'],
    }
if args['rebalancing_rule'] == 'PERIODIC':
    duplicate_dict['period'] = args['period']
elif args['rebalancing_rule'] == 'EVENT':
    duplicate_dict['theta'] = args['theta']
doc = runs.find_one(duplicate_dict)
if doc is not None:
    raise ValueError('Duplicate found!')

if args["dataset"] in ['FTSE', 'HSCI', 'HDAX', 'SP500']:
    ANNUAL_MULTIPLIER = np.sqrt(938/18)
elif args["dataset"] == 'DJIA':
    ANNUAL_MULTIPLIER = np.sqrt(1362/26.25)
elif args["dataset"] == 'NASDAQ100':
    ANNUAL_MULTIPLIER = np.sqrt(596/11.5)
elif args["dataset"] == 'FTSE100':
    ANNUAL_MULTIPLIER = np.sqrt(716/13.83)
elif args["dataset"] == 'SP500_new':
    ANNUAL_MULTIPLIER = np.sqrt(594/11.5)
elif args["dataset"] == 'NASDAQComp':
        ANNUAL_MULTIPLIER = np.sqrt(684/13.25)

if args["dataset"] in ['FTSE', 'HSCI', 'HDAX', 'SP500']:
    ESTIMATION_START = 469
    ESTIMATION_END = 938

    data = pd.read_csv(f'data/wk_price_{args["dataset"]}.txt', sep="   ", header=None, engine='python').iloc[:ESTIMATION_END+1]
    columns = [f'Asset {i}' for i in range(data.shape[1])]
    data.columns = columns
    index = [f'Week {i}' for i in range(data.shape[0])]
    data.index = index
    STOCKS_NUM = data.shape[1]
    returns = data / data.shift(1) - 1.
    RISK_FREE = pd.read_csv(f'data/rf_00_17.txt', header=None, engine='python').T[0].iloc[:ESTIMATION_END+1]
    RISK_FREE.index = index[:-1]

else:
    # [0] is used to convert DataFrame to Series
    RISK_FREE = pd.read_csv(f'data/rf_{args["dataset"]}.txt', header=None, engine='python').T[0]
    T = len(RISK_FREE)
    if T % 2 == 1:
        T -= 1
    ESTIMATION_START = T//2
    ESTIMATION_END = T
    index = [f'Week {i}' for i in range(T+1)]
    RISK_FREE = RISK_FREE.iloc[:ESTIMATION_END]
    RISK_FREE.index = index[:-1]
    returns = pd.read_excel(f'data/{args["dataset"]}.xlsx', index_col=None, header=None).iloc[:ESTIMATION_END]
    returns = pd.concat([pd.DataFrame([np.NaN]), returns], ignore_index=True)
    
    columns = [f'Asset {i}' for i in range(returns.shape[1])]
    returns.columns = columns
    returns.index = index

# print(ESTIMATION_START)
# print(ESTIMATION_END)
# print(RISK_FREE)
# print(returns)

x = np.ones(returns.shape[1])/returns.shape[1]
portfolio = pd.DataFrame(np.nan, index=index, columns=['price', 'return'])
gamma_log = []
metric_logs = []
metric_ex_ante_logs = []
x_arr = []
event_happened = np.nan
rebalance_count = 0

week_performance_coutner = 0
bool_history = []


for i in range(returns.shape[0])[ESTIMATION_START: ESTIMATION_END+1]:

    r = returns[1:i+1].to_numpy()
    mu = returns[1:i+1].mean().to_numpy()
    sigma = returns[1:i+1].cov().to_numpy()
    _sigma = cp.atoms.affine.wraps.psd_wrap(sigma)

    # Estimate porfolio performance
    # We choose portfolio today, but estimate its performance tomorrow.
    if i > ESTIMATION_START:
        portfolio['return'].iloc[i-1] = returns.iloc[i] @ x
        rr = returns.iloc[i]
        portfolio_returns = portfolio['return'].iloc[ESTIMATION_START:i]
        excess_portfolio_returns = portfolio_returns - RISK_FREE.iloc[ESTIMATION_START:i]
        R = excess_portfolio_returns.mean()
        if args['objective'] == 'SR':
            metric_logs.append(
                R/excess_portfolio_returns.std()
            )
        elif args['objective'] == 'SoR':
            excess_portfolio_returns[excess_portfolio_returns > 0] = 0
            epr_sqr = excess_portfolio_returns**2
            risk = np.sum(excess_portfolio_returns**2)/len(excess_portfolio_returns)
            if risk != 0:
                sor = R/np.sqrt(risk)
                metric_logs.append(sor)
            else:
                metric_logs.append(0)

        numerator = mu.T@x - RISK_FREE.iloc[i-1]
        if args['objective'] == 'SR':
            denominator = x@sigma@x
        elif args['objective'] == 'SoR':
            excess_x_returns = r@x - RISK_FREE.iloc[:i]
            excess_x_returns[excess_x_returns > 0] = 0
            exr_sqr = (excess_x_returns)**2
            denominator = exr_sqr.mean()
        metric_ex_ante_logs.append(np.divide(numerator, np.sqrt(denominator)))
    
    # We have estimated last portfolio performance, exit
    if i == ESTIMATION_END: continue

    # Choose portfolio
    if args['rebalancing_rule'] == 'EVENT':
        if args['event_type'] == 'DELTA_SUM_RETURN':
            event_happened = np.abs(
                    (returns.iloc[i]*x).sum() - (returns.iloc[i-1]*x).sum()
                ) > args['theta']
        elif args['event_type'] == 'PORTFOLIO_RETURN_MU':
            if (returns.iloc[i]*x).sum() < mu.mean():
                event_happened = True
            else:
                event_happened = False
        elif args['event_type'] == 'PORTFOLIO_RETURN_RF':
            if (returns.iloc[i]*x).sum() < RISK_FREE.iloc[i]:
                event_happened = True
            else:
                event_happened = False
        elif args['event_type'] == 'PORTFOLIO_RETURN_MU_NWEEKS':
            bool_history.append((returns.iloc[i]*x).sum() < mu.mean())
            if all(bool_history[-int(args['theta']):]):
                event_happened = False
            else:
                event_happened = True
        elif args['event_type'] == 'PORTFOLIO_RETURN_RF_NWEEKS':
            bool_history.append((returns.iloc[i]*x).sum() < RISK_FREE.iloc[i])
            if all(bool_history[-int(args['theta']):]):
                event_happened = False
            else:
                event_happened = True

    if (args['rebalancing_rule'] != 'DISABLE' and
            (
                (args['rebalancing_rule'] == 'EVENT' and event_happened) or
                (args['rebalancing_rule'] == 'PERIODIC' and i % args['period'] == 0)
            )
        ):

        rebalance_count += 1
        gamma_dynamics = []
        gamma_prev = 0

        x = np.ones(returns.shape[1])/returns.shape[1]
        for t in range(1000000):
            time_s = time.time()

            numerator = mu.T@x - RISK_FREE.iloc[i]
            if args['objective'] == 'SR':
                denominator = x@sigma@x
            elif args['objective'] == 'SoR':
                excess_x_returns = r@x - RISK_FREE.iloc[:i]
                excess_x_returns[excess_x_returns > 0] = 0
                exr_sqr = (excess_x_returns)**2
                denominator = exr_sqr.mean()
            gamma = np.divide(numerator, denominator)
            assert gamma > 0

            if args['test']:
                print(f'{i}\t{t}\tgamma: {gamma:.5f}\n')

            if abs(gamma-gamma_prev) < 1e-6: break

            x_ = cp.Variable(shape=len(x), nonneg=True)
            x_.value = x

            A = np.ones((len(x),))
            b = np.array([1.])
            G = np.diag(-np.ones(len(x)))
            h = np.zeros(len(x))

            if args['objective'] == 'SR':
                prob = cp.Problem(cp.Minimize(0.5*(gamma**2)*cp.quad_form(x_, _sigma) - gamma*mu.T @ x_ - RISK_FREE.iloc[i]),
                        [cp.sum(x_) == 1])
            elif args['objective'] == 'SoR':
                rf = RISK_FREE.iloc[:i]
                rff = RISK_FREE.iloc[i]
                excess_r = r @ x_ - RISK_FREE.iloc[i]
                excess_r = cp.atoms.elementwise.minimum.minimum(excess_r, 0)
                r3 = (excess_r)**2
                dsr = r3.value
                risk = cp.sum(r3)/r3.size
                a = risk.value
                prob = cp.Problem(cp.Minimize((gamma**2)/2 * risk - gamma*(mu @ x_ - RISK_FREE.iloc[i])),
                            [cp.sum(x_) == 1])
            prob.solve(solver=cp.OSQP, warm_start=True, max_iter=100000, eps_abs=1e-8, eps_rel=1e-8)
            x = x_.value

            gamma_dynamics.append(gamma)
            gamma_prev = gamma
        gamma_log.append(gamma_dynamics)
    x_arr.append(x.tolist())

portfolio_returns = np.array(portfolio_returns)
portfolio_returns = portfolio['return'] + 1
cumulative_product = np.nancumprod(portfolio_returns)
annualized_returns = np.power(cumulative_product, 1/9) - 1

run_data = {
    "objective": args["objective"],
    "dataset": args["dataset"],
    "data_partition": args["data_partition"],
    "rebalancing_rule": args["rebalancing_rule"],
    "estimation_window": args["estimation_window"],
    "metric_ex_ante_seq": metric_ex_ante_logs,
    "metric_seq": metric_logs,
    "metric": metric_logs[-1],
    "portfolios": x_arr,
    "portfolio_returns": portfolio['return'].tolist(),
    "annualized_returns": annualized_returns.tolist(),
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

run_metric_print = run_data['metric_ex_ante_seq'][-1]

for key in ['portfolio_returns', 'gamma_log', 'metric_seq', 'metric_ex_ante_seq', 'annualized_returns', 'portfolios']:
    del run_data[key]

print(
        f'{args["dataset"]} {args["objective"]} \n'
        f'{run_data}'
    )
print(
        f'{args["objective"]}: {run_data["metric"]*ANNUAL_MULTIPLIER}\n'
        f'{rebalance_count} rebalancings.\n'
        f'Annualized return: {annualized_returns[-1]}.\n'
        f'{args["objective"]} ex-ante: {run_metric_print*ANNUAL_MULTIPLIER}.\n'
        f'{run_data["run_time"]:.2f} sec.\n'
    )
