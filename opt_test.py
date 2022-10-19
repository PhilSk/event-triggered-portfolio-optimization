import numpy as np
import pandas as pd
import cvxpy as cp
import time
from utils import none_or_int


args = {
    'dataset': 'HSI',
    'data_partition': 1/3,
    'rebalancing_rule': 'PERIODIC',
    'period': 1,
    'return': 'SIMPLE',
    'estimation_window': None,
}

if args['data_partition'] == 1/3:
    ESTIMATION_START = 313
elif args['data_partition'] == 1/2:
    ESTIMATION_START = 469
elif args['data_partition'] == 2/3:
    ESTIMATION_START = 625
ESTIMATION_END = 938

ANNUAL_MULTIPLYER = np.sqrt(52.1111111)

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
print(len(RISK_FREE))
RISK_FREE.index = index[:-1]

x = np.ones(data.shape[1])/data.shape[1]

i = 683
time_s = time.time()

# Check mean and cov computations are correct
# mu = mu_array.iloc[i].to_numpy().copy()
# mu1 = returns[:i+1].mean().to_numpy()
# assert np.all(np.abs(mu - mu1) < 1e-5)
# sigma = sigma_array.loc[f"Week {i}"].to_numpy().copy()
# sigma1 = returns[:i+1].cov().to_numpy()
# X = returns[1:i+1].to_numpy()
# X -= X.mean(axis=0) 
# fact = X.shape[0] - 1 
# sigma_by_hand = np.dot(X.T, X.conj()) / fact
# assert np.all(np.abs(sigma - sigma1) < 1e-5)
# assert np.all(np.abs(sigma - sigma_by_hand) < 1e-5)


mu = returns[:i+1].mean().to_numpy()
sigma = returns[:i+1].cov().to_numpy()
_sigma = cp.atoms.affine.wraps.psd_wrap(sigma)
# Optimization
gamma_prev = 0

SOLVER = 'ECOS'

r = returns[1:i+1].to_numpy()
for t in range(10):
    numerator = mu.T@x-RISK_FREE.iloc[i]
    # denominator = x.T@sigma@x
    excess_x_returns = r@x - RISK_FREE.iloc[i]
    # excess_x_returns[excess_x_returns >= 0] = 0
    excess_x_returns[excess_x_returns > 0] = 0
    dsr = np.sum(excess_x_returns**2)/len(excess_x_returns)
    gamma = np.divide(numerator, dsr)
    if (t != 0) and np.abs(gamma-gamma_prev) < 1e-5:
        break
    # assert gamma > 0

    print(f'{i}\t{t}\t{gamma:.2f}\t{numerator:.2f}\t{dsr}\n\n')

    A = np.ones((len(x),))
    b = np.array([1.])
    G = np.diag(-np.ones(len(x)))
    h = np.zeros(len(x))

    x_ = cp.Variable(shape=len(x), nonneg=True)
    x_.value = x
    # x_broad = np.ones((len(x), 1)) @ cp.reshape(x_, (1,len(x)))
    # risk = cp.sum(cp.atoms.elementwise.minimum.minimum(
    #         cp.multiply(cp.multiply(x_broad.T, sigma), x_broad), 
    #         0))
    # risk = cp.sum(cp.multiply(x_broad.T, sigma))
    # risk = cp.sum(cp.multiply(cp.multiply(x_broad.T, sigma), x_broad))

    r1 = r @ x_ - RISK_FREE.iloc[i]
    r2 = cp.atoms.elementwise.minimum.minimum(r1, 0)
    r3= r2**2
    risk = cp.sum(r3)/len(excess_x_returns)

    prob = cp.Problem(cp.Minimize(0.5*(gamma**2)*risk - gamma*mu.T @ x_ - RISK_FREE.iloc[i]),
                 [G @ x_ <= h,
                  A @ x_ == b])

    # prob = cp.Problem(cp.Minimize(0.5*(gamma**2)*cp.quad_form(x_, _sigma) - gamma*mu.T @ x_ - RISK_FREE.iloc[i]),
    #              [G @ x_ <= h,
    #               A @ x_ == b])

    prob.solve(solver=getattr(cp, SOLVER))
    x = x_.value

    gamma_prev = gamma

# Test if sigma is positive-definite and symmetric
# M = (gamma**2)*sigma
# print(np.all(np.linalg.eigvals(M) > 0), '||>')
# print(np.all(np.sum(M-M.T) == 0), '|>')

# Check if x is really a solution to obj
# if i == COLLECT_DATA_WEEKS_NUM + 50:

# print(G@x <= h)
# print(x.sum())
# print(x)

print(f'{time.time() - time_s:.2f} sec.')

#TODO CHECK IF THIS MAXIMIZES SoR
def f(x):
    excess_x_returns = r@x - RISK_FREE.iloc[i]
    excess_x_returns[excess_x_returns >= 0] = 0
    dsr = np.sum(excess_x_returns**2)/len(excess_x_returns)
    return 0.5*(gamma**2)*dsr - gamma*mu@x - RISK_FREE.iloc[i]

val = f(x)
print(val)
for iteration in range(len(x)):
    x_pert = x.copy()
    x_pert[iteration] += 0.001
    x_pert = x_pert/x_pert.sum()
    assert np.abs(x_pert.sum() - 1.0) < 1e-5
    assert np.all(x_pert >= 0)
    assert f(x_pert) >= val
    # print(f'{f(x_pert)>=val}      {f(x_pert)-val}')

np.save(f'data/{SOLVER}.npy', x_.value)
x_.value