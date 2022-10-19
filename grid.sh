#!/bin/bash
LC_NUMERIC=en_US.UTF-8


parallel -j 32 'python run.py                                \
        --objective={1}                                      \
        --dataset={2}                                        \
        --rebalancing_rule=DISABLE'                      ::: \
SR SoR                                                   ::: \
DJIA NASDAQ100 FTSE100                             


parallel -j 32 'python run.py                                \
        --objective={1}                                      \
        --dataset={2}                                        \
        --rebalancing_rule=PERIODIC'                     ::: \
SR SoR                                                   ::: \
DJIA NASDAQ100 FTSE100


# parallel -j 32 'python run.py                                   \
#         --objective={1}                                        \
#         --dataset={2}                                          \
#         --rebalancing_rule=EVENT                               \
#         --event_type={3}'                                  ::: \
# SR SoR                                                     ::: \
# DJIA NASDAQ100 FTSE100                                     ::: \
# "PORTFOLIO_RETURN_RF" "PORTFOLIO_RETURN_MU"


# parallel -j 32 'python run.py                                \
#         --objective={1}                                      \
#         --dataset={2}                                        \
#         --theta={3}                                          \
#         --rebalancing_rule=EVENT                             \
#         --event_type=DELTA_SUM_RETURN'                   ::: \
# SR SoR                                                   ::: \
# DJIA NASDAQ100 FTSE100                                   ::: \
# $(seq 0.00001 0.00075 0.3)