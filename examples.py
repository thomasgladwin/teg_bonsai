import numpy as np
import teg_bonsai

#
# Standard test
#
nObs = 1500
nPred = 4
maxDepth = 4
max_peek_depth = 0
nSamples = 5
internal_cross_val = 1
X = np.round(np.random.random_sample(size=(nObs, nPred)), 2)
y = np.zeros(nObs)
LogicalInd = (X[:, 0] >= 0.8) & (X[:, 2] < 0.33)
y[LogicalInd] = 1
y_true = y
y = y + 0.1 * np.random.random_sample(size=(nObs))
alpha0 = 0.5
beta0_vec = [0.01, np.inf]
Output = teg_bonsai.teg_regression_tree_peeks(X, y, maxDepth, alpha0, max_peek_depth, nSamples=nSamples, internal_cross_val=internal_cross_val, beta0_vec=beta0_vec)
# Output contains keys: tree0, cost_complexity_criterion, best_peek_crit, raw_tree, CV_distr, null_distr, p
# Access via, e.g., Output['p']

#
# XOR problem requiring peek-ahead
#
nObs = 1000
nPred = 4
maxDepth = 4 # Max. number of splits
alpha0 = 0.5
max_peek_depth = 1
X = np.round(np.random.random_sample(size=(nObs, nPred)), 2)
y = np.zeros(nObs)
LogicalInd = (X[:, 0] >= 0.5) & (X[:, 1] < 0.5)
y[LogicalInd] = 1
LogicalInd = (X[:, 0] < 0.5) & (X[:, 1] >= 0.5)
y[LogicalInd] = 1
y = y + 0.1 * np.random.randn(nObs)
y = np.round(y, 2)
Output = teg_bonsai.teg_regression_tree_peeks(X, y, maxDepth, alpha0, max_peek_depth)
