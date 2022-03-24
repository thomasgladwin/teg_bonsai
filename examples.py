import numpy as np
import teg_bonsai

#
# Standard test
#
nObs = 1500
nPred = 4
maxDepth = 4
peek_ahead_max_depth = 0
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
tree = teg_bonsai.Tree(X, y, maxDepth, alpha0, peek_ahead_max_depth=peek_ahead_max_depth, nSamples=nSamples, internal_cross_val=internal_cross_val, beta0_vec=beta0_vec)
Output = tree.build_tree()
# Output contains keys: tree, cost_complexity_criterion, best_peek_crit, raw_tree, CV_distr, null_distr, p
# Access via, e.g., Output['p']

#
# XOR problem requiring peek-ahead
#
nObs = 1500
nPred = 4
maxDepth = 4 # Max. number of splits
alpha0 = 0.5
peek_ahead_max_depth = 1
nSamples = 5
internal_cross_val = 1
beta0_vec = [0, np.inf]
X = np.round(np.random.random_sample(size=(nObs, nPred)), 2)
y = np.zeros(nObs)
LogicalInd = (X[:, 0] >= 0.5) & (X[:, 1] < 0.5)
y[LogicalInd] = 1
LogicalInd = (X[:, 0] < 0.5) & (X[:, 1] >= 0.5)
y[LogicalInd] = 1
y = y + 0.01 * np.random.randn(nObs)
y = np.round(y, 2)
tree = teg_bonsai.Tree(X, y, maxDepth, alpha0, peek_ahead_max_depth=peek_ahead_max_depth, nSamples=nSamples, internal_cross_val=internal_cross_val, beta0_vec=beta0_vec)
Output = tree.build_tree()
