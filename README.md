# teg_regression_tree

Practice project and for possible messing-around purposes: regression tree function, a la The Elements of Statistical Learning. Mostly base Python, just uses some basic numpy for convenience.

Example usage with sanity-check simulated data:

nObs = 2000
nPred = 6
maxDepth = 4 # Max. number of splits
alpha0 = 0.5
X = np.random.random_sample(size=(nObs, nPred))
y = 0.1 * np.random.random_sample(size=(nObs))
LogicalInd = (X[:, 1] > 0.8) & (X[:, 2] < 0.33)
y[LogicalInd] = 1 - (1 - y[LogicalInd]) * 0.25
tree0, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0)
print(tree0)
print(cost_complexity_criterion)

Which returns a nested list representing the expected tree, with a split on X1 around .8 and a second split for the right-branch on X2 at .33:

[[1, 0.8104412867733357], None, [[2, 0.3317265864314578], None, None]]

Each node is a triplet containing the feature-index and split-point, the left-branch, and the right-branch. Terminal nodes are given as None.

The cost-complexity criterion is also returned, for cross-validation of the alpha parameter.
