# teg_regression_tree

Practice project and for messing-around purposes: regression tree function, a la The Elements of Statistical Learning, with an added "two-step" variation to deal with "XOR" patterns. That is: for each possible split, all possible immediately following splits (for a given set of quantiles for the second split to keep computation time down; the median would seem to be the main one) are used to evaluate the first split. This avoids the usual greedy algorithm's trap of not being able to recognize a split that is only good in combination with a subsequent split.

It's mostly made in base Python, just uses some basic numpy for convenience.

Example usage with sanity-check simulated data:
```
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
```
Which returns a nested list representing the expected tree, with a split on X1 around .8 and a second split for the right-branch on X2 at .33:

`[[1, 0.8104412867733357], None, [[2, 0.3317265864314578], None, None]]`

Each node is a triplet containing the feature-index and split-point of the node, the left-branch, and the right-branch. Terminal nodes are given as None.

The cost-complexity criterion is also returned, for cross-validation of the alpha parameter.

An "XOR" example where the traditional tree fails but the current implementation can deal with is as follows:

```
nObs = 2000
nPred = 6
maxDepth = 4 # Max. number of splits
alpha0 = 0.5
X = np.random.random_sample(size=(nObs, nPred))
y = 0.1 * np.random.random_sample(size=(nObs))
LogicalInd = (X[:, 1] > 0.5) & (X[:, 2] < 0.5)
y[LogicalInd] = 1 - (1 - y[LogicalInd]) * 0.25
LogicalInd = (X[:, 1] < 0.5) & (X[:, 2] > 0.5)
y[LogicalInd] = 1 - (1 - y[LogicalInd]) * 0.25
# Traditional greedy tree
tree0, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0, twostep=0)
print(tree0)
print(cost_complexity_criterion)
# Two-step tree
tree0, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0)
print(tree0)
print(cost_complexity_criterion)
```

This would (after a minute or so, it's not fast...) print out the correct solution, splitting first on X1 and then on each branch on X2:

```
 [1, 0.501237639249083]
	 [2, 0.5021007129003712]
	 [2, 0.5001143296214183]
```
