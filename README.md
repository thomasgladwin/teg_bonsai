# teg_regression_tree

Practice project and for messing-around purposes: regression tree function, a la The Elements of Statistical Learning, with an added "peek ahead" or "two-step" variation to deal with XOR patterns. That is: for each possible split, all possible immediately following splits (currently: for a limited set of quantiles for the second split to keep computation time down, trading off the resolution of the peek-ahead) are used to evaluate the first split. This avoids the usual greedy algorithm's trap of not being able to recognize a split that is only good in combination with a subsequent split.

It's mostly made in base Python, just uses some basic numpy for convenience.

Example usage with sanity-check simulated data:
```
nObs = 2000
nPred = 6
maxDepth = 4 # Max. number of splits
alpha0 = 0.5
X = np.random.random_sample(size=(nObs, nPred))
y = 0.1 * np.random.random_sample(size=(nObs))
LogicalInd = (X[:, 1] > 0.8) & (X[:, 2] < 0.33) & (X[:, 4] < 0.5)
y[LogicalInd] = 1 - (1 - y[LogicalInd]) * 0.25
# Traditional greedy tree
tree0, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0, twostep=0)
```
The function prints out the tree as follows, with low and high branches starting on different lines with the same indentation, with the predicted value shown for terminal nodes:

```
[1, 0.8125486774415731]
	 terminal node:  0.05188583804970138
	 [2, 0.33236530087901717]
		 [4, 0.5128665191583686]
			 terminal node:  0.7623787750856714
			 terminal node:  0.04541482950659097
		 terminal node:  0.051131886885080774
```

The function also returns a nested list representing the tree:

`[[1, 0.8125486774415731], 0.05188583804970138, [[2, 0.33236530087901717], [[4, 0.5128665191583686], 0.7623787750856714, 0.04541482950659097], 0.051131886885080774]]`

Each non-terminal node is a triplet containing a 2-element list with the feature-index and split-point of the node, the left-branch, and the right-branch. Terminal nodes are represented by the predicted value at that node.

The cost-complexity criterion is also returned, for cross-validation of the alpha parameter.

An XOR example where the traditional tree fails but the current implementation can deal with is as follows:

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
 [1, 0.500026730694157]
	 [2, 0.5013652634065338]
		 terminal node:  0.0495724290546912
		 terminal node:  0.7629087465214373
	 [2, 0.5010964456753848]
		 terminal node:  0.7627006991381066
		 terminal node:  0.050296324591483414
```


[![DOI](https://zenodo.org/badge/458932097.svg)](https://zenodo.org/badge/latestdoi/458932097)

