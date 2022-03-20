# teg_regression_tree

Originally a practice project and for messing-around purposes: regression tree function, a la The Elements of Statistical Learning, with a some variations that might make it more suitable for certain research purposes.

First, I added "peek ahead" variation to deal with XOR patterns. "Peeking ahead" means: for each possible split, all possible immediately following splits (for a set of quantiles to keep computation time down, trading off the resolution of the peek-ahead) are used to evaluate the first split; and this is repeated up to a given peek-ahead depth. This avoids the usual greedy algorithm's trap of not being able to recognize a split that is only good in combination with a subsequent split.

Second, it has an internal cross-validation option in which the tree is built on one randomly selected split-half of the data and pruned based on its application to the other, non-overlapping split-half; the best tree based on the cross-validated cost complexity criterion over a number of random splits is chosen. This is controlled by the combination of the nSamples parameter and setting internal_cross_val = 1. nSamples specifies the number of random splits over which the best tree is selected based on the cross-validated criterion.

Third, building on the cross-validation using independent sub-samples, a p-value is generated for null hypothesis significance testing. This is based on a random permutation of the cross-validation data. Thus, both the cross-validation and null sets are independent from the set used to create the tree under the null hypothesis of there being no dependence of the target y on the predictors X. The function returns the p-value for the one-sample t-test of the cross-validation-set minus null-set cost complexity criterion difference score over the random samples.

Finally, a "beta" parameter can be added to the usual alpha pruning parameter, which punishes potential splits, during tree generation, below a given proportion of the original number of observations. This prevents what might be obviously spurious or theoretically unimportant splits that could nevertheless affect the branching. This would lead, hopefully, to more interpretable trees.

Example usage with sanity-check simulated data:
```
nObs = 2000
nPred = 6
maxDepth = 4 # Max. number of splits
alpha0 = 0.5
max_peek_depth = 2
X = np.random.random_sample(size=(nObs, nPred))
y = 0.1 * np.random.random_sample(size=(nObs))
LogicalInd = (X[:, 1] > 0.8) & (X[:, 2] < 0.33) & (X[:, 4] < 0.5)
y[LogicalInd] = 1 - (1 - y[LogicalInd]) * 0.25
tree0, cost_complexity_criterion, best_peek_crit, raw_tree, CV_distr, null_distr, p = teg_regression_tree_peeks(X, y, maxDepth, alpha0, max_peek_depth)
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
max_peek_depth = 2
alpha0 = 0.5
X = np.random.random_sample(size=(nObs, nPred))
y = 0.1 * np.random.random_sample(size=(nObs))
LogicalInd = (X[:, 1] > 0.5) & (X[:, 2] < 0.5)
y[LogicalInd] = 1 - (1 - y[LogicalInd]) * 0.25
LogicalInd = (X[:, 1] < 0.5) & (X[:, 2] > 0.5)
y[LogicalInd] = 1 - (1 - y[LogicalInd]) * 0.25
tree0, cost_complexity_criterion, best_peek_crit, raw_tree, CV_distr, null_distr, p = teg_regression_tree_peeks(X, y, maxDepth, alpha0, max_peek_depth)
```

This would tend to  (after some time, it's not fast...) print out the correct solution for a peek-ahead depth of 1, splitting first on X1 and then on each branch on X2:

```
 [1, 0.500026730694157]
	 [2, 0.5013652634065338]
		 terminal node:  0.0495724290546912
		 terminal node:  0.7629087465214373
	 [2, 0.5010964456753848]
		 terminal node:  0.7627006991381066
		 terminal node:  0.050296324591483414
```

Thomas E. Gladwin (2022). thomasgladwin/teg_regression_tree: Update 1 (v1.0). Zenodo. https://doi.org/10.5281/zenodo.6371375


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6371375.svg)](https://doi.org/10.5281/zenodo.6371375)

