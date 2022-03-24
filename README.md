# teg_bonsai

Originally a practice project and for messing-around purposes: a regression tree function, a la The Elements of Statistical Learning, with a some variations that might make it more suitable for certain research purposes. The functions are in teg_bonsai.py, and usage examples are in examples.py.

First, I added "peek ahead" variation to deal with XOR patterns (and, up to a specified maximum peek-ahead depth, arbirtarily higher-order nested XOR's, although it gets slow quickly). "Peeking ahead" means: for each possible split, all possible immediately following splits (for a set of quantiles to keep computation time down, trading off the resolution of the peek-ahead) are used to evaluate the first split; and this is repeated up to a given peek-ahead depth. This avoids the usual greedy algorithm's trap of not being able to recognize a split that is only good in combination with a subsequent split.

Second, it has an internal cross-validation option in which the tree is built on one randomly selected split-half of the data and pruned based on its application to the other, non-overlapping split-half; the best tree based on the cross-validated cost complexity criterion over a number of random splits is chosen. This is controlled by the combination of the nSamples parameter and setting internal_cross_val = 1. nSamples specifies the number of random splits over which the best tree is selected based on the cross-validated criterion.

Third, building on the cross-validation using independent sub-samples, a p-value is generated for null hypothesis significance testing. This is based on a random permutation of the cross-validation data. Thus, both the cross-validation and null sets are independent from the set used to create the tree under the null hypothesis of there being no dependence of the target y on the predictors X. The function returns the p-value for the one-sample t-test of the cross-validation-set minus null-set cost complexity criterion difference score over the random samples.

Finally, a "beta" parameter can be added to the usual alpha pruning parameter, which punishes potential splits, during tree generation, below a given proportion of the original number of observations. This is controlled by a two-element vector: the proportion of observation when the punishment starts, and a coefficient for a linear proportional increase in sum of squares (infinity can be used to forbid small splits). The idea is to prevent what might be obviously spurious or theoretically unimportant splits that could nevertheless affect the branching. This would lead, hopefully, to more interpretable trees.

Example usage with sanity-check simulated data:
```
nObs = 1500
nPred = 4
maxDepth = 4
peek_ahead_max_depth = 0
nSamples = 5
internal_cross_val = 1
X = np.round(np.random.random_sample(size=(nObs, nPred)), 2)
y = np.zeros(nObs)
LogicalInd = (X[:, 0] >= 0.8) & (X[:, 2] < 0.33) & (X[:, 4] < 0.5)
y[LogicalInd] = 1
y_true = y
y = y + 0.1 * np.random.random_sample(size=(nObs))
alpha0 = 0.5
beta0_vec = [0.01, np.inf]
tree = teg_bonsai.Tree(X, y, maxDepth, alpha0, peek_ahead_max_depth=peek_ahead_max_depth, nSamples=nSamples, internal_cross_val=internal_cross_val, beta0_vec=beta0_vec)
Output = tree.build_tree()
```
The function prints out the tree as follows, with low and high branches starting on different lines with the same indentation, with the predicted value shown for terminal nodes:

```
[1, 0.8125486774415731]
	 terminal node:  0.05188583804970138
	 [2, 0.33236530087901717]
		 [4, 0.5128665191583686]
			 terminal node:  0.9623787750856714
			 terminal node:  0.04541482950659097
		 terminal node:  0.051131886885080774
```

The function also returns a nested list representing the tree, as one of the elements in Output:

`[[1, 0.8125486774415731], 0.05188583804970138, [[2, 0.33236530087901717], [[4, 0.5128665191583686], 0.9623787750856714, 0.04541482950659097], 0.051131886885080774]]`

Each non-terminal node is a triplet containing a 2-element list with the feature-index and split-point of the node, the left-branch, and the right-branch. Terminal nodes are represented by the predicted value at that node.

The cost-complexity criterion is also returned, for cross-validation of the alpha parameter.

An XOR example where the traditional tree fails but the current implementation can deal with is as follows:

```
nObs = 1000
nPred = 4
maxDepth = 4 # Max. number of splits
alpha0 = 0.5
peek_ahead_max_depth = 1
X = np.round(np.random.random_sample(size=(nObs, nPred)), 2)
y = np.zeros(nObs)
LogicalInd = (X[:, 0] >= 0.5) & (X[:, 1] < 0.5)
y[LogicalInd] = 1
LogicalInd = (X[:, 0] < 0.5) & (X[:, 1] >= 0.5)
y[LogicalInd] = 1
y = y + 0.1 * np.random.randn(nObs)
y = np.round(y, 2)
tree = teg_bonsai.Tree(X, y, maxDepth, alpha0, peek_ahead_max_depth=peek_ahead_max_depth, nSamples=nSamples, internal_cross_val=internal_cross_val, beta0_vec=beta0_vec)
Output = tree.build_tree()
```

This would tend to  (after some time, it's not fast...) print out the correct solution for a peek-ahead depth of 1, splitting first on X1 and then on each branch on X2:

```
 [1, 0.500026730694157]
	 [2, 0.5013652634065338]
		 terminal node:  0.0495724290546912
		 terminal node:  0.9629087465214373
	 [2, 0.5010964456753848]
		 terminal node:  0.9627006991381066
		 terminal node:  0.050296324591483414
```

Thomas E. Gladwin (2022). thomasgladwin/teg_bonsai: Class version (v1.1). Zenodo. https://doi.org/10.5281/zenodo.6374306.

[![DOI](https://zenodo.org/badge/458932097.svg)](https://zenodo.org/badge/latestdoi/458932097)


