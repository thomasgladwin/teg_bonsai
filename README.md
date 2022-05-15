# teg_bonsai

Originally a practice project and for messing-around purposes, possibly some useful elements to it now: a regression tree function, a la The Elements of Statistical Learning, with variations that might make it more suitable for certain research purposes. The functions are in teg_bonsai.py, and usage examples are in examples.py.

First, it has an internal cross-validation option in which the tree is built on one randomly selected split-half of the data and selected and optionally additionally pruned based on its application to the other, non-overlapping split-half; the best tree based on the cross-validated cost complexity criterion over a number of random splits is chosen. This is controlled by the combination of the nSamples parameter and setting internal_cross_val = 1 and internal_cross_val_nodes = 1. nSamples specifies the number of random splits over which the best tree is selected based on the cross-validated criterion.

Second, building on the idea of cross-validation using independent sub-samples, a p-value is generated for null hypothesis significance testing, including hierarchical testing against a baseline set. The null distribution of minimal cost-complexity criterion values is based on random permutations of the cross-validation data. Thus, both the cross-validation and null sets are independent from the set used to create the tree under the null hypothesis of there being no dependence of the target y on the predictors X (or, the predictors not in the baseline). The function returns the p-value for the binomial test of the cross-validation data having a better cost-complexity criterionthan the permuted data. The parameters controlling the NHST are nSamples for the number of random samples used to create the linked cross-validation and null distributions; and baseline_features, a list of column indices specifying which predictors form the baseline set. The p-value (returned in Output['p']) when a baseline is defined tests whether the remaining predictors significantly improve the cost-complexity criterion above the baseline.

Third, an auto-selected alpha parameter can be generated, by setting alpha0 = None. This searches for the smallest alpha that does not tend to produce a non-empty tree for randomly permuted data.

Fourth, there's a "peek ahead" variation to deal with XOR patterns (and, up to a specified maximum peek-ahead depth, arbirtarily higher-order nested XOR's, although it gets slow quickly). "Peeking ahead" means: for each possible split, all possible immediately following splits (for a set of quantiles to keep computation time down, trading off the resolution of the peek-ahead) are used to evaluate the first split; and this is repeated up to a given peek-ahead depth. This avoids the usual greedy algorithm's trap of not being able to recognize a split that is only good in combination with a subsequent split.

Example usage with sanity-check simulated data are provided in examples.py. E.g.,

```
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
alpha0 = None
beta0_vec = [0.01, np.inf]
tree = teg_bonsai.Tree(X, y, maxDepth, alpha0, peek_ahead_max_depth=peek_ahead_max_depth, nSamples=nSamples, internal_cross_val=internal_cross_val, beta0_vec=beta0_vec)
Output = tree.build_tree()
```
The function prints out the tree as follows, with low and high branches starting on different lines with the same indentation, with the predicted value shown for terminal nodes:

```
 [0, 0.8]
	 terminal node:  0.048479900254497554
	 [2, 0.33]
		 terminal node:  1.0560146013204075
		 terminal node:  0.050794105233023815

```

The function also returns a nested list representing the tree, as one of the elements in Output:

`[[0, 0.8], 0.048479900254497554, [[2, 0.33], 1.0560146013204075, 0.050794105233023815]]`

Each non-terminal node is a triplet containing a 2-element list with the feature-index and split-point of the node, the left-branch, and the right-branch. Terminal nodes are represented by the predicted value at that node.

The cost-complexity criterion is also returned, for cross-validation of the alpha parameter.

An XOR example that the traditional tree would fail on but the current implementation can deal with is as follows:

```
nObs = 1500
nPred = 4
maxDepth = 4 # Max. number of splits
alpha0 = 0.01
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
```

This would tend to  (after some time, it's not fast...) print out the correct solution for a peek-ahead depth of 1, splitting first on X1 and then on each branch on X2:

```
[0, 0.5]
	 [1, 0.5]
		 terminal node:  0.00044943820224707665
		 terminal node:  0.9996446700507615
	 [1, 0.5]
		 terminal node:  0.9997395833333333
		 terminal node:  0.00010989010989015391

```

Thomas Edward Gladwin. (2022). thomasgladwin/teg_bonsai: Auto-alpha version (v1.2). Zenodo. https://doi.org/10.5281/zenodo.6456317

[![DOI](https://zenodo.org/badge/458932097.svg)](https://zenodo.org/badge/latestdoi/458932097)




