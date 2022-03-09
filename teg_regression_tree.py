import numpy as np
import scipy
from scipy import stats

#
# Decision tree: Regression
#
def teg_regression_tree_externalEnsemble(X, y, maxDepth, alpha0, twostep = 1, internalEnsemble = 1):
    nSamples = N_ensemble_bootstraps
    best_split_feature_vec = []
    for iSample in range(nSamples):
        y_boot = np.random.choice(y, size=len(y))
        teg_regression_tree(X, y_boot, maxDepth, alpha0, twostep=twostep, internalEnsemble=internalEnsemble)

def teg_regression_tree(X, y, maxDepth, alpha0, twostep = 1, internalEnsemble = 1):

    N_ensemble_bootstraps = 5
    peek_ahead_quantiles = [0.25, 0.5, 0.75]

    def teg_tree_scale_variants(X, y, maxDepth, iDepth=0, node_index_v = [0]):
        # print("Params: ", twostep, internalEnsemble)
        if (iDepth == 0):
            node_index_v[0] = 0
        else:
            node_index_v[0] = node_index_v[0] + 1
        #print(node_index_v)
        SS_pre_split = f_SS(y)
        # Check whether maxdepth passed or y is empty
        if (iDepth >= maxDepth) or (len(y) <= 1) or (SS_pre_split == 0):
            terminal_node_pred = np.nanmean(y)
            return [[np.NaN, terminal_node_pred, SS_pre_split, 0, 0, 0, node_index_v[0], iDepth, y], np.NaN, np.NaN]
        # Create branches
        # Repeat for bootstrapped samples, pick top vote for this split
        if (internalEnsemble == 0):
            nSamples = 1
        else:
            nSamples = N_ensemble_bootstraps
        best_split_feature_vec = []
        best_split_list = []
        best_SS_list = []
        for iSample in range(nSamples):
            # print("iSample: ", iSample)
            if (internalEnsemble == 0):
                y_boot = y
                X_boot = X
            else:
                indices_boot = np.random.choice(len(y), size=len(y))
                y_boot = y[indices_boot]
                X_boot = X[indices_boot, :]
            best_split = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
            SS_best = np.inf
            SS_best_left = np.inf
            SS_best_right = np.inf
            iFeature_best = np.NaN
            val_best = np.NaN
            # Check one step ahead
            for iFeature1 in range(X_boot.shape[1]):
                # print("iFeature1: ", iFeature1)
                # print(iSample, iFeature1)
                best_split, SS_best = f_get_best_split(iFeature1, SS_best, best_split, y_boot, X_boot)
            best_split_list.append(best_split)
            best_SS_list.append(SS_best_left + SS_best_right)
            best_split_feature_vec.append(iFeature_best)
        # print(iDepth, best_split)
        # Use best feature from the bootstraps and apply to full dataset
        best_sample = np.argmin(best_SS_list)
        best_split = best_split_list[best_sample]
        iFeature1 = best_split[0]
        #print(iFeature1)
        splitting_var1 = X[:, iFeature1]
        #print(splitting_var1)
        best_split, SS_best = f_get_best_split(iFeature1, np.inf, best_split, y, X)
        #print(best_split)
        val1 = best_split[1]
        ind_left = (splitting_var1 < val1)
        ind_right = (splitting_var1 >= val1)
        SS_left = f_SS(y[ind_left])
        SS_right = f_SS(y[ind_right])
        SS_best_left = SS_left
        SS_best_right = SS_right
        iFeature_best = iFeature1
        val_best = val1
        best_split = [iFeature_best, val_best, SS_pre_split, SS_best_left, SS_best_right, len(y), node_index_v[0], iDepth, y]
        branch_left = teg_tree_scale_variants(X[ind_left, :], y[ind_left], maxDepth, iDepth + 1)
        branch_right = teg_tree_scale_variants(X[ind_right, :], y[ind_right], maxDepth, iDepth + 1)
        return [best_split, branch_left, branch_right]

    def f_get_best_split(iFeature1, SS_best, best_split, y_boot, X_boot):
        splitting_var1 = X_boot[:, iFeature1]
        splitting_vals1 = np.unique(splitting_var1)  # np.quantile(splitting_var1, peek_ahead_quantiles)
        for val1 in splitting_vals1:
            # print("val1: ", val1)
            if (twostep == 0):
                feature2vec = [0]
            else:
                feature2vec = range(X.shape[1])
            for iFeature2 in feature2vec:
                if (iFeature1 == iFeature2):
                    continue
                splitting_var2 = X_boot[:, iFeature2]
                if (twostep == 0):
                    splitting_vals2 = [-np.inf]
                else:
                    splitting_vals2 = np.quantile(splitting_var2, peek_ahead_quantiles)
                for val2 in splitting_vals2:
                    # print("val2: ", val2)
                    # print(iFeature1, ' ', iFeature2, ' ', val1, ' ', val2, '\n')
                    ind_left_left = ((splitting_var1 < val1) & (splitting_var2 < val2))
                    ind_left_right = ((splitting_var1 < val1) & (splitting_var2 >= val2))
                    ind_right_left = ((splitting_var1 >= val1) & (splitting_var2 < val2))
                    ind_right_right = ((splitting_var1 >= val1) & (splitting_var2 >= val2))
                    SS_left_left = f_SS(y_boot[ind_left_left])
                    SS_left_right = f_SS(y_boot[ind_left_right])
                    SS_right_left = f_SS(y_boot[ind_right_left])
                    SS_right_right = f_SS(y_boot[ind_right_right])
                    SS_this = SS_left_left + SS_left_right + SS_right_left + SS_right_right
                    if (SS_this < SS_best):
                        # Use double-split for best
                        SS_best = SS_this
                        # Use first split
                        ind_left = (splitting_var1 < val1)
                        ind_right = (splitting_var1 >= val1)
                        SS_left = f_SS(y_boot[ind_left])
                        SS_right = f_SS(y_boot[ind_right])
                        SS_best_left = SS_left
                        SS_best_right = SS_right
                        iFeature_best = iFeature1
                        val_best = val1
                        # print('New best: ', iFeature1, val1, iFeature2, val2, SS_best)
                        best_split = [iFeature_best, val_best]
        return best_split, SS_best

    def f_SS(v):
        if len(v) == 0:
            return 0
        return np.sum((v - np.mean(v))**2)

    # Cost-Complexity Pruning
    def retrieve_info_from_terminal_nodes(this_tree, nodes_to_collapse_tmp = [-1]):
        #print(nodes_to_collapse_tmp, this_tree[0][6], nodes_to_collapse_tmp.count(this_tree[0][6]))
        if np.isnan(this_tree[0][0]) or (nodes_to_collapse_tmp.count(this_tree[0][6]) > 0):
            # print(this_tree)
            # Elements divides by and then multiplies by Nm
            return this_tree[0][2], 1
        else:
            SS_left, N_left = retrieve_info_from_terminal_nodes(this_tree[1], nodes_to_collapse_tmp)
            SS_right, N_right = retrieve_info_from_terminal_nodes(this_tree[2], nodes_to_collapse_tmp)
            # print(N_left, N_right)
            return (SS_left + SS_right), (N_left + N_right)

    def f_C(this_tree, alpha0, nodes_to_collapse_tmp = [-1]):
        #print('zz', nodes_to_collapse_tmp)
        if nodes_to_collapse_tmp[0] == -1:
            node_indices = []
        this_SS, this_N = retrieve_info_from_terminal_nodes(this_tree, nodes_to_collapse_tmp)
        return this_SS + alpha0 * this_N

    def get_all_node_indices(this_tree, node_indices = [-1]):
        if node_indices[0] == -1:
            node_indices = []
        #print(this_tree)
        node_indices.append(this_tree[0][6])
        if not(np.isnan(this_tree[0][0])):
            get_all_node_indices(this_tree[1], node_indices)
            get_all_node_indices(this_tree[2], node_indices)
        return node_indices

    def get_internal_node_indices(this_tree, node_indices = [-1]):
        if node_indices[0] == -1:
            node_indices = []
        #print(this_tree)
        if not(np.isnan(this_tree[0][0])):
            node_indices.append(this_tree[0][6])
            get_internal_node_indices(this_tree[1], node_indices)
            get_internal_node_indices(this_tree[2], node_indices)
        return node_indices

    def get_downstream_nodes(this_tree, iNode_to_collapse, downstream_nodes = [-1], downstream_on = 0):
        if len(downstream_nodes) > 0:
            if downstream_nodes[0] == -1:
                downstream_nodes = []
        if this_tree[0][6] == iNode_to_collapse:
            downstream_on = 1
        if downstream_on == 1:
            downstream_nodes.append(this_tree[0][6])
        if not(np.isnan(this_tree[0][0])):
            get_downstream_nodes(this_tree[1], iNode_to_collapse, downstream_nodes, downstream_on)
            get_downstream_nodes(this_tree[2], iNode_to_collapse, downstream_nodes, downstream_on)
        return downstream_nodes

    def prune_the_tree(this_tree, alpha0):
        node_indices = get_internal_node_indices(this_tree)
        #print(node_indices)
        uncollapsed_v = [1 for a in node_indices]
        nodes_collapsed = []
        C = []
        while sum(uncollapsed_v) > 0:
            #print('x')
            C_vec_tmp = []
            iNode_indices_tmp = []
            iiNode_indices_tmp = []
            #print(node_indices)
            #print(uncollapsed_v)
            for iiNode in range(len(node_indices)):
                iNode = node_indices[iiNode]
                if uncollapsed_v[iiNode] == 0:
                    continue
                nodes_to_collapse_tmp = nodes_collapsed.copy()
                nodes_to_collapse_tmp.append(iNode)
                this_C = f_C(this_tree, alpha0, nodes_to_collapse_tmp)
                C_vec_tmp.append(this_C)
                iNode_indices_tmp.append(iNode)
                iiNode_indices_tmp.append(iiNode)
            #print(iiNode_indices_tmp)
            #print(iNode_indices_tmp)
            iiiNode_to_collapse = np.argmin(C_vec_tmp)
            iiNode_to_collapse = iiNode_indices_tmp[iiiNode_to_collapse]
            iNode_to_collapse = iNode_indices_tmp[iiiNode_to_collapse]
            ndf = get_downstream_nodes(this_tree, iNode_to_collapse)
            for intc in ndf: # iNodeToCollapse, includes source-collapser
                #print(intc)
                for ii in range(len(node_indices)):
                    #print(ii)
                    if intc == node_indices[ii]:
                        uncollapsed_v[ii] = 0
            nodes_collapsed.append(iNode_to_collapse)
            C.append(min(C_vec_tmp))
            #print(iiNode_to_collapse, iNode_to_collapse)
            # Collapse all downstream internal nodes
        return C, nodes_collapsed

    def print_tree(this_tree, C, nodes_collapsed, mean_y, sd_y):
        def print_tree_inner(this_tree, nodes_collapsed_choice, mean_y, sd_y):
            #print(this_tree[0][0])
            iDepth = int(this_tree[0][7])
            indent0 = ''
            for t in range(iDepth):
                indent0 = indent0 + '\t'
            if nodes_collapsed_choice.count(this_tree[0][6]) == 0 and not(np.isnan(this_tree[0][0])):
                print(indent0, this_tree[0][0:2])
                print_tree_inner(this_tree[1], nodes_collapsed_choice, mean_y, sd_y)
                print_tree_inner(this_tree[2], nodes_collapsed_choice, mean_y, sd_y)
            else:
                print(indent0, 'terminal node: ', mean_y + sd_y * np.nanmean(this_tree[0][-1]))
        if len(C) == 0:
            print('Empty tree.');
            return
        best_collapse_seq_end = np.argmin(C)
        nodes_collapsed_choice = nodes_collapsed[0:(best_collapse_seq_end + 1)]
        print_tree_inner(this_tree, nodes_collapsed_choice, mean_y, sd_y)

    def collapse_tree(this_tree, C, nodes_collapsed, mean_y, sd_y):
        def build_tree_inner(this_tree, nodes_collapsed_choice, mean_y, sd_y):
            if (nodes_collapsed_choice.count(this_tree[0][6]) == 0 and not(np.isnan(this_tree[0][0]))):
                to_report = this_tree[0][0:2]
                return [to_report, build_tree_inner(this_tree[1], nodes_collapsed_choice, mean_y, sd_y), build_tree_inner(this_tree[2], nodes_collapsed_choice, mean_y, sd_y)]
            else:
                to_report = mean_y + sd_y * np.nanmean(this_tree[0][-1])
                return to_report
        if len(C) == 0:
            return []
        best_collapse_seq_end = np.argmin(C)
        nodes_collapsed_choice = nodes_collapsed[0:(best_collapse_seq_end + 1)]
        return build_tree_inner(this_tree, nodes_collapsed_choice, mean_y, sd_y)

    mean_y = np.nanmean(y)
    sd_y = np.sqrt(np.var(y))
    y = (y - mean_y) / sd_y
    tree0 = teg_tree_scale_variants(X, y, maxDepth)

    C, nodes_collapsed = prune_the_tree(tree0, alpha0)
    #print(tree0)
    #print(C)
    #print(nodes_collapsed)
    # print(len(C))
    print_tree(tree0, C, nodes_collapsed, mean_y, sd_y)
    collapsed_tree = collapse_tree(tree0, C, nodes_collapsed, mean_y, sd_y)
    if len(C) > 0:
        return collapsed_tree, min(C)
    else:
        return collapsed_tree, np.NaN

def tree_prediction(X, tree0):
    def tree_prediction_inner(xvec, current_tree):
        this_split_var = current_tree[0][0]
        this_split_val = current_tree[0][1]
        if (xvec[this_split_var] < this_split_val):
            branch = current_tree[1]
        else:
            branch = current_tree[2]
        if isinstance(branch, list):
            prediction = tree_prediction_inner(xvec, branch)
        else:
            prediction = branch
        return prediction
    y_pred = []
    for xrow in X:
        y_pred.append(tree_prediction_inner(xrow, tree0))
    return y_pred

nObs = 2000
nPred = 6
maxDepth = 4 # Max. number of splits
alpha0 = 0.5
X = np.random.random_sample(size=(nObs, nPred))
y = 0 * np.random.random_sample(size=(nObs))
LogicalInd = (X[:, 1] > 0.8) & (X[:, 2] < 0.33) & (X[:, 4] < 0.5)
# y[LogicalInd] = 1 - (1 - y[LogicalInd]) * 0.25
y[LogicalInd] = 1
y = y + 0.1 * np.random.random_sample(size=(nObs))
# Traditional greedy tree
tree0, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0, twostep = 0, internalEnsemble = 0)
print(tree0)
print(cost_complexity_criterion)
# Greedy tree with internal ensemble
tree0, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0, twostep=0, internalEnsemble = 1)
print(tree0)
print(cost_complexity_criterion)
# Two-step peek-ahead tree, no internal ensemble
tree0, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0, internalEnsemble = 0)
print(tree0)
print(cost_complexity_criterion)
# Two-step tree, internal ensemble split by split
tree0, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0)
print(tree0)
print(cost_complexity_criterion)

# XOR problem: basic
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
# Two-step peek-ahead tree, no internal ensemble
tree0, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0, internalEnsemble = 0)
print(tree0)
print(cost_complexity_criterion)
# Two-step tree, internal ensemble split by split
tree0, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0)
print(tree0)
print(cost_complexity_criterion)

# XOR problem: tests
# Define "truth" by the true-model prediction of y, instead of trying to follow and compare paths.
tree_true = [[3, 0.5], [[0, 0.5], 0, 1], [[0, 0.5], 1, 0]]
nObs = 2000
nPred = 6
maxDepth = 4  # Max. number of splits
alpha0 = 0.5
nIts = 30
error_greedy_v = []
error_peek_v = []
error_peek_ensemble_v = []
noise0 = 0.1
noise_prop_randperm = 0
noise_prop_malice = 0.25
N_noise_prop_randperm = int(np.ceil(noise_prop_randperm * nObs))
N_noise_prop_malice = int(np.ceil(noise_prop_malice * nObs))
for iIt in range(nIts):
    print(iIt)
    X = np.random.random_sample(size=(nObs, nPred))
    y_true = tree_prediction(X, tree_true)
    # Add noise to true y
    y = np.array(y_true)
    y[-N_noise_prop_malice:] = 1 - y[-N_noise_prop_malice:]
    y[0:N_noise_prop_randperm] = np.random.permutation(y[0:N_noise_prop_randperm])
    y = y + noise0 * np.random.randn(nObs)
    # Run tree variants
    tree_greedy, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0, twostep=0, internalEnsemble = 0)
    y_greedy = tree_prediction(X, tree_greedy)
    error_greedy = np.mean((np.array(y_true) - np.array(y_greedy))**2)
    error_greedy_v.append(error_greedy)
    tree_peek, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0, twostep=1, internalEnsemble = 0)
    y_peek = tree_prediction(X, tree_peek)
    error_peek = np.mean((np.array(y_true) - np.array(y_peek))**2)
    error_peek_v.append(error_peek)
    tree_peek, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0, twostep=1, internalEnsemble = 1)
    y_peek_ensemble = tree_prediction(X, tree_peek)
    error_peek_ensemble = np.mean((np.array(y_true) - np.array(y_peek_ensemble))**2)
    error_peek_ensemble_v.append(error_peek_ensemble)
# Compare
# Greedy versus peek
d = np.array(error_greedy_v) - np.array(error_peek_v)
print(scipy.stats.ttest_1samp(d, 0))
# Greedy versus ensemble
d = np.array(error_greedy_v) - np.array(error_peek_ensemble_v)
print(scipy.stats.ttest_1samp(d, 0))
# Peek versus ensemble
d = np.array(error_peek_v) - np.array(error_peek_ensemble_v)
print(scipy.stats.ttest_1samp(d, 0))

#
# Empirical data
#
import pandas as pd
fn = 'D:/Dropbox/Work/Projects/GladwinParadigms/Current/2022_01_30_tree0/Data/Data.txt'
DF = pd.read_csv(fn, delimiter="\t")
print(DF.columns)
y = DF['AUDIT3'].to_numpy()
yz = (y - np.nanmean(y)) / np.sqrt(np.var(y))
pred_names = ['PhysicalAggr', 'VerbalAggr', 'Anger', 'Hostility', 'PHQ9', 'Total', 'STAI1', 'DMQ_Soc', 'DMQ_Coping', 'DMQ_Enh', 'DMQ_Conform', 'RALD_LossOfControl', 'RALD_AdverseConseq', 'RALD_Convictions', 'ACS_Compuls', 'ACS_Expect', 'ACS_Purpose', 'ACS_Emot']
X = DF[pred_names].to_numpy()
Xz = np.array([(Xcol - np.nanmean(Xcol, axis=0)) / np.sqrt(np.var(Xcol, axis=0)) for Xcol in X.T]).T
#
maxDepth = 4 # Max. number of splits
alpha0 = 0.5
tree0, cost_complexity_criterion = teg_regression_tree(Xz, yz, maxDepth, alpha0)
for index, value in enumerate(pred_names):
    print(index, value, end='.\t')
    if (index + 1) % 5 == 0:
        print()
#
Xtest = np.random.normal(size=(1000, 5))
np.corrcoef(Xtest.T)
#PCA for indep predictors
Cov0 = np.cov(Xz.T)
eigen_vals0, eigen_vecs0 = np.linalg.eig(Cov0)
ind0 = np.flip(np.argsort(eigen_vals0))
eigen_vals0 = eigen_vals0[ind0]
eigen_vecs0 = eigen_vecs0[ind0]
elbow0 = 5
eigen_vecs = eigen_vecs0[:, 0:elbow0]
L = Xz @ eigen_vecs
np.corrcoef(L.T)
# print(eigen_vecs.T[3])
a = [10, 11, 12, 4, 5, 1, 9, 4, 3, 22, 10, 9, 1, 2, 1, 0, 8, 2];
A = np.reshape(a, (3, 6)).T
A = A - np.mean(A, axis=0)
Cov0 = np.cov(A.T)
eigen_vals0, eigen_vecs0 = np.linalg.eig(Cov0)
L = A @ eigen_vecs0
np.corrcoef(L.T)
# FA
from sklearn.decomposition import FactorAnalysis, PCA
fa = FactorAnalysis(rotation='varimax')
elbow0 = 5
fa.set_params(n_components=elbow0)
fa.fit(Xz)
components = fa.components_
L = fa.fit_transform(Xz)
#
print(np.corrcoef(L.T))

alpha0 = 0.5
tree0, cost_complexity_criterion = teg_regression_tree(L, yz, maxDepth, alpha0, twostep = 1, internalEnsemble=1)
# 1 = Social drinker, emotionality-craving
# 3 = Enhancement, expectancy-craving
# Statistics
# Test overall fit vs null distrib
# -> Or: Test absolute terminal value vs null distrib (FWE), only consider "paths" to those values as meaningful
def get_max_abs_mean_preds(this_tree, current_max =-np.inf):
    if isinstance(this_tree, float):
        if np.abs(this_tree) > current_max:
            current_max = np.abs(this_tree)
        return current_max
    else:
        L_max = get_max_abs_mean_preds(this_tree[1], current_max)
        R_max = get_max_abs_mean_preds(this_tree[2], current_max)
        if L_max > current_max:
            current_max = L_max
        if R_max > current_max:
            current_max = R_max
        return current_max
nIts = 200
null_dist = []
null_dist_CCC = []
twostep = 1
internalEnsemble = 0
alpha0 = 0.5
maxDepth = 4 # Max. number of splits
y = yz.copy()
# X = Xz.copy()
X = L.copy()
for iIt in range(nIts):
    print(iIt, end='\t')
    y_perm = np.random.choice(y, size=len(y))
    tree0_perm, cost_complexity_criterion_perm = teg_regression_tree(X, y_perm, maxDepth, alpha0, twostep = twostep, internalEnsemble = internalEnsemble) # Make sure this matches observed tree
    max_term_perm = get_max_abs_mean_preds(tree0_perm)
    null_dist.append(max_term_perm)
    null_dist_CCC.append(cost_complexity_criterion_perm)
print()
tree0, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0, twostep = twostep, internalEnsemble = internalEnsemble)
max_term_obs = get_max_abs_mean_preds(tree0)
p = sum(null_dist > max_term_obs)/len(null_dist)
print('p (max abs mean pred) = ', p)
p = sum(null_dist_CCC < cost_complexity_criterion)/len(null_dist_CCC) # Note low is good for CCC
print('p (CCC) = ', p)
for index, value in enumerate(components):
    for index2, this_label in enumerate(pred_names):
        print(index, index2, this_label, np.around(value[index2], 3))
# Give proportional reduction in unexplained variance

# NHST for max abs mean terminal val for fixed max depth

# Fix tree for subsets of variables from PCA and get p-value; find more precise predictors within those subsets.

