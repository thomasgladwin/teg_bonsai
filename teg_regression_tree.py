import numpy as np
import scipy
from scipy import stats

#
# Decision tree: Regression
#
def teg_regression_tree_peeks(X, y, maxDepth, alpha0, peek_ahead_max_depth, peek_ahead_quantiles = [0.25, 0.5, 0.75, 1], no_immediate_return_to_feature = 0, never_return_to_feature = 0):
    tree0 = []
    cost_complexity_criterion = np.inf
    best_peek_crit = np.NaN
    for peek_ahead_depth in range(peek_ahead_max_depth + 1):
        print('Finding tree for peek_ahead_depth = ', peek_ahead_depth)
        tree0_this, cost_complexity_criterion_this = teg_regression_tree(X, y, maxDepth, alpha0, peek_ahead_depth=peek_ahead_depth, peek_ahead_quantiles=peek_ahead_quantiles, no_immediate_return_to_feature = no_immediate_return_to_feature, never_return_to_feature = never_return_to_feature)
        print('Cost-Complexity Criterion = ', cost_complexity_criterion_this)
        if cost_complexity_criterion_this < cost_complexity_criterion:
            tree0 = tree0_this
            cost_complexity_criterion = cost_complexity_criterion_this
            best_peek_crit = peek_ahead_depth
            print(" ! New best tree !")
        print("\n")
    print("Best tree was found at peek-ahead depth = ", best_peek_crit)
    return tree0, cost_complexity_criterion, best_peek_crit

def teg_regression_tree(X, y, maxDepth, alpha0, peek_ahead_depth = 0, peek_ahead_quantiles = [0.25, 0.5, 0.75, 1], no_immediate_return_to_feature = 1, never_return_to_feature = 0):

    def init_best_split():
        return np.NaN * np.zeros(9)

    def teg_tree_scale_variants(X, y, maxDepth, visited_features = [], iDepth=0, node_index_v = [0]):
        # print("Params: ", twostep, internalEnsemble)
        if (iDepth == 0):
            node_index_v[0] = 0
        else:
            node_index_v[0] = node_index_v[0] + 1
        #print(node_index_v)
        SS_pre_split = f_SS(y)
        # Check whether maxdepth passed or y is empty
        if (iDepth >= maxDepth) or (len(y) <= 1) or (SS_pre_split == 0):
            if len(y) > 0:
                terminal_node_pred = np.nanmean(y)
            else:
                terminal_node_pred = np.NaN
            return [[np.NaN, terminal_node_pred, SS_pre_split, 0, 0, 0, node_index_v[0], iDepth, y], np.NaN, np.NaN]
        # Create branches
        # Check one step ahead
        best_split_feature = np.NaN
        best_split_val = np.NaN
        SS_best = np.inf
        for iFeature1 in range(X.shape[1]):
            # print("iFeature1: ", iFeature1)
            # print(iSample, iFeature1)
            if (visited_features.count(iFeature1) > 0) and (never_return_to_feature == 1):
                continue
            if len(visited_features) > 0:
                if (visited_features[-1] == iFeature1) and (no_immediate_return_to_feature == 1):
                    continue
            best_split_val_this, SS_best_this = f_get_best_split_val(iFeature1, y, X, maxDepth - iDepth)
            if SS_best_this < SS_best:
                best_split_feature = iFeature1
                best_split_val = best_split_val_this
                SS_best = SS_best_this
        if np.isnan(best_split_feature):
            if len(y) > 0:
                terminal_node_pred = np.nanmean(y)
            else:
                terminal_node_pred = np.NaN
            return [[np.NaN, terminal_node_pred, SS_pre_split, 0, 0, 0, node_index_v[0], iDepth, y], np.NaN, np.NaN]
        ind_left = (X[:, best_split_feature] < best_split_val)
        ind_right = (X[:, best_split_feature] >= best_split_val)
        SS_left = f_SS(y[ind_left])
        SS_right = f_SS(y[ind_right])
        best_split = [best_split_feature, best_split_val, SS_pre_split, SS_left, SS_right, len(y), node_index_v[0], iDepth, y]
        visited_features.append(best_split_feature)
        branch_left = teg_tree_scale_variants(X[ind_left, :], y[ind_left], maxDepth, visited_features, iDepth + 1)
        branch_right = teg_tree_scale_variants(X[ind_right, :], y[ind_right], maxDepth, visited_features, iDepth + 1)
        return [best_split, branch_left, branch_right]

    def f_get_best_SS_peek(visited_features, y, X, peek_ahead_maxDepth_limiter, current_peek_depth = 0):
        if len(y) <= 1:
            return 0
        if ((current_peek_depth >= peek_ahead_depth) or current_peek_depth > peek_ahead_maxDepth_limiter):
            best_SS = f_SS(y)
        else:
            best_SS = np.inf
            for iFeature_this in range(X.shape[1]):
                if visited_features.count(iFeature_this) > 0:
                    continue
                splitting_vals_this = np.quantile(X[:, iFeature_this], peek_ahead_quantiles)
                new_visited_features = visited_features.copy()
                new_visited_features.append(iFeature_this)
                for val_this in splitting_vals_this:
                    ind_left = (X[:, iFeature_this] < val_this)
                    ind_right = (X[:, iFeature_this] >= val_this)
                    best_SS_left = f_get_best_SS_peek(new_visited_features, y[ind_left], X[ind_left, :], peek_ahead_maxDepth_limiter, current_peek_depth + 1)
                    best_SS_right = f_get_best_SS_peek(new_visited_features, y[ind_right], X[ind_right, :], peek_ahead_maxDepth_limiter, current_peek_depth + 1)
                    current_SS = best_SS_left + best_SS_right
                    if (current_SS < best_SS):
                        best_SS = current_SS
            if np.isinf(best_SS):
                best_SS = f_SS(y)
        return best_SS

    def f_get_best_split_val(iFeature1, y, X, peek_ahead_maxDepth_limiter):
        best_split_val = np.NaN
        SS_best = np.inf
        splitting_vals1 = np.unique(X[:, iFeature1])
        for val1 in splitting_vals1:
            ind_left = (X[:, iFeature1] < val1)
            ind_right = (X[:, iFeature1] >= val1)
            SS_left = f_get_best_SS_peek([iFeature1], y[ind_left], X[ind_left, :], peek_ahead_maxDepth_limiter)
            SS_right = f_get_best_SS_peek([iFeature1], y[ind_right], X[ind_right, :], peek_ahead_maxDepth_limiter)
            SS_this = SS_left + SS_right
            if (SS_this < SS_best):
                SS_best = SS_this
                best_split_val = val1
        return best_split_val, SS_best

    def f_SS(v):
        if len(v) == 0:
            return 0
        # print('141')
        return_val = np.sum((v - np.mean(v))**2)
        # print('141')
        return return_val

    # Cost-Complexity Pruning
    def retrieve_info_from_terminal_nodes(this_tree, nodes_to_collapse_tmp = [-1]):
        #print(nodes_to_collapse_tmp, this_tree[0][6], nodes_to_collapse_tmp.count(this_tree[0][6]))
        if np.isnan(this_tree[0][0]) or (nodes_to_collapse_tmp.count(this_tree[0][6]) > 0):
            # print(this_tree)
            # Elements divides by and then multiplies by Nm
            return this_tree[0][2], 1, this_tree[0][7]
        else:
            SS_left, N_left, depth_left = retrieve_info_from_terminal_nodes(this_tree[1], nodes_to_collapse_tmp)
            SS_right, N_right, depth_right = retrieve_info_from_terminal_nodes(this_tree[2], nodes_to_collapse_tmp)
            # print(N_left, N_right)
            return (SS_left + SS_right), (N_left + N_right), max(depth_left, depth_right)

    def f_C(this_tree, alpha0, nodes_to_collapse_tmp = [-1]):
        #print('zz', nodes_to_collapse_tmp)
        if nodes_to_collapse_tmp[0] == -1:
            node_indices = []
        this_SS, this_N, max_depth_terminals = retrieve_info_from_terminal_nodes(this_tree, nodes_to_collapse_tmp)
        # print("fC: ", this_N, max_depth_terminals)
        return this_SS + alpha0 * this_N
        # return this_SS + alpha0 * this_N * max_depth_terminals

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
                m = np.nanmean(this_tree[0][-1])
                print(indent0, 'terminal node: ', mean_y + sd_y * m)
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
        if not isinstance(current_tree, list):
            prediction = current_tree
        else:
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

nObs = 400
nPred = 4
maxDepth = 4
alpha0 = 1
max_peek_depth = 2
X = np.random.random_sample(size=(nObs, nPred))
y = np.zeros(nObs)
LogicalInd = (X[:, 0] > 0.8) & (X[:, 2] < 0.33)
y[LogicalInd] = 1
y = y + 0.1 * np.random.random_sample(size=(nObs))
tree0, cost_complexity_criterion, best_peek_crit = teg_regression_tree_peeks(X, y, maxDepth, alpha0, max_peek_depth)

# XOR problem
nObs = 1000
nPred = 4
maxDepth = 4 # Max. number of splits
alpha0 = 0.5
max_peek_depth = 2
X = np.random.random_sample(size=(nObs, nPred))
y = np.zeros(nObs)
LogicalInd = (X[:, 1] >= 0.5) & (X[:, 2] < 0.5)
y[LogicalInd] = 1
LogicalInd = (X[:, 1] < 0.5) & (X[:, 2] >= 0.5)
y[LogicalInd] = 1
y = y + 0.01 * np.random.randn(nObs)
tree0, cost_complexity_criterion, best_peek_crit = teg_regression_tree_peeks(X, y, maxDepth, alpha0, max_peek_depth)

# Higher-order XOR problem
nObs = 4000
nPred = 4
maxDepth = 4 # Max. number of splits
alpha0 = 0.5
max_peek_depth = 2
X = np.random.random_sample(size=(nObs, nPred))
y = np.zeros(nObs)
LogicalInd = (X[:, 0] < 0.5) & (X[:, 1] >= 0.5) & (X[:, 2] < 0.5)
y[LogicalInd] = 1
LogicalInd = (X[:, 0] < 0.5) & (X[:, 1] < 0.5) & (X[:, 2] >= 0.5)
y[LogicalInd] = 1
LogicalInd = (X[:, 0] >= 0.5) & (X[:, 1] >= 0.5) & (X[:, 2] >= 0.5)
y[LogicalInd] = 1
LogicalInd = (X[:, 0] >= 0.5) & (X[:, 1] < 0.5) & (X[:, 2] < 0.5)
y[LogicalInd] = 1
y = y + 0.01 * np.random.randn(nObs)
tree0, cost_complexity_criterion, best_peek_crit = teg_regression_tree_peeks(X, y, maxDepth, alpha0, max_peek_depth)

# XOR problem: tests
# Define "truth" by the true-model prediction of y, instead of trying to follow and compare paths.
tree_true = [[3, 0.5], [[0, 0.5], 0, 1], [[0, 0.5], 1, 0]]
nObs = 200
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

# Tests, non-XOR-specific
tree_true = [[3, 0.5], [[0, 0.33], 0, [[1, 0.6], -1, 1]], 0]
nObs = 400
nPred = 6
maxDepth = 4  # Max. number of splits
alpha0 = 1
nIts = 30
error_greedy_v = []
error_ensemble_v = []
noise0 = 0.25
noise_prop_randperm = 0
noise_prop_malice = 0
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
    tree_greedy, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0, twostep=1, internalEnsemble = 0)
    y_greedy = tree_prediction(X, tree_greedy)
    error_greedy = np.mean((np.array(y_true) - np.array(y_greedy))**2)
    error_greedy_v.append(error_greedy)
    tree_ensemble, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0, twostep=0, internalEnsemble = 1)
    y_ensemble = tree_prediction(X, tree_ensemble)
    error_ensemble = np.mean((np.array(y_true) - np.array(y_ensemble))**2)
    error_ensemble_v.append(error_ensemble)
# Compare
# Greedy versus ensemble
d = np.array(error_greedy_v) - np.array(error_ensemble_v)
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
print(np.corrcoef(L.T))
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
#
maxDepth = 4
alpha0 = 0.5
max_peek_depth = 2
tree0, cost_complexity_criterion, best_peek_crit = teg_regression_tree_peeks(L, y, maxDepth, alpha0, max_peek_depth)

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

