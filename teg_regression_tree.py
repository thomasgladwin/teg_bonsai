import numpy as np

#
# Decision tree: Regression
#
def teg_regression_tree(X, y, maxDepth, alpha0, twostep = 1, internalEnsemble = 1):

    def teg_tree_scale(X, y, maxDepth, iDepth=0, node_index_v = [0]):
        if (iDepth == 0):
            node_index_v[0] = 0
        else:
            node_index_v[0] = node_index_v[0] + 1
        #print(node_index_v)
        SS_pre_split = f_SS(y)
        # Check whether maxdepth passed or y is empty
        if (iDepth >= maxDepth) or (len(y) <= 1) or (SS_pre_split== 0):
            terminal_node_pred = np.nanmean(y)
            return [[np.NaN, terminal_node_pred, SS_pre_split, 0, 0, 0, node_index_v[0], iDepth, y], np.NaN, np.NaN]
        # Create branches
        SS_best = np.inf
        SS_best_left = np.inf
        SS_best_right = np.inf
        iFeature_best = np.NaN
        val_best = np.NaN
        for iFeature in range(X.shape[1]):
            splitting_var = X[:, iFeature]
            for val in splitting_var:
                ind_left = (splitting_var < val)
                ind_right = (splitting_var >= val)
                SS_left = f_SS(y[ind_left])
                SS_right = f_SS(y[ind_right])
                SS_this = SS_left + SS_right
                if (SS_this < SS_best):
                    SS_best = SS_this
                    SS_best_left = SS_left
                    SS_best_right = SS_right
                    iFeature_best = iFeature
                    val_best = val
        best_split = [iFeature_best, val_best, SS_pre_split, SS_best_left, SS_best_right, len(y), node_index_v[0], iDepth, y]
        # print(iDepth, best_split)
        ind_left = (X[:, iFeature_best] < val_best)
        ind_right = (X[:, iFeature_best] >= val_best)
        branch_left = teg_tree_scale(X[ind_left, :], y[ind_left], maxDepth, iDepth + 1)
        branch_right = teg_tree_scale(X[ind_right, :], y[ind_right], maxDepth, iDepth + 1)
        return [best_split, branch_left, branch_right]

    def teg_tree_scale_internalEnsemble(X, y, maxDepth, iDepth=0, node_index_v = [0]):
        if (iDepth == 0):
            node_index_v[0] = 0
        else:
            node_index_v[0] = node_index_v[0] + 1
        #print(node_index_v)
        SS_pre_split = f_SS(y)
        # Check whether maxdepth passed or y is empty
        if (iDepth >= maxDepth) or (len(y) <= 1) or (SS_pre_split== 0):
            terminal_node_pred = np.nanmean(y)
            return [[np.NaN, terminal_node_pred, SS_pre_split, 0, 0, 0, node_index_v[0], iDepth, y], np.NaN, np.NaN]
        # Create branches
        # Repeat for bootstrapped samples, pick top vote for this split
        nSamples = 3
        best_split_feature_vec = []
        for iSample in range(nSamples):
            y_boot = np.random.choice(y, size=len(y))
            SS_best = np.inf
            SS_best_left = np.inf
            SS_best_right = np.inf
            iFeature_best = np.NaN
            val_best = np.NaN
            for iFeature in range(X.shape[1]):
                splitting_var = X[:, iFeature]
                for val in splitting_var:
                    ind_left = (splitting_var < val)
                    ind_right = (splitting_var >= val)
                    SS_left = f_SS(y_boot[ind_left])
                    SS_right = f_SS(y_boot[ind_right])
                    SS_this = SS_left + SS_right
                    if (SS_this < SS_best):
                        SS_best = SS_this
                        SS_best_left = SS_left
                        SS_best_right = SS_right
                        iFeature_best = iFeature
                        val_best = val
            best_split = [iFeature_best, val_best, SS_pre_split, SS_best_left, SS_best_right, len(y), node_index_v[0], iDepth, y_boot]
            best_split_feature_vec.append(iFeature_best)
        # print(iDepth, best_split)
        # Pick most-voted iFeature and calculate other params on full dataset
        values, counts = np.unique(best_split_feature_vec, return_counts=True)
        iFeature = values[np.argmax(counts)]
        splitting_var = X[:, iFeature]
        SS_best = np.inf
        for val in splitting_var:
            ind_left = (splitting_var < val)
            ind_right = (splitting_var >= val)
            SS_left = f_SS(y[ind_left])
            SS_right = f_SS(y[ind_right])
            SS_this = SS_left + SS_right
            if (SS_this < SS_best):
                SS_best = SS_this
                SS_best_left = SS_left
                SS_best_right = SS_right
                iFeature_best = iFeature
                val_best = val
        best_split = [iFeature_best, val_best, SS_pre_split, SS_best_left, SS_best_right, len(y), node_index_v[0], iDepth, y]
        ind_left = (X[:, iFeature_best] < val_best)
        ind_right = (X[:, iFeature_best] >= val_best)
        branch_left = teg_tree_scale(X[ind_left, :], y[ind_left], maxDepth, iDepth + 1)
        branch_right = teg_tree_scale(X[ind_right, :], y[ind_right], maxDepth, iDepth + 1)
        return [best_split, branch_left, branch_right]

    def teg_tree_scale_XOR_trap(X, y, maxDepth, iDepth=0, node_index_v = [0]):
        if (iDepth == 0):
            node_index_v[0] = 0
        else:
            node_index_v[0] = node_index_v[0] + 1
        #print(node_index_v)
        SS_pre_split = f_SS(y)
        # Check whether maxdepth passed or y is empty
        if (iDepth >= maxDepth) or (len(y) <= 1) or (SS_pre_split== 0):
            terminal_node_pred = np.nanmean(y)
            return [[np.NaN, terminal_node_pred, SS_pre_split, 0, 0, 0, node_index_v[0], iDepth, y], np.NaN, np.NaN]
        # Create branches
        SS_best = np.inf
        SS_best_left = np.inf
        SS_best_right = np.inf
        iFeature_best = np.NaN
        val_best = np.NaN
        # Check one step ahead
        for iFeature1 in range(X.shape[1]):
            #print(iFeature1)
            splitting_var1 = X[:, iFeature1]
            splitting_vals1 = np.unique(splitting_var1) # np.quantile(splitting_var1, [.2, .35, .5, .65, .8])
            for val1 in splitting_vals1:
                for iFeature2 in range(X.shape[1]):
                    if (iFeature1 == iFeature2):
                        continue
                    splitting_var2 = X[:, iFeature2]
                    splitting_vals2 = np.quantile(splitting_var2, [.25, .5, .75])
                    for val2 in splitting_vals2:
                        #print(iFeature1, ' ', iFeature2, ' ', val1, ' ', val2, '\n')
                        ind_left_left = ((splitting_var1 < val1) & (splitting_var2 < val2))
                        ind_left_right = ((splitting_var1 < val1) & (splitting_var2 >= val2))
                        ind_right_left = ((splitting_var1 >= val1) & (splitting_var2 < val2))
                        ind_right_right = ((splitting_var1 >= val1) & (splitting_var2 >= val2))
                        SS_left_left = f_SS(y[ind_left_left])
                        SS_left_right = f_SS(y[ind_left_right])
                        SS_right_left = f_SS(y[ind_right_left])
                        SS_right_right = f_SS(y[ind_right_right])
                        SS_this = SS_left_left + SS_left_right + SS_right_left + SS_right_right
                        if (SS_this < SS_best):
                            # Use double-split for best
                            SS_best = SS_this
                            # Use first split
                            ind_left = (splitting_var1 < val1)
                            ind_right = (splitting_var1 >= val1)
                            SS_left = f_SS(y[ind_left])
                            SS_right = f_SS(y[ind_right])
                            SS_this = SS_left + SS_right
                            SS_best_left = SS_left
                            SS_best_right = SS_right
                            iFeature_best = iFeature1
                            val_best = val1
                            #print('New best: ', iFeature1, iFeature2, SS_best)
        best_split = [iFeature_best, val_best, SS_pre_split, SS_best_left, SS_best_right, len(y), node_index_v[0], iDepth, y]
        # print(iDepth, best_split)
        ind_left = (X[:, iFeature_best] < val_best)
        ind_right = (X[:, iFeature_best] >= val_best)
        branch_left = teg_tree_scale(X[ind_left, :], y[ind_left], maxDepth, iDepth + 1)
        branch_right = teg_tree_scale(X[ind_right, :], y[ind_right], maxDepth, iDepth + 1)
        return [best_split, branch_left, branch_right]

    def teg_tree_scale_XOR_trap_internalEnsemble(X, y, maxDepth, iDepth=0, node_index_v = [0]):
        if (iDepth == 0):
            node_index_v[0] = 0
        else:
            node_index_v[0] = node_index_v[0] + 1
        #print(node_index_v)
        SS_pre_split = f_SS(y)
        # Check whether maxdepth passed or y is empty
        if (iDepth >= maxDepth) or (len(y) <= 1) or (SS_pre_split== 0):
            terminal_node_pred = np.nanmean(y)
            return [[np.NaN, terminal_node_pred, SS_pre_split, 0, 0, 0, node_index_v[0], iDepth, y], np.NaN, np.NaN]
        # Create branches
        # Repeat for bootstrapped samples, pick top vote for this split
        nSamples = 3
        best_split_feature_vec = []
        for iSample in range(nSamples):
            y_boot = np.random.choice(y, size=len(y))
            SS_best = np.inf
            SS_best_left = np.inf
            SS_best_right = np.inf
            iFeature_best = np.NaN
            val_best = np.NaN
            # Check one step ahead
            for iFeature1 in range(X.shape[1]):
                #print(iFeature1)
                splitting_var1 = X[:, iFeature1]
                splitting_vals1 = np.unique(splitting_var1) # np.quantile(splitting_var1, [.2, .35, .5, .65, .8])
                for val1 in splitting_vals1:
                    for iFeature2 in range(X.shape[1]):
                        if (iFeature1 == iFeature2):
                            continue
                        splitting_var2 = X[:, iFeature2]
                        splitting_vals2 = np.quantile(splitting_var2, [.25, .5, .75])
                        for val2 in splitting_vals2:
                            #print(iFeature1, ' ', iFeature2, ' ', val1, ' ', val2, '\n')
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
                                SS_left = f_SS(y[ind_left])
                                SS_right = f_SS(y[ind_right])
                                SS_this = SS_left + SS_right
                                SS_best_left = SS_left
                                SS_best_right = SS_right
                                iFeature_best = iFeature1
                                val_best = val1
                                #print('New best: ', iFeature1, iFeature2, SS_best)
            best_split = [iFeature_best, val_best, SS_pre_split, SS_best_left, SS_best_right, len(y_boot), node_index_v[0], iDepth, y_boot]
            best_split_feature_vec.append(iFeature_best)
        # print(iDepth, best_split)
        # Pick most-voted iFeature and calculate other params on full dataset
        values, counts = np.unique(best_split_feature_vec, return_counts=True)
        iFeature1 = values[np.argmax(counts)]
        splitting_var1 = X[:, iFeature1]
        splitting_vals1 = np.unique(splitting_var1)  # np.quantile(splitting_var1, [.2, .35, .5, .65, .8])
        for val1 in splitting_vals1:
            for iFeature2 in range(X.shape[1]):
                if (iFeature1 == iFeature2):
                    continue
                splitting_var2 = X[:, iFeature2]
                splitting_vals2 = np.quantile(splitting_var2, [.25, .5, .75])
                for val2 in splitting_vals2:
                    # print(iFeature1, ' ', iFeature2, ' ', val1, ' ', val2, '\n')
                    ind_left_left = ((splitting_var1 < val1) & (splitting_var2 < val2))
                    ind_left_right = ((splitting_var1 < val1) & (splitting_var2 >= val2))
                    ind_right_left = ((splitting_var1 >= val1) & (splitting_var2 < val2))
                    ind_right_right = ((splitting_var1 >= val1) & (splitting_var2 >= val2))
                    SS_left_left = f_SS(y[ind_left_left])
                    SS_left_right = f_SS(y[ind_left_right])
                    SS_right_left = f_SS(y[ind_right_left])
                    SS_right_right = f_SS(y[ind_right_right])
                    SS_this = SS_left_left + SS_left_right + SS_right_left + SS_right_right
                    if (SS_this < SS_best):
                        # Use double-split for best
                        SS_best = SS_this
                        # Use first split
                        ind_left = (splitting_var1 < val1)
                        ind_right = (splitting_var1 >= val1)
                        SS_left = f_SS(y[ind_left])
                        SS_right = f_SS(y[ind_right])
                        SS_this = SS_left + SS_right
                        SS_best_left = SS_left
                        SS_best_right = SS_right
                        iFeature_best = iFeature1
                        val_best = val1
                        # print('New best: ', iFeature1, iFeature2, SS_best)
        best_split = [iFeature_best, val_best, SS_pre_split, SS_best_left, SS_best_right, len(y_boot), node_index_v[0], iDepth, y]
        ind_left = (X[:, iFeature_best] < val_best)
        ind_right = (X[:, iFeature_best] >= val_best)
        branch_left = teg_tree_scale(X[ind_left, :], y[ind_left], maxDepth, iDepth + 1)
        branch_right = teg_tree_scale(X[ind_right, :], y[ind_right], maxDepth, iDepth + 1)
        return [best_split, branch_left, branch_right]

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
        best_collapse_seq_end = np.argmin(C)
        nodes_collapsed_choice = nodes_collapsed[0:(best_collapse_seq_end + 1)]
        return build_tree_inner(this_tree, nodes_collapsed_choice, mean_y, sd_y)

    mean_y = np.nanmean(y)
    sd_y = np.sqrt(np.var(y))
    y = (y - mean_y) / sd_y
    if (twostep == 0):
        if (internalEnsemble == 0):
            tree0 = teg_tree_scale(X, y, maxDepth, alpha0)
        else:
            tree0 = teg_tree_scale_internalEnsemble(X, y, maxDepth, alpha0)
    else:
        if (internalEnsemble == 0):
            tree0 = teg_tree_scale_XOR_trap(X, y, maxDepth, alpha0)
        else:
            tree0 = teg_tree_scale_XOR_trap_internalEnsemble(X, y, maxDepth, alpha0)
    C, nodes_collapsed = prune_the_tree(tree0, alpha0)
    #print(tree0)
    #print(C)
    #print(nodes_collapsed)
    print_tree(tree0, C, nodes_collapsed, mean_y, sd_y)
    collapsed_tree = collapse_tree(tree0, C, nodes_collapsed, mean_y, sd_y)
    return collapsed_tree, min(C)

nObs = 2000
nPred = 6
maxDepth = 4 # Max. number of splits
alpha0 = 0.5
X = np.random.random_sample(size=(nObs, nPred))
y = 0.1 * np.random.random_sample(size=(nObs))
LogicalInd = (X[:, 1] > 0.8) & (X[:, 2] < 0.33) & (X[:, 4] < 0.5)
y[LogicalInd] = 1 - (1 - y[LogicalInd]) * 0.25
# Traditional greedy tree
tree0, cost_complexity_criterion = teg_regression_tree(X, y, maxDepth, alpha0, twostep=0, internalEnsemble = 0)
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

# XOR problem
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
