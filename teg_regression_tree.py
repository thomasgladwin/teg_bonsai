import numpy as np

#
# Decision tree: Regression
#

def teg_regression_tree(X, y, maxDepth, alpha0):

    def teg_tree_scale(X, y, maxDepth, iDepth=0, node_index_v = [0]):
        if (iDepth == 0):
            node_index_v[0] = 0
        else:
            node_index_v[0] = node_index_v[0] + 1
        print(node_index_v)
        SS_pre_split = f_SS(y)
        # Check whether maxdepth passed or y is empty
        if (iDepth >= maxDepth) or (len(y) == 0) or (SS_pre_split== 0):
            return [[np.NaN, np.NaN, SS_pre_split, 0, 0, 0, node_index_v[0]], np.NaN, np.NaN]
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
        best_split = [iFeature_best, val_best, SS_pre_split, SS_best_left, SS_best_right, len(y), node_index_v[0]]
        # print(iDepth, best_split)
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
        print(nodes_to_collapse_tmp, this_tree[0][6], nodes_to_collapse_tmp.count(this_tree[0][6]))
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
        print('zz', nodes_to_collapse_tmp)
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
        print(node_indices)
        uncollapsed_v = [1 for a in node_indices]
        nodes_collapsed = []
        C = []
        while sum(uncollapsed_v) > 0:
            print('x')
            C_vec_tmp = []
            iNode_indices_tmp = []
            iiNode_indices_tmp = []
            print(node_indices)
            print(uncollapsed_v)
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
            print(iiNode_indices_tmp)
            print(iNode_indices_tmp)
            iiiNode_to_collapse = np.argmin(C_vec_tmp)
            iiNode_to_collapse = iiNode_indices_tmp[iiiNode_to_collapse]
            iNode_to_collapse = iNode_indices_tmp[iiiNode_to_collapse]
            ndf = get_downstream_nodes(this_tree, iNode_to_collapse)
            for intc in ndf: # iNodeToCollapse, includes source-collapser
                print(intc)
                for ii in range(len(node_indices)):
                    print(ii)
                    if intc == node_indices[ii]:
                        uncollapsed_v[ii] = 0
            nodes_collapsed.append(iNode_to_collapse)
            C.append(min(C_vec_tmp))
            print(iiNode_to_collapse, iNode_to_collapse)
            # Collapse all downstream internal nodes
        return C, nodes_collapsed

    def print_tree(this_tree, C, nodes_collapsed):
        def print_tree_inner(this_tree, nodes_collapsed_choice):
            if (nodes_collapsed_choice.count(this_tree[0][6]) == 0):
                print(this_tree[0][0:2])
                print_tree_inner(this_tree[1], nodes_collapsed_choice)
                print_tree_inner(this_tree[2], nodes_collapsed_choice)
        best_collapse_seq_end = np.argmin(C)
        nodes_collapsed_choice = nodes_collapsed[0:(best_collapse_seq_end + 1)]
        print_tree_inner(this_tree, nodes_collapsed_choice)

    def collapse_tree(this_tree, C, nodes_collapsed):
        def build_tree_inner(this_tree, nodes_collapsed_choice):
            if (nodes_collapsed_choice.count(this_tree[0][6]) == 0):
                return [this_tree[0][0:2], build_tree_inner(this_tree[1], nodes_collapsed_choice), build_tree_inner(this_tree[2], nodes_collapsed_choice)]
        best_collapse_seq_end = np.argmin(C)
        nodes_collapsed_choice = nodes_collapsed[0:(best_collapse_seq_end + 1)]
        return build_tree_inner(this_tree, nodes_collapsed_choice)

    tree0 = teg_tree_scale(X, y, maxDepth, alpha0)
    C, nodes_collapsed = prune_the_tree(tree0, alpha0)
    print_tree(tree0, C, nodes_collapsed)
    collapsed_tree = collapse_tree(tree0, C, nodes_collapsed)
    return collapsed_tree, min(C)

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
