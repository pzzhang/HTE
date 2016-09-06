import numpy as np


def grouping(values, Adj, tol):
    # first, do not consider the topology, group points by values
    ind_sorted = np.argsort(values)
    vals_sorted = values[ind_sorted]
    clusters = dict()
    nstart = 0
    ncurr = nstart
    while ncurr < len(values):
        if ncurr + 1 == len(values) or vals_sorted[ncurr + 1] - vals_sorted[ncurr] > tol:
            clusters[(vals_sorted[nstart], vals_sorted[ncurr])] = ind_sorted[nstart:ncurr + 1]
            nstart = ncurr + 1
            ncurr += 1
        else:
            ncurr += 1
    # splitting each cluster into groups based on the topology given by Adj
    # this problem is: given a cluster of points in a graph, group them by connected components
    groups = []
    for val, cluster in clusters.iteritems():
        if val[0] > tol or val[1] < -tol:
            # only count the non-zero clusters
            Adj_loc = Adj[np.ix_(cluster, cluster)]
            groups_loc = getComponents(len(cluster), Adj_loc)
            groups += [cluster[group_loc] for group_loc in groups_loc]
    # construct the design matrix
    U = np.zeros((len(values), len(groups)))
    for groupid, group in enumerate(groups):
        U[group, groupid] = 1.0
    return U


def getComponents(n, A):
    # https://discuss.leetcode.com/topic/32677/short-union-find-in-python-ruby-c
    # n is the number of nodes, A is the (symmetric) adjacency matrix
    p = range(n)

    def find(v):
        if p[v] != v:
            p[v] = find(p[v])
        return p[v]

    for v in range(n):
        for w in range(v+1, n):
            if A[v, w]:
                p[find(v)] = find(w)
    p = map(find, p)
    # get the groups
    groups = dict()
    for nodeid, nodeparent in enumerate(p):
        if nodeparent in groups:
            groups[nodeparent].append(nodeid)
        else:
            groups[nodeparent] = [nodeid]
    return groups.values()
