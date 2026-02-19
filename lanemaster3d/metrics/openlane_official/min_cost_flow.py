from __future__ import annotations

import numpy as np

try:
    from ortools.graph import pywrapgraph

    _ORTOOLS_OK = True
except Exception:
    pywrapgraph = None
    _ORTOOLS_OK = False

from scipy.optimize import linear_sum_assignment


def _solve_with_ortools(adj_mat: np.ndarray, cost_mat: np.ndarray) -> list[list[int]]:
    solver = pywrapgraph.SimpleMinCostFlow()
    cnt_1, cnt_2 = adj_mat.shape
    nonzero_row = int(np.sum(np.sum(adj_mat, axis=1) > 0))
    nonzero_col = int(np.sum(np.sum(adj_mat, axis=0) > 0))
    start_nodes = [0] * cnt_1 + np.repeat(np.arange(1, cnt_1 + 1), cnt_2).tolist()
    start_nodes += [i for i in range(cnt_1 + 1, cnt_1 + cnt_2 + 1)]
    end_nodes = [i for i in range(1, cnt_1 + 1)]
    end_nodes += np.repeat(np.arange(cnt_1 + 1, cnt_1 + cnt_2 + 1)[None], cnt_1, axis=0).ravel().tolist()
    end_nodes += [cnt_1 + cnt_2 + 1] * cnt_2
    capacities = [1] * cnt_1 + adj_mat.astype(np.int64).ravel().tolist() + [1] * cnt_2
    costs = [0] * cnt_1 + cost_mat.astype(np.int64).ravel().tolist() + [0] * cnt_2
    flow = min(nonzero_row, nonzero_col)
    supplies = [flow] + [0] * (cnt_1 + cnt_2) + [-flow]
    for i in range(len(start_nodes)):
        solver.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i], capacities[i], costs[i])
    for i, supply in enumerate(supplies):
        solver.SetNodeSupply(i, supply)
    if solver.Solve() != solver.OPTIMAL:
        return []
    source = 0
    sink = cnt_1 + cnt_2 + 1
    matches: list[list[int]] = []
    for arc in range(solver.NumArcs()):
        if solver.Tail(arc) == source or solver.Head(arc) == sink:
            continue
        if solver.Flow(arc) > 0:
            matches.append([solver.Tail(arc) - 1, solver.Head(arc) - cnt_1 - 1, solver.UnitCost(arc)])
    return matches


def _solve_with_hungarian(adj_mat: np.ndarray, cost_mat: np.ndarray) -> list[list[int]]:
    valid_rows = np.where(adj_mat.sum(axis=1) > 0)[0]
    valid_cols = np.where(adj_mat.sum(axis=0) > 0)[0]
    if valid_rows.size == 0 or valid_cols.size == 0:
        return []
    sub_adj = adj_mat[np.ix_(valid_rows, valid_cols)]
    sub_cost = cost_mat[np.ix_(valid_rows, valid_cols)].astype(np.float64)
    invalid_cost = sub_cost.max() + 10_000.0 if sub_cost.size > 0 else 10_000.0
    sub_cost[sub_adj <= 0] = invalid_cost
    rows, cols = linear_sum_assignment(sub_cost)
    matches: list[list[int]] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        if sub_adj[r, c] <= 0:
            continue
        matches.append([int(valid_rows[r]), int(valid_cols[c]), int(cost_mat[valid_rows[r], valid_cols[c]])])
    return matches


def solve_min_cost_flow(adj_mat: np.ndarray, cost_mat: np.ndarray) -> list[list[int]]:
    if adj_mat.size == 0:
        return []
    if _ORTOOLS_OK:
        return _solve_with_ortools(adj_mat, cost_mat)
    return _solve_with_hungarian(adj_mat, cost_mat)
