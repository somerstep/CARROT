import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from constants import *

def flatten(xss):
    return [x for xs in xss for x in xs]
    
def route(scores, cost, cost_pred, correctness, lamb_range = np.arange(0, 1.001, 0.001)):
    router_cost = np.zeros(shape = (scores.shape[0], lamb_range.shape[0]))
    router_perf = np.zeros_like(router_cost)
    for idx_lam, lam in enumerate(lamb_range):
        model_idx = ((1 - lam) * scores - lam * cost_pred).argmax(axis = 1, keepdims = True)
        router_perf[:, idx_lam] = np.take_along_axis(correctness, model_idx, axis = 1).reshape((-1))
        router_cost[:, idx_lam] = np.take_along_axis(cost, model_idx, axis = 1).reshape((-1))
    return router_cost.mean(0), router_perf.mean(0)

def route_lamb(scores, cost, cost_pred, correctness):
    router_cost = np.zeros(shape = (scores.shape[0],1))
    router_perf = np.zeros_like(router_cost)
    model_idx = scores.argmax(axis = 1, keepdims = True)   
    router_perf = np.take_along_axis(correctness, model_idx, axis = 1).reshape((-1))
    router_cost = np.take_along_axis(cost, model_idx, axis = 1).reshape((-1))
    return router_cost.mean(0), router_perf.mean(0)
    
def route_routerbench(scores, cost, correctness, lamb_range=np.arange(0, 1.001, 0.001)):
    """Routerbench rule: same argmax routing as `route`, but with static per-model
    average cost (computed from `cost`) instead of per-prompt predicted cost.
    Consumes the same `scores` (accuracy predictions) as CARROT uses."""
    avg_cost = cost.mean(axis=0, keepdims=True)
    cost_pred = np.broadcast_to(avg_cost, cost.shape)
    return route(scores, cost, cost_pred, correctness, lamb_range=lamb_range)


def route_pairwise(predictions, cost, correctness, large_model_ind, small_model_ind, alpha_range=None):
    """Sweep a threshold over `predictions` to interpolate between the small and large
    model. The curve's endpoints are the pure small-model and pure large-model points,
    so the sweep must span the full prediction range, not assume [0, 1]."""
    if alpha_range is None:
        lo = float(np.min(predictions)) - 1e-6
        hi = float(np.max(predictions)) + 1e-6
        alpha_range = np.linspace(lo, hi, 1001)
    perfs, costs = [], []
    for alpha in alpha_range:
        ind_small = predictions >= alpha
        ind_large = predictions < alpha
        perfs.append((correctness[ind_small, small_model_ind].sum() + correctness[ind_large, large_model_ind].sum()) / correctness.shape[0])
        costs.append((cost[ind_small, small_model_ind].sum() + cost[ind_large, large_model_ind].sum()) / cost.shape[0])
    return np.array(costs), np.array(perfs)
    
