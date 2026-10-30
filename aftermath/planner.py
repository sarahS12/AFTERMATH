"""Bounded counterfactual search and a reactive baseline policy."""
from itertools import product
from .core import ACTIONS, State, recovered, rollout, observations

ACTION_COST = {"restart_payments": 12, "throttle_retries": 7, "expand_db_pool": 18,
               "clear_queue": 9, "restore_auth_config": 5, "wait": 2}

def score(s: State, actions):
    result = rollout(s, list(actions))

    peak = max(result["losses"][1:])
    return round(result["cumulative_loss"] + 0.5 * peak +
                 sum(ACTION_COST[a] for a in actions) - (180 if result["recovered"] else 0), 2)

def plan(s: State, horizon: int = 3, top_k: int = 5):
    if recovered(s):
        return [{"actions": [], "score": 0, **rollout(s, [])}]

    candidates = []
    for actions in product(ACTIONS, repeat=horizon):
        if actions.count("restore_auth_config") > 1 or actions.count("expand_db_pool") > 1:
            continue
        outcome = rollout(s, list(actions))
        candidates.append({"score": score(s, actions), **outcome})
    return sorted(candidates, key=lambda x: x["score"])[:top_k]

def baseline_action(s: State):
    """Simple reactive heuristic; not an LLM baseline."""
    m = observations(s)["metrics"]
    if s.auth_broken: return "restore_auth_config"
    if m["payment_util_pct"] > 100: return "restart_payments"
    if m["db_util_pct"] > 100: return "expand_db_pool"
    if m["queue"] > 30: return "clear_queue"
    return "wait"
