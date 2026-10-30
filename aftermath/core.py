"""Deterministic digital twin. All figures are synthetic illustrative units.

This is a sandbox, not a model of any real production service.
"""
from dataclasses import dataclass, replace, asdict
from typing import Dict, List

SCENARIOS = ("retry_storm", "db_saturation", "auth_and_queue", "healthy", "mixed_failure")

ACTIONS = ["restart_payments", "throttle_retries", "expand_db_pool", "clear_queue", "restore_auth_config", "wait"]

@dataclass
class State:
    db_capacity: float = 100.0
    db_load: float = 80.0
    payment_load: float = 110.0
    retry_backlog: float = 75.0
    queue_backlog: float = 55.0
    auth_broken: bool = False
    payment_running: bool = True
    retry_throttled: bool = False
    steps: int = 0
    scenario: str = "retry_storm"

def initial_state(scenario: str) -> State:
    options = {
        "retry_storm": State(scenario=scenario),
        "db_saturation": State(db_capacity=75, db_load=120, payment_load=85, retry_backlog=35, queue_backlog=40, scenario=scenario),
        "auth_and_queue": State(db_load=45, payment_load=55, retry_backlog=10, queue_backlog=90, auth_broken=True, scenario=scenario),
        "healthy": State(db_load=40, payment_load=45, retry_backlog=0, queue_backlog=5, scenario=scenario),
        "mixed_failure": State(db_capacity=80, db_load=110, payment_load=115, retry_backlog=85, queue_backlog=80, auth_broken=True, scenario=scenario),
    }
    if scenario not in options:
        raise ValueError(f"Unknown scenario: {scenario}")
    return replace(options[scenario])

def metrics(s: State) -> Dict[str, float]:
    db_overload = max(0, s.db_load / s.db_capacity - 1)
    payment_overload = max(0, s.payment_load / 100 - 1)
    errors = min(100.0, 10 + 70 * db_overload + 45 * payment_overload + (35 if s.auth_broken else 0) + (20 if not s.payment_running else 0))
    latency = 120 + 800 * db_overload + 550 * payment_overload + 2.5 * s.queue_backlog
    return {"error_rate_pct": round(errors, 1), "latency_ms": round(latency, 1),
            "db_util_pct": round(100 * s.db_load / s.db_capacity, 1),
            "payment_util_pct": round(s.payment_load, 1), "queue": round(s.queue_backlog, 1),
            "retry_backlog": round(s.retry_backlog, 1), "auth_broken": s.auth_broken}

def loss(s: State) -> float:
    m = metrics(s)
    return round(2.5 * m["error_rate_pct"] + 0.06 * m["latency_ms"] + 0.8 * m["queue"]
                 + (80 if s.auth_broken else 0) + (40 if not s.payment_running else 0), 2)

def recovered(s: State) -> bool:
    m = metrics(s)
    return (m["error_rate_pct"] <= 12 and m["latency_ms"] <= 330 and
            m["queue"] <= 30 and not s.auth_broken and s.payment_running)

def transition(state: State, action: str) -> State:
    if action not in ACTIONS:
        raise ValueError(f"Action not allowed: {action}")
    s = replace(state)
    s.steps += 1
    if action == "restart_payments":
        s.payment_running = True
        s.payment_load = max(35, s.payment_load - 45)
       
        surge = s.retry_backlog * (0.12 if s.retry_throttled else 1.15)

        s.db_load += surge
        s.retry_backlog = max(0, s.retry_backlog - 35)
        s.queue_backlog += 12

    elif action == "throttle_retries":
        s.retry_throttled = True
        s.retry_backlog = max(0, s.retry_backlog - 40)
        s.payment_load = max(30, s.payment_load - 30)
        s.db_load = max(20, s.db_load - 25)
    elif action == "expand_db_pool":
        s.db_capacity += 65
        s.db_load += 8  
    elif action == "clear_queue":
        
        s.db_load += 0.65 * s.queue_backlog
        s.queue_backlog = max(0, s.queue_backlog - 70)
    elif action == "restore_auth_config":
        s.auth_broken = False
        s.payment_load += 5
    elif action == "wait":
        s.retry_backlog = max(0, s.retry_backlog - (20 if s.retry_throttled else 5))
        s.queue_backlog = max(0, s.queue_backlog - 10)
        s.db_load = max(20, s.db_load - 12)
        s.payment_load = max(25, s.payment_load - 8)
    # Background work settles at each step.
    s.db_load = max(15, s.db_load - 5)
    s.payment_load = max(20, s.payment_load - 4)
    s.queue_backlog = max(0, s.queue_backlog - 4)
    return s

def observations(s: State) -> Dict:
    m = metrics(s)
    logs = []
    if m["db_util_pct"] > 100: logs.append("database: connection pool saturation; request timeouts")
    if m["payment_util_pct"] > 100: logs.append("payments: high concurrency; retry amplification")
    if s.auth_broken: logs.append("auth: token validation failed; configuration mismatch")
    if s.queue_backlog > 50: logs.append("notifications: consumer lag; pending jobs rising")
    if not logs: logs.append("services: nominal or stabilizing")
    return {"metrics": m, "logs": logs, "dependencies": ["auth -> payments", "payments -> database", "payments -> notifications"]}

def rollout(s: State, actions: List[str]) -> Dict:
    states = [asdict(s)]
    losses = [loss(s)]
    for action in actions:
        s = transition(s, action)
        states.append(asdict(s))
        losses.append(loss(s))
    return {"actions": actions, "states": states, "losses": losses,
            "final_loss": losses[-1], "cumulative_loss": round(sum(losses[1:]), 2),
            "recovered": recovered(s)}
