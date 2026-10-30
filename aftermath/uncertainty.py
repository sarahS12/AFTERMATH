"""Seeded hidden execution dynamics, distinct from nominal planning transitions."""
from dataclasses import dataclass, replace
import random
from .core import State, transition, loss

@dataclass(frozen=True)
class HiddenConditions:
    restart_surge: float = 1.0
    queue_db_cost: float = 1.0
    capacity_effect: float = 1.0
    background_disturbance: float = 0.0


def hidden_conditions(seed: int = 42, intensity: float = 1.0) -> HiddenConditions:
    """Fixed hidden parameters per incident; deterministic given seed.

    Underlying execution physics can differ from the planner model.
    """
    rng = random.Random(int(seed))
    z = max(0.0, min(2.0, float(intensity)))
    return HiddenConditions(
        restart_surge=1.0 + z * rng.uniform(0.3, 1.25),
        queue_db_cost=1.0 + z * rng.uniform(0.0, 1.2),
        capacity_effect=max(0.25, 1.0 - z * rng.uniform(0.0, 0.5)),
        background_disturbance=z * rng.uniform(0, 6.0),
    )


def actual_transition(s: State, action: str, hidden: HiddenConditions, seed: int = 0) -> State:
    """Execute action using hidden dynamics. Planner cannot read hidden coefficients.

    No unpredictable network or production activity is ever performed.
    """
    next_s = transition(s, action)
    if action == 'restart_payments':
        surge = s.retry_backlog * (0.12 if s.retry_throttled else 1.15)
        next_s.db_load += surge * (hidden.restart_surge - 1.0)
    elif action == 'clear_queue':
        next_s.db_load += 0.65 * s.queue_backlog * (hidden.queue_db_cost - 1.0)
    elif action == 'expand_db_pool':
        next_s.db_capacity += 65.0 * (hidden.capacity_effect - 1.0)
    elif action == 'wait':
        next_s.db_load += hidden.background_disturbance
    rng = random.Random(int(seed) * 9176 + int(s.steps) * 67 + sum(map(ord, action)))
    next_s.db_load = max(10.0, next_s.db_load + rng.uniform(-1, 1) * hidden.background_disturbance / 4)
    return next_s


def prediction_error(s: State, action: str, actual: State):
    predicted = transition(s, action)
    return {
        'predicted_loss': loss(predicted),
        'actual_loss': loss(actual),
        'prediction_error': round(loss(actual) - loss(predicted), 2),
        'predicted_db_util': round(predicted.db_load / predicted.db_capacity * 100, 1),
        'actual_db_util': round(actual.db_load / actual.db_capacity * 100, 1),
    }
