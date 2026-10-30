"""Anomaly detection and action-impact regression on synthetic telemetry."""
from functools import lru_cache
import numpy as np
from sklearn.ensemble import IsolationForest
from .core import State, metrics

FEATURES = ('error_rate_pct', 'latency_ms', 'db_util_pct', 'payment_util_pct', 'queue', 'retry_backlog')


def vector(s: State):
    m = metrics(s)
    return [float(m[k]) for k in FEATURES]


@lru_cache(maxsize=1)
def trained_detector():
    rng = np.random.default_rng(29)
    training = []
    for _ in range(700):
        normal = State(
            db_capacity=float(rng.uniform(90, 140)),
            db_load=float(rng.uniform(30, 78)),
            payment_load=float(rng.uniform(25, 82)),
            retry_backlog=float(rng.uniform(0, 12)),
            queue_backlog=float(rng.uniform(0, 15)),
            auth_broken=False,
            payment_running=True,
        )
        training.append(vector(normal))
    model = IsolationForest(n_estimators=120, contamination=0.035, random_state=29)
    model.fit(np.asarray(training))
    return model


def assess(s: State):
    model = trained_detector()
    sample = np.asarray([vector(s)])
    return {
        'model': 'IsolationForest (scikit-learn; trained on synthetic nominal telemetry)',
        'anomaly': bool(model.predict(sample)[0] == -1),
        'anomaly_score': round(float(model.decision_function(sample)[0]), 4),
        'features': {name: value for name, value in zip(FEATURES, vector(s))},
    }



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from .core import ACTIONS, transition, loss


def _action_features(s: State, action: str):
    return [float(s.db_capacity), float(s.db_load), float(s.payment_load),
            float(s.retry_backlog), float(s.queue_backlog), float(s.auth_broken),
            float(s.payment_running), float(s.retry_throttled)] + [float(a == action) for a in ACTIONS]


@lru_cache(maxsize=1)
def trained_impact_predictor():
    """Fit a regressor and report holdout error on synthetic transitions."""
    rng = np.random.default_rng(17)
    X,y=[],[]
    for _ in range(2200):
        state=State(db_capacity=float(rng.uniform(60,180)),
             db_load=float(rng.uniform(20,190)),payment_load=float(rng.uniform(20,180)),
             retry_backlog=float(rng.uniform(0,160)),queue_backlog=float(rng.uniform(0,140)),
             auth_broken=bool(rng.uniform()<.25),payment_running=bool(rng.uniform()>.15),
             retry_throttled=bool(rng.uniform()<.3))
        action=ACTIONS[int(rng.integers(0,len(ACTIONS)))]
        X.append(_action_features(state,action))
        y.append(loss(transition(state,action)))
    X=np.asarray(X);y=np.asarray(y)
    cut=1800
    reg=RandomForestRegressor(n_estimators=75,min_samples_leaf=2,random_state=17,n_jobs=-1)
    reg.fit(X[:cut],y[:cut])
    val=reg.predict(X[cut:])
    return reg,round(float(mean_absolute_error(y[cut:],val)),2)


def predict_action_loss(s: State, action: str):
    reg, _ = trained_impact_predictor()
    return round(float(reg.predict(np.asarray([_action_features(s,action)]))[0]),2)


def impact_model_summary():
    _,mae=trained_impact_predictor()
    return {'model':'RandomForestRegressor (trained on synthetic action outcomes)',
            'synthetic_holdout_MAE':mae, 'units':'synthetic loss points'}
