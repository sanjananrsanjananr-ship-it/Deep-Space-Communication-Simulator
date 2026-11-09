import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from src.communication_engine import link_metrics

def train_evaluate(seed=42, samples=400):
    """Train/test an educational model using analytical link-model generated data."""
    x = np.linspace(500.0, 800000.0, samples).reshape(-1, 1)
    y = np.array([link_metrics(float(d))["packet_loss"] * 100 for d in x[:, 0]])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=seed)
    model = RandomForestRegressor(n_estimators=80, random_state=seed, n_jobs=1).fit(x_train, y_train)
    pred = model.predict(x_test)
    return model, {"train_samples": len(x_train), "test_samples": len(x_test),
                   "mae": float(mean_absolute_error(y_test, pred)), "r2": float(r2_score(y_test, pred)),
                   "source": "analytical model-derived data; not measured spacecraft telemetry"}
