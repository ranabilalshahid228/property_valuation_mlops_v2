import os, pandas as pd, numpy as np
from utils import population_stability_index
from config import CONFIG

def load_recent_predictions(path: str = "logs/recent_predictions.csv") -> pd.DataFrame:
    if not os.path.exists(path): 
        return pd.DataFrame(columns=["timestamp","property_id","predicted_value","confidence","location"])
    df = pd.read_csv(path)
    return df

def compute_perf_summary() -> dict:
    perf_csv = "reports/model_performance.csv"
    if os.path.exists(perf_csv):
        mp = pd.read_csv(perf_csv)
        r2_best = mp["R2"].max()
        mae_best = mp["MAE"].min()
    else:
        r2_best, mae_best = 0.0, 0.0
    recents = load_recent_predictions()
    avg_latency = 0
    throughput = 0.0
    if len(recents)>0:
        if 'latency_ms' in recents.columns:
            avg_latency = int(recents['latency_ms'].tail(200).mean())
        recents['timestamp'] = pd.to_datetime(recents['timestamp'])
        cutoff = recents['timestamp'].max() - pd.Timedelta(minutes=15)
        throughput = len(recents[recents['timestamp']>=cutoff]) / 15.0
    kpis = {
        "models_trained": 3 if os.path.exists(perf_csv) else 0,
        "active_model_accuracy": round(r2_best,2),
        "predictions_made": int(len(recents)),
        "data_drift_score": round(drift_score(recents),2) if len(recents)>0 else 0.0
    ,
        "avg_latency_ms": int(avg_latency),
        "throughput_per_min": round(float(throughput),2)
    }
    return kpis

def drift_score(recents: pd.DataFrame) -> float:
    # compare predicted_value distribution over last 5 days vs earlier
    if len(recents) < 20:
        return 0.0
    recents["timestamp"] = pd.to_datetime(recents["timestamp"])
    cutoff = recents["timestamp"].max() - pd.Timedelta(days=5)
    new = recents.loc[recents["timestamp"] >= cutoff, "predicted_value"].values
    old = recents.loc[recents["timestamp"] < cutoff, "predicted_value"].values
    return population_stability_index(old, new, bins=10)
