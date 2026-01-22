from fastapi import FastAPI
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

app = FastAPI()

reference = pd.read_csv("reference_data.csv")
current = pd.read_csv("current_data.csv")

@app.get("/drift")
def check_drift():
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()

    return {
        "dataset_drift": result["metrics"][0]["result"]["dataset_drift"],
        "share_of_drifted_features": result["metrics"][0]["result"]["share_of_drifted_columns"],
        "drifted_features": [
            k for k, v in result["metrics"][0]["result"]["drift_by_columns"].items()
            if v["drift_detected"]
        ]
    }
