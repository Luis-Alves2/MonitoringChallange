"""Endpoint para cálculo de Performance."""
from fastapi import APIRouter
from datetime import datetime
from typing import List

import pandas as pd
import numpy as np
import pickle
import sklearn

router = APIRouter(prefix="/performance")

def process_records(records: List):
    # Initialize an empty dictionary to store the volume count for each month
    volumes = {}

    # Loop through each record in the list
    for record in records:
        # Extract the reference date from the record and convert it to a datetime object
        reference_date_str = record.get("REF_DATE")
        reference_date = datetime.strptime(reference_date_str, "%Y-%m-%d %H:%M:%S%z")

        # Calculate the month and year as a string
        month = f"{reference_date.month:02d}"

        # Increment the volume count for the month
        volumes[month] = volumes.get(month, 0) + 1

    # Return the volumes dictionary as the response to the POST request
    return volumes

@router.post("/")
def evaluate_model(records: List):

    # Convert the list of records to a Pandas DataFrame
    df = pd.DataFrame.from_dict(records)

    # Replace null values with np.nan
    df.fillna(inplace=True, value=np.nan)

    # Load the pre-trained model
    #Usando Path estatico para lidar com erro, e preciso mudar em caso de teste em outra maquina
    with open("D:/Projects/APITEST/challenge-data-scientist/monitoring/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Evaluate the model and get the ROC AUC score
    y_pred = model.predict_proba(df)[:, 1]
    roc_auc_score = sklearn.metrics.roc_auc_score(df["TARGET"], y_pred)

    # Calculate the monthly volume counts using the process_records function
    monthly_volumes = process_records(records)

    # Return the ROC AUC score and the monthly volume counts
    return {"roc_auc_score": roc_auc_score, "monthly_volumes": monthly_volumes}