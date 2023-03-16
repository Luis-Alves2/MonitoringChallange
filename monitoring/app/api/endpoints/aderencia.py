"""Endpoint para cálculo de aderência."""
from fastapi import APIRouter
import pandas as pd
from scipy.stats import ks_2samp
import pickle
import gzip

import os

router = APIRouter(prefix="/aderencia")

@router.post("/")
async def compare_scores(file_path: str):

    new_input_path = os.path.abspath(file_path)
    # Load the pre-trained model
    #Path estatico pode ser interessante mudar em caso de teste em outra maquina
    with open("D:/Projects/APITEST/challenge-data-scientist/monitoring/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Read the input dataset
    with gzip.open(new_input_path, "rb") as f:
        input_data = pd.read_csv(f)

    # Score the input dataset
    if 'TARGET' in input_data.columns:
        input_data["score"] = model.predict_proba(input_data.drop("TARGET", axis=1))[:, 1]
    else:
        input_data["score"] = model.predict_proba(input_data)[:, 1]

    # Read the test dataset used for modeling
    #Path estatico pode ser interessante mudar em caso de teste em outra maquina
    with gzip.open("D:/Projects/APITEST/challenge-data-scientist/datasets/credit_01/test.gz", "rb") as f:
        test_data = pd.read_csv(f, low_memory=False)

    # Calculate the KS statistic between the input dataset and the test dataset
    ks_statistic = ks_2samp(input_data["score"], test_data["TARGET"]).statistic

    # Return the KS statistic as the response to the POST request
    return {"ks_statistic": ks_statistic}