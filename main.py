import pandas as pd
import os
import logging
import gc
import joblib
import json
from numerapi import NumerAPI
from lightgbm import LGBMRegressor
from pathlib import Path

if not 'NUMERAI_PUBLIC_ID' in os.environ:
    raise Exception('missing NUMERAI_PUBLIC_ID')
if not 'NUMERAI_SECRET_KEY' in os.environ:
    raise Exception('missing NUMERAI_SECRET_KEY')

TRAINED_MODEL_PREFIX = './trained_model'
MODEL_ID = 'e382072c-6c4f-4220-a908-a467848a1362'

napi = NumerAPI()
current_round = napi.get_current_round()

model_name = TRAINED_MODEL_PREFIX
if MODEL_ID:
    model_name += f"_{MODEL_ID}"


def download_data():
    print('Downloading dataset files...')

    Path("./v4").mkdir(parents=False, exist_ok=True)
    napi.download_dataset("v4/train.parquet")
    napi.download_dataset("v4/live.parquet", f"v4/live_{current_round}.parquet")
    napi.download_dataset("v4/features.json")

    live_data = pd.read_parquet(f'v4/live_{current_round}.parquet')

    with open("v4/features.json", "r") as f:
        feature_metadata = json.load(f)
    # features = list(feature_metadata["feature_stats"].keys()) # get all the features
    features = feature_metadata["feature_sets"]["small"] # get the small feature set

    return features, live_data

def train(features):
    print('Training model...')

    model_name = 'numerai_model'

    if os.path.exists(model_name):
        logging.info('loading existing trained model')
        model = joblib.load(model_name)
        return model

    model = LGBMRegressor()

    training_data = pd.read_parquet('v4/train.parquet')

    logging.info('training model')
    model.fit(
        training_data[features],
        training_data['target']
    )

    logging.info('saving model')
    joblib.dump(model, model_name)

    gc.collect()

    return model

def predict(model, live_data, features):
    print('Making a prediction...')


    live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, features])

    live_data["prediction"] = live_data[f"preds_{model_name}"].rank(pct=True)
    logging.info(f'Live predictions and ranked')

    gc.collect()

    return live_data

def submit(live_data):
    print('Submitting...')

    predict_output_path = f"/tmp/live_predictions_{current_round}.csv"

    #make new dataframe with only the index (contains ids)
    predictions_df = live_data.index.to_frame()
    #copy predictions into new dataframe
    predictions_df["prediction"] = live_data["prediction"].copy()
    predictions_df.to_csv(predict_output_path, index=False)

    print(f'submitting {predict_output_path}')
    napi.upload_predictions(predict_output_path, model_id=MODEL_ID)
    print('submission complete!')

    # "garbage collection" (gc) gets rid of unused data and frees up memory
    gc.collect()



def main():
    """ Download, train, predict and submit for this model """

    features, live_data = download_data()
    # "garbage collection" (gc) gets rid of unused data and frees up memory
    gc.collect()
    model = train(features)
    # "garbage collection" (gc) gets rid of unused data and frees up memory
    gc.collect()
    new_live_data = predict(
        model,
        live_data,
        features
    )
    # "garbage collection" (gc) gets rid of unused data and frees up memory
    gc.collect()

    submit(new_live_data)

    # "garbage collection" (gc) gets rid of unused data and frees up memory
    gc.collect()

if __name__ == '__main__':
    main()
