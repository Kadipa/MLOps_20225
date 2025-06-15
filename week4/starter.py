#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import pandas as pd
import numpy as np

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, required=True)
parser.add_argument('--month', type=int, required=True)
args = parser.parse_args()

year = args.year
month = args.month

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

# Load model + DictVectorizer from model.bin already included in base image
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
df = read_data(input_file)

dicts = df[categorical].to_dict(orient='records')
X = dv.transform(dicts)
y_pred = model.predict(X)

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred

output_file = f'predictions_{year:04d}-{month:02d}.parquet'
df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)

print("Mean predicted duration:", round(np.mean(y_pred), 2))
