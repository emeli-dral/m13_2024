import pandas as pd
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import xgboost as xgb

best_params = {
    "max_depth":6,
    "reg_alpha":0.07465666333107646
}

def load_data(path):
    data = pd.read_parquet(path)
    return data

def process_dataframe(data):
    data.lpep_dropoff_datetime = pd.to_datetime(data.lpep_dropoff_datetime)
    data.lpep_pickup_datetime = pd.to_datetime(data.lpep_pickup_datetime)

    data['duration'] = data.lpep_dropoff_datetime - data.lpep_pickup_datetime
    data.duration = data.duration.apply(lambda td: td.total_seconds() / 60)
    data = data[(data.duration >= 1) & (data.duration <= 90)]
    
    data['PULocationID'].astype(str, copy=False)
    data['DOLocationID'].astype(str, copy=False)  
    return data

def prepare_dataset(data):
    num_features = ['trip_distance', 'extra', 'fare_amount']
    cat_features = ['PULocationID', 'DOLocationID']
    X = data[num_features + cat_features]
    y = data['duration']
    xgb_dataset = xgb.DMatrix(X, label=y)
    return xgb_dataset
    
def train_model(train, validation):   
    booster = xgb.train(
                params=best_params,
                dtrain=train,
                evals=[(validation, "validation")],
                num_boost_round=100,
                early_stopping_rounds=50  
            )
    return booster

def evaluate_model(booster, validation):
    y_pred = booster.predict(validation)
    rmse = root_mean_squared_error(validation.get_label(), y_pred)
    print(f"rmse: {rmse:.3f}")
    return rmse

def main():
    raw_train = load_data('data/green_tripdata_2024-03.parquet')
    raw_validation = load_data('data/green_tripdata_2024-04.parquet')
    processed_train = process_dataframe(raw_train)
    processed_validation = process_dataframe(raw_validation)
    train = prepare_dataset(processed_train)
    validation = prepare_dataset(processed_validation)
    booster = train_model(train, validation)
    evaluate_model(booster, validation)  

if __name__ == '__main__':
    main()