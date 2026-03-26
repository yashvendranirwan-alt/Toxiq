import pandas as pd
import xgboost as xgb
from data_pipeline import load_data, preprocess_data
from model_trainer import ModelTrainer

def main():
    # Load Tox21 dataset
    data = load_data('path/to/tox21_dataset.csv')  # Adjust the path accordingly

    # Preprocess data
    X, y = preprocess_data(data)

    # Set up XGBoost parameters
    params = {
        'objective': 'multi:softprob',
        'num_class': 12,  # Update this based on the number of classes in your Tox21 dataset
        'max_depth': 6,
        'eta': 0.1,
        'eval_metric': 'mlogloss'
    }

    # Train model
    model_trainer = ModelTrainer(model=xgb.XGBClassifier(**params))
    model_trainer.fit(X, y)

    # Save the model for future use
    model_trainer.save_model('tox21_xgboost_model.json')

if __name__ == '__main__':
    main()