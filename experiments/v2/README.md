# V2 Experiments Directory

This folder holds experimental artifacts, trained model checkpoints, and evaluation metrics for each candidate classifier under the V2 benchmark suite:

```
experiments/v2/
├── rf/         # Random Forest model checkpoints, params, and logs
├── svm/        # SVM model checkpoints, scalers, and logs
├── xgboost/    # XGBoost model checkpoints and training logs
└── knn/        # KNN model checkpoints, scalers, and logs
```

## Structure per Experiment
- `model.joblib`: Serialized classifier and fitted scaler.
- `metrics.json`: Accuracy, Precision, Recall, and F1-score computed by Person 2's evaluation pipeline.
- `params.json`: Hyperparameter configurations used during training.
