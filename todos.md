### TODOS:

0. EDA
- Loot at pure time series plots
- Frequency domain? (+Wavelet)
- Differences between patients in terms of sleep stage distribution e.g.

1. Feature Engineering
- Adapt Static Features per epoch from Yasa: Measure Feature Importance (VIF, use model?, Forward/Backward Elimination?)
- Time Series Feature: Use Lags from previous intervals

2. Modeling
- Class Imbalance --> Weights
- Hyperparameter tuning of XGB model
- Try specialized models for each class

3. XAI
- Use SHAP/LIME also to get gist of which features are importance, where model is failing

4. Miscellaneous
- Use Model Logging tool such as: mlflow, W&B or tensorboard? to track results on the way
- Read Papers to get ideas on other features and modeling