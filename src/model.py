import mlflow
import mlflow.sklearn

def log_model_to_mlflow(clf, metrics, X_train, run_name=None):
    """
    Log model, configuration, metrics, and features to MLflow.
    
    Parameters:
    clf (HistGradientBoostingClassifier): The fitted classifier
    metrics (dict): Dictionary containing metrics from get_metrics function
    X_train (pd.DataFrame): Training features used to fit the model
    run_name (str, optional): Name for the MLflow run
    """
    # experiment_id = mlflow.create_experiment("New_Experiment_Name")
    # mlflow.set_experiment(experiment_id)

    with mlflow.start_run(run_name=run_name):
        # Log model configuration
        config = {
            'learning_rate': clf.learning_rate,
            'max_iter': clf.max_iter,
            'max_depth': clf.max_depth,
            'max_leaf_nodes': clf.max_leaf_nodes,
            'min_samples_leaf': clf.min_samples_leaf,
            'l2_regularization': clf.l2_regularization,
            'max_bins': clf.max_bins
        }
        for key, value in config.items():
            mlflow.log_param(key, value)
        
        # Log overall metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("balanced_accuracy", metrics['balanced_accuracy'])
        mlflow.log_metric("avg_precision", metrics['avg_precision'])
        mlflow.log_metric("avg_recall", metrics['avg_recall'])
        mlflow.log_metric("avg_f1", metrics['avg_f1'])
        mlflow.log_metric("avg_auc", metrics['avg_auc'])
        
        # Log detailed metrics for each class
        for i, row in metrics['metrics_df'].iterrows():
            mlflow.log_metric(f"precision_class_{i}", row['Precision'])
            mlflow.log_metric(f"recall_class_{i}", row['Recall'])
            mlflow.log_metric(f"f1_score_class_{i}", row['F1-Score'])
            mlflow.log_metric(f"auc_class_{i}", row['AUC'])
        
        # Log the model
        mlflow.sklearn.log_model(clf, "model")
        
        # Log feature names
        mlflow.log_param("features", list(X_train.columns))


if __name__ == "__main__":
    from sklearn.ensemble import HistGradientBoostingClassifier
    from data_utils import load_train_test_set
    from metrics import get_metrics
    mlflow.set_tracking_uri('../mlruns')

    X_train, y_train, X_test, y_test = load_train_test_set()

    clf = HistGradientBoostingClassifier()
    clf.fit(X_train, y_train)
    classes=[0, 1, 2, 3, 4]

    metrics = get_metrics(y_test, clf.predict(X_test), classes, normalize='true')
    log_model_to_mlflow(clf, metrics, X_train, run_name="HistGradientBoost_DefaultExperiment")
