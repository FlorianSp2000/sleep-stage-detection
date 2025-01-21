from sklearn.inspection import permutation_importance
import pandas as pd

def get_permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42):
    # Calculate permutation importance
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    
    # Create and sort feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': result.importances_mean
    }).sort_values('importance', ascending=False)
    
    return feature_importance

if __name__ == "__main__":
    from sklearn.ensemble import HistGradientBoostingClassifier
    from data_utils import load_train_test_set
    
    X_train, X_test, y_train, y_test = load_train_test_set()
    model = HistGradientBoostingClassifier().fit(X_train, y_train)
    importances = get_permutation_importance(model, X_test, y_test)
    print(importances)
 


