import shap
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

def create_explanations(clf: ClassifierMixin, X: pd.DataFrame, fname: str) -> None:
    """
    Creates SHAP (SHapley Additive exPlanations) values for a given classifier and dataset, 
    and saves the importance of each feature as a CSV file and the SHAP values as a compressed NPZ file.
    
    :param clf: The trained classifier (must implement the fit and predict methods, 
                typically a decision tree-based model such as XGBoost or LightGBM).
    :param X: The input data used for explanation, expected to be a pandas DataFrame.
    :param fname: The base filename (without extension) for saving the explanation outputs.
    
    :return: None
    
    The function computes SHAP values for each feature and summarizes their importance by
    calculating the average absolute SHAP values across all samples and stages. It exports
    the results as both a compressed NPZ file containing SHAP values and a CSV file with 
    the sorted feature importance values.
    For reference see: https://github.com/raphaelvallat/yasa_classifier/blob/master/05_SHAP_importance.py
    """
    
    # Calculate SHAP feature importance - limit the number of trees for speed
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X, tree_limit=50)
    
    # Sum absolute values across all stages and average across all samples
    shap_sum = np.abs(shap_values).sum(axis=0).mean(axis=0)
    df_shap = pd.Series(shap_sum, index=X.columns.tolist(), name="Importance")
    df_shap.sort_values(ascending=False, inplace=True)
    df_shap.index.name = 'Features'

    # Export results
    np.savez_compressed(fname + ".npz", shap_values=shap_values)
    df_shap.to_csv(fname + ".csv")

    # Disabled: plot (commented out plotting code)
    # from matplotlib import colors
    # cmap_stages = ['#99d7f1', '#009DDC', 'xkcd:twilight blue',
    #                'xkcd:rich purple', 'xkcd:sunflower']
    # cmap = colors.ListedColormap(np.array(cmap_stages)[class_inds])
    # class_inds = np.argsort(
    #   [-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
    # shap.summary_plot(shap_values, X, plot_type='bar', max_display=15,
    #                   color=cmap, class_names=clf.classes_)
