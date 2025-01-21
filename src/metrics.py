import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

def get_metrics(y_true, y_pred, classes, normalize=None):
    metric_dict = {}
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=classes, average=None)
    
    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes, normalize=normalize)
    
    # Calculate AUC (AUROC) for each class
    y_true_bin = label_binarize(y_true, classes=classes)
    y_pred_proba = label_binarize(y_pred, classes=classes)  # Assuming y_pred is class labels, not probabilities
    auc_scores = []
    for i in range(len(classes)):
        auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
        auc_scores.append(auc)
    
    metrics_df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support,
        'AUC': auc_scores
    })
    
    # Calculate averages
    avg_precision = metrics_df['Precision'].mean()
    avg_recall = metrics_df['Recall'].mean()
    avg_f1 = metrics_df['F1-Score'].mean()
    avg_auc = metrics_df['AUC'].mean()
    
    metric_dict['accuracy'] = accuracy
    metric_dict['balanced_accuracy'] = balanced_accuracy
    metric_dict['metrics_df'] = metrics_df
    metric_dict['conf_matrix'] = conf_matrix

    metric_dict['avg_precision'] = avg_precision
    metric_dict['avg_recall'] = avg_recall
    metric_dict['avg_f1'] = avg_f1
    metric_dict['avg_auc'] = avg_auc

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}\n")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nDetailed Classification Metrics:")
    print(metrics_df)
    print("\nAverage Metrics:")
    print(f"Avg Precision: {avg_precision:.4f}")
    print(f"Avg Recall: {avg_recall:.4f}")
    print(f"Avg F1-Score: {avg_f1:.4f}")
    print(f"Avg AUC: {avg_auc:.4f}")

    return metric_dict
