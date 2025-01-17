from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_metrics(y_true, y_pred, classes):
    accuracy = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=classes, average=None)
    
    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    
    metrics_df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nDetailed Classification Metrics:")
    print(metrics_df)

    return accuracy, metrics_df, conf_matrix

def plot_hypnogram(y_true, y_pred, size=500):

    time = np.arange(len(y_true[:size]))

    sleep_stages = {0: "Wake", 1: "N1 sleep", 2: "N2 sleep", 3: "N3 sleep", 4: "REM sleep"}

    plt.figure(figsize=(10, 5))
    plt.plot(time, y_pred[:size], drawstyle='steps-post', color='b')
    plt.plot(time, y_true[:size], drawstyle='steps-post', color='r')

    plt.yticks(np.arange(len(sleep_stages)), list(sleep_stages.values()))
    plt.xlabel('Intervals')
    plt.ylabel('Sleep Stage')
    plt.title('Hypnogram')

    plt.grid(True)
    plt.show()
